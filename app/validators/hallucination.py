"""Post-generation hallucination prevention validator.

Detects unauthorized brand names in LLM output and routes to
severity-appropriate fallback content. Zero I/O on the hot path —
AUTHORIZED_BRANDS is populated at startup by initialize_validator().
"""
import re

from app.config import (
    HALLUCINATION_SEVERE_PATTERNS,
    HALLUCINATION_RECOVERABLE_PATTERNS,
    HALLUCINATION_IMAGE_URL_BLOCKLIST,
)
from app.log import get_logger
from app.validators.response import ValidationResult

__all__ = ["validate_response", "initialize_validator", "AUTHORIZED_BRANDS", "KNOWN_SAFE_TERMS", "is_blocked_image_url"]

log = get_logger()

# Module-level state — populated by initialize_validator() at startup
AUTHORIZED_BRANDS: set[str]    = set()
_AUTHORIZED_BRAND_NAMES: list[str] = []

KNOWN_SAFE_TERMS: set[str] = {
    "The", "This", "That", "These", "Those",
    "A", "An", "And", "Or", "But", "For", "With", "From", "About",
    "I", "You", "We", "Our", "Your", "My",
    "Numobel", "Indian", "India",
    # Common sentence starters and discourse words that are not brand names
    "Thanks", "Thank", "Sure", "Yes", "Please", "Sorry", "Hello", "Hi",
    "Here", "Also", "Note", "Please", "Great", "Good", "Best",
    "Currently", "Available", "Products", "Product",
}

_SEVERE_RE      = [re.compile(p, re.IGNORECASE) for p in HALLUCINATION_SEVERE_PATTERNS]
_RECOVERABLE_RE = [re.compile(p, re.IGNORECASE) for p in HALLUCINATION_RECOVERABLE_PATTERNS]
_ENTITY_RE      = re.compile(r'\b[A-Z][a-z]{2,}(?:[A-Z][a-z]*)?\b')


def is_blocked_image_url(url: str | None) -> bool:
    """Return True if url is a placeholder/hallucinated domain that must never reach users.

    Checks against HALLUCINATION_IMAGE_URL_BLOCKLIST from app.config.
    Case-insensitive substring match so sub-paths are also caught.
    """
    if not url:
        return False
    url_lower = url.lower()
    return any(blocked in url_lower for blocked in HALLUCINATION_IMAGE_URL_BLOCKLIST)


def initialize_validator(brands: list[str], product_names: list[str] | None = None) -> None:
    """Populate AUTHORIZED_BRANDS from brand and product names. Idempotent — clears first.

    Args:
        brands: Top-level brand names (e.g. ["Rubio Monocoat", "Nutoy"]).
        product_names: Individual product names from the catalogue
            (e.g. ["Waldorf", "Building Block Series", "Poplar"]). Each name
            is split into tokens so multi-word names contribute every word.
            Pass an empty list or None to skip (backwards-compatible).
    """
    global AUTHORIZED_BRANDS, _AUTHORIZED_BRAND_NAMES
    AUTHORIZED_BRANDS.clear()
    _AUTHORIZED_BRAND_NAMES.clear()
    _AUTHORIZED_BRAND_NAMES.extend(brands)

    def _add_name(name: str) -> None:
        # Split on whitespace and hyphens — DB product names use both as separators
        # e.g. "Nutoy-On Wheels-Rabbit" → ["Nutoy", "On", "Wheels", "Rabbit"]
        tokens = re.split(r'[\s\-]+', name)
        for token in tokens:
            if token:
                AUTHORIZED_BRANDS.add(token)
        AUTHORIZED_BRANDS.add(name)

    for brand in brands:
        _add_name(brand)

    for product in (product_names or []):
        _add_name(product)

    log.info(
        "HALLUCINATION | validator initialized with %d brands, %d products (%d total tokens)",
        len(brands), len(product_names or []), len(AUTHORIZED_BRANDS),
    )


def _extract_user_entities(user_query: str, history: list[dict] | None) -> set[str]:
    """Extract capitalized tokens from user messages to prevent false positives."""
    texts = [user_query or ""]
    for entry in (history or []):
        if entry.get("role") == "user":
            texts.append(entry.get("content", ""))
        elif "user_message" in entry:
            texts.append(entry["user_message"])
    combined = " ".join(texts)
    return set(_ENTITY_RE.findall(combined))


def _build_generic_fallback() -> str:
    brands_str = ", ".join(sorted(_AUTHORIZED_BRAND_NAMES))
    return (
        f"I can only provide information about Numobel products. "
        f"Could you ask about one of our brands: {brands_str}?"
    )


def _build_context_fallback(context_data: dict) -> str:
    """Build a factual fallback from the top retrieved hit.

    ChromaDB product chunks store the product name under 'product_name' (not 'name').
    Falls back to the generic out-of-scope message only when no usable metadata is found.
    """
    hits = context_data.get("hits", [])
    if hits:
        meta  = hits[0].get("metadata", {})
        # ChromaDB product metadata uses 'product_name'; approved QnA chunks use 'name'
        name  = meta.get("product_name") or meta.get("name")
        brand = meta.get("brand")
        if name and brand:
            return f"{name} is a {brand} product offered by Numobel."
        if brand:
            return f"This is a {brand} product offered by Numobel."
    return _build_generic_fallback()


def validate_response(
    content: str,
    user_query: str,
    allowed_entities: set[str],
    response_plan: dict,
    context_data: dict,
) -> ValidationResult:
    """Validate LLM response for unauthorized brand mentions.

    Returns ValidationResult(valid=True) if clean, or valid=False with
    severity and fallback_content if unauthorized entities are detected.
    response_plan is accepted for forward-compat but currently unused.
    """
    history = context_data.get("history")
    combined_allowed = (
        allowed_entities
        | AUTHORIZED_BRANDS
        | KNOWN_SAFE_TERMS
        | _extract_user_entities(user_query, history)
    )

    tokens       = _ENTITY_RE.findall(content)
    unauthorized = [t for t in tokens if t not in combined_allowed]

    if not unauthorized:
        return ValidationResult(valid=True)

    # Severity classification
    if any(pattern.search(content) for pattern in _SEVERE_RE):
        severity          = "severe"
        recovery_strategy = "use_generic_fallback"
        fallback_content  = _build_generic_fallback()
    elif any(pattern.search(content) for pattern in _RECOVERABLE_RE):
        severity          = "recoverable"
        recovery_strategy = "use_context_fallback"
        fallback_content  = _build_context_fallback(context_data)
    else:
        severity          = "recoverable"
        recovery_strategy = "use_context_fallback"
        fallback_content  = _build_context_fallback(context_data)

    log.warning(
        "HALLUCINATION | severity=%s tokens=%s strategy=%s",
        severity, unauthorized, recovery_strategy,
    )

    return ValidationResult(
        valid=False,
        severity=severity,
        violations=unauthorized,
        fallback_content=fallback_content,
        recovery_strategy=recovery_strategy,
    )
