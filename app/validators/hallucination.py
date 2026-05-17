"""Post-generation hallucination prevention validator.

Detects unauthorized brand names in LLM output and routes to
severity-appropriate fallback content. Zero I/O on the hot path —
AUTHORIZED_BRANDS is populated at startup by initialize_validator().
"""
import re

from app.config import HALLUCINATION_SEVERE_PATTERNS, HALLUCINATION_RECOVERABLE_PATTERNS
from app.log import get_logger
from app.validators.response import ValidationResult

__all__ = ["validate_response", "initialize_validator", "AUTHORIZED_BRANDS", "KNOWN_SAFE_TERMS"]

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


def initialize_validator(brands: list[str]) -> None:
    """Populate AUTHORIZED_BRANDS from brand names. Idempotent — clears first."""
    global AUTHORIZED_BRANDS, _AUTHORIZED_BRAND_NAMES
    AUTHORIZED_BRANDS.clear()
    _AUTHORIZED_BRAND_NAMES.clear()
    _AUTHORIZED_BRAND_NAMES.extend(brands)
    for brand in brands:
        tokens = brand.split()
        for token in tokens:
            AUTHORIZED_BRANDS.add(token)
        AUTHORIZED_BRANDS.add(brand)
    log.info("HALLUCINATION | validator initialized with %d brands: %s", len(brands), brands)


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
    hits = context_data.get("hits", [])
    if hits:
        meta  = hits[0].get("metadata", {})
        name  = meta.get("name")
        brand = meta.get("brand")
        if name and brand:
            return f"{name} is a {brand} product offered by Numobel."
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
