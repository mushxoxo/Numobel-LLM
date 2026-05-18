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

# KNOWN_SAFE_TERMS — common English Title-case words that legitimately appear in
# catalog answers (sentence starters, section headings, color/material vocabulary).
# Keeping a generous whitelist here avoids false-positive hallucination flags for
# domain vocabulary like "Colors", "Reds", "Slotted", "Grooved" etc.
KNOWN_SAFE_TERMS: set[str] = {
    # Pronouns and discourse markers
    "The", "This", "That", "These", "Those",
    "A", "An", "And", "Or", "But", "For", "With", "From", "About",
    "I", "You", "We", "Our", "Your", "My", "Its", "It",
    # Numobel-internal
    "Numobel", "Indian", "India",
    # Common sentence starters and discourse words
    "Thanks", "Thank", "Sure", "Yes", "No", "Please", "Sorry",
    "Hello", "Hi", "Hey", "Welcome",
    "Here", "There", "Also", "Note", "Great", "Good", "Best",
    "Currently", "Available", "Products", "Product",
    "Specific", "Specifically", "Each", "Every", "Both", "All", "Some",
    "Many", "Most", "Few", "Several", "Various", "Multiple", "Other", "Another",
    "When", "Where", "What", "Why", "How", "Which", "Who",
    "If", "Then", "Else", "Such", "While", "During", "After", "Before",
    "Okay",
    # Catalog vocabulary — sections, headings, units
    "Colors", "Color", "Sizes", "Size", "Price", "Prices",
    "Description", "Specifications", "Specification", "Features", "Feature",
    "Brand", "Brands", "Line", "Lines", "Series", "Range", "Variant", "Variants",
    "Catalogue", "Catalog",
    # Color family names that appear as variant labels
    "Red", "Reds", "Blue", "Blues", "Green", "Greens",
    "Yellow", "Yellows", "Orange", "Oranges", "Purple", "Purples",
    "Pink", "Pinks", "Brown", "Browns", "Black", "Blacks", "White", "Whites",
    "Grey", "Greys", "Gray", "Grays", "Beige", "Cream", "Ivory",
    "Walnut", "Oak", "Teak", "Maple", "Pine", "Cherry", "Mahogany",
    "Natural", "Neutral", "Pastel", "Vibrant", "Matte", "Glossy",
    # Material / surface / design descriptors
    "Wood", "Wooden", "Metal", "Plastic", "Polyester", "Acrylic", "Fabric",
    "Acoustic", "Acoustics", "Sound", "Soundproof",
    "Panel", "Panels", "Sheet", "Sheets", "Board", "Boards",
    "Slotted", "Grooved", "Perforated", "Embossed", "Smooth", "Textured",
    "Non", "Woven", "Compressed", "Layered",
    "Design", "Designer", "Designs", "Style", "Styles",
    "Interior", "Exterior", "Indoor", "Outdoor",
    "Standard", "Premium", "Basic", "Custom", "Special",
    "Performance", "Quality", "Durability",
    # Common verbs/adjectives that sometimes capitalize
    "Use", "Used", "Made", "Built", "Crafted", "Includes", "Including",
    "Offers", "Offering", "Provides", "Comes",
    "Designed", "Engineered", "Manufactured",
    # Misc safe
    "Stunning", "Beautiful", "Elegant", "Modern", "Classic", "Traditional",
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


# Header prefixes used by product_to_text() — chunks start with these label lines.
# When building a context fallback we skip them so the user sees prose, not metadata.
_CHUNK_HEADER_PREFIXES = (
    "Product:", "Brand:", "Product Line:", "Specifications:",
    "Price:", "Available Colors:", "Available Sizes:", "Weight:", "Keywords:",
)


def _strip_chunk_headers(chunk_text: str) -> str:
    """Remove metadata header lines from a product chunk, keeping only prose.

    Product chunks built by product_to_text() begin with lines like
    'Product: X', 'Brand: Y', 'Product Line: Z', 'Description: ...'.
    For a user-facing fallback we want the Description prose, not the labels.
    """
    lines = chunk_text.splitlines()
    cleaned: list[str] = []
    for line in lines:
        stripped = line.strip()
        if not stripped:
            continue
        if stripped.startswith("Description:"):
            cleaned.append(stripped[len("Description:"):].strip())
            continue
        if any(stripped.startswith(prefix) for prefix in _CHUNK_HEADER_PREFIXES):
            continue
        cleaned.append(stripped)
    return " ".join(cleaned).strip()


def _build_context_fallback(context_data: dict) -> str:
    """Build a factual fallback from the top retrieved hit.

    Preference order:
    1. Description prose from the chunk text (headers stripped).
    2. Product name + brand from metadata.
    3. Generic out-of-scope message.
    """
    hits = context_data.get("hits", [])
    if hits:
        chunk_text = hits[0].get("text", "") or hits[0].get("document", "") or ""
        if chunk_text:
            prose = _strip_chunk_headers(chunk_text) or chunk_text
            # First 1–2 sentences up to ~240 chars — informative but concise.
            sentences = [s.strip() for s in prose.split(".") if s.strip()]
            excerpt = ""
            for s in sentences:
                candidate = (excerpt + ". " + s) if excerpt else s
                if len(candidate) > 240:
                    break
                excerpt = candidate
            if not excerpt and sentences:
                excerpt = sentences[0][:240]
            if excerpt:
                return f"{excerpt}. (Based on our product catalogue)"
        meta  = hits[0].get("metadata", {})
        name  = meta.get("product_name") or meta.get("name")
        brand = meta.get("brand")
        if name and brand:
            return f"{name} is a {brand} product offered by Numobel."
        if brand:
            return f"This is a {brand} product offered by Numobel."
    return _build_generic_fallback()


def _has_consecutive_unknown(content: str, allowed: set[str]) -> list[str]:
    """Return any runs of 2+ adjacent unknown Title-case tokens.

    Used as the strict-mode trigger when no severe/recoverable pattern matched.
    Isolated Title-case words (sentence starters, color names, single descriptors)
    are NOT flagged. Only adjacent runs like 'Asian Paints' or 'Waldorf Stacker'
    are returned — these look like proper-noun product/brand phrases.

    Adjacency is determined by walking _ENTITY_RE matches in order and checking
    that the gap between the previous match end and the next match start contains
    only whitespace (allowing for hyphens within a single token via the regex).
    """
    suspicious: list[str] = []
    matches = list(_ENTITY_RE.finditer(content))
    run: list[str] = []
    prev_end = -1
    for m in matches:
        token = m.group(0)
        token_unknown = token not in allowed
        # Adjacency: only whitespace between previous match end and this match start.
        gap = content[prev_end:m.start()] if prev_end >= 0 else ""
        adjacent = bool(gap) and gap.strip() == ""
        if token_unknown and (not run or adjacent):
            run.append(token)
        else:
            if len(run) >= 2:
                suspicious.append(" ".join(run))
            run = [token] if token_unknown else []
        prev_end = m.end()
    if len(run) >= 2:
        suspicious.append(" ".join(run))
    return suspicious


def validate_response(
    content: str,
    user_query: str,
    allowed_entities: set[str],
    response_plan: dict,
    context_data: dict,
) -> ValidationResult:
    """Validate LLM response for unauthorized brand mentions.

    Two-stage gate:
      1. If a severe / recoverable competitor pattern matches AND there are
         unknown capitalized tokens → flag with severity = severe / recoverable.
      2. Otherwise, only flag when 2+ ADJACENT unknown Title-case tokens form a
         proper-noun-like phrase (e.g. 'Asian Paints', 'Waldorf Stacker').
         Isolated capitalizations (sentence starters, color names, descriptors)
         pass — eliminates false positives on common English vocabulary.

    Returns ValidationResult(valid=True) if clean, or valid=False with severity
    and fallback_content if unauthorized entities are detected.
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

    severe_hit      = any(pattern.search(content) for pattern in _SEVERE_RE)
    recoverable_hit = any(pattern.search(content) for pattern in _RECOVERABLE_RE)

    if severe_hit:
        severity          = "severe"
        recovery_strategy = "use_generic_fallback"
        fallback_content  = _build_generic_fallback()
    elif recoverable_hit:
        severity          = "recoverable"
        recovery_strategy = "use_context_fallback"
        fallback_content  = _build_context_fallback(context_data)
    else:
        # No competitor-signal pattern matched. Only flag when unknown tokens
        # appear ADJACENT in the text (proper-noun phrase shape).
        consecutive = _has_consecutive_unknown(content, combined_allowed)
        if not consecutive:
            log.debug(
                "HALLUCINATION | unauthorized=%s but no competitor pattern and no "
                "adjacent proper-noun phrase — passing through.",
                unauthorized[:5],
            )
            return ValidationResult(valid=True)
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
