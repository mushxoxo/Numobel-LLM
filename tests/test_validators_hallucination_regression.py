"""Regression tests for response-quality-regression debug session.

Three new behaviors:
  1. Isolated Title-case common-vocabulary words do NOT trigger the hallucination
     guard (no severe/recoverable competitor pattern, no adjacent unknown phrase).
  2. Adjacent unknown Title-case tokens DO trigger the guard even without a
     competitor pattern (so 'Waldorf Stacker' is still caught).
  3. Context fallback strips 'Product:'/'Brand:'/'Product Line:' header lines from
     product chunks and returns the Description prose instead of a metadata dump.
"""
import pytest

_hmod = pytest.importorskip("app.validators.hallucination")
_rmod = pytest.importorskip("app.validators.response")
validate_response = _hmod.validate_response
initialize_validator = _hmod.initialize_validator

ALL_BRANDS = ["Rubio Monocoat", "Nuacoustics", "Nutoy", "Nupanel", "Nuwork"]


def _vr(content, user_query="hi", allowed_entities=None, hits=None):
    initialize_validator(ALL_BRANDS)
    return validate_response(
        content=content,
        user_query=user_query,
        allowed_entities=allowed_entities or set(),
        response_plan={"message_type": "text"},
        context_data={"history": [], "hits": hits or []},
    )


# ─── Bug 1: isolated Title-case vocabulary must not be flagged ────────────────

def test_isolated_color_family_words_not_flagged():
    """'Colors', 'Reds', 'Blues', 'Greens' alone must not trip the guard."""
    result = _vr(
        "The acoustic panels come in beautiful Colors including Reds, Blues, and Greens."
    )
    assert result.valid is True, (
        f"isolated color-family words must not be flagged. violations={result.violations}"
    )


def test_isolated_descriptor_words_not_flagged():
    """'Specific', 'Slotted', 'Grooved', 'Embossed' alone must not trip the guard."""
    result = _vr(
        "The PET VG panels feature Slotted, Grooved, and Embossed surfaces "
        "for Specific acoustic performance."
    )
    assert result.valid is True, (
        f"isolated descriptor words must not be flagged. violations={result.violations}"
    )


def test_sentence_start_titlecase_not_flagged():
    """Sentence-initial common nouns like 'Specific' must not be flagged."""
    result = _vr(
        "Specific finishes are available for indoor use. Slotted variants exist."
    )
    assert result.valid is True


def test_unknown_token_isolated_without_competitor_pattern_passes():
    """A single unknown Title-case token with no severe/recoverable pattern passes."""
    result = _vr("Mahogany is a beautiful finish option.")
    assert result.valid is True


# ─── Bug 1 guard: adjacent unknown tokens still flagged ──────────────────────

def test_adjacent_unknown_tokens_still_flagged():
    """'Waldorf Stacker' (2 adjacent unknown Title-case tokens) IS flagged."""
    result = _vr("The Waldorf Stacker is a great Nutoy product.")
    assert result.valid is False
    assert any(t in result.violations for t in ("Waldorf", "Stacker"))


def test_adjacent_unknown_competitor_phrase_flagged():
    """'Asian Paints' (adjacent unknown phrase) is flagged even with 'similar to'
    pattern absent."""
    result = _vr("Numobel partners with Asian Paints in some markets.")
    assert result.valid is False


# ─── Bug 2: context fallback strips chunk header lines ───────────────────────

def test_context_fallback_skips_product_header_lines():
    """Context fallback returns Description prose, not 'Product:'/'Brand:' headers.

    Chunks built by product_to_text() start with header lines. The fallback must
    skip these and return the Description prose so the user sees useful content,
    not a raw metadata dump.
    """
    chunk = (
        "Product: Numobel acoustics-MDF Perforated\n"
        "Brand: Nuacoustics\n"
        "Product Line: MDF Perforated\n"
        "Description: Numobel Acoustics MDF acoustic panels are designer's choice "
        "for vibrant interior looks with scientifically engineered acoustic "
        "performance for any room"
    )
    hits = [{
        "text": chunk,
        "metadata": {
            "product_name": "Numobel acoustics-MDF Perforated",
            "brand": "Nuacoustics",
            "product_line": "MDF Perforated",
        },
    }]
    # Force the fallback path with a recoverable-pattern hit
    result = _vr(
        "Numobel panels are similar to Asian Paints offerings.",
        hits=hits,
    )
    assert result.valid is False
    fb = result.fallback_content
    # Header lines must not appear verbatim in the fallback
    assert "Product:" not in fb, f"Product: header leaked into fallback: {fb!r}"
    assert "Brand:" not in fb, f"Brand: header leaked into fallback: {fb!r}"
    assert "Product Line:" not in fb, f"Product Line: header leaked into fallback: {fb!r}"
    # The actual prose must be present
    assert "designer" in fb.lower() or "acoustic panels" in fb.lower(), (
        f"Description prose missing from fallback: {fb!r}"
    )


def test_context_fallback_handles_chunk_with_only_description():
    """If chunk text is already prose (no header lines), it is returned as-is."""
    hits = [{
        "text": "PET VG panels offer beautiful color families like Reds, Blues, Greens.",
        "metadata": {"product_name": "PET VG-1", "brand": "Nuacoustics"},
    }]
    result = _vr(
        "These panels are similar to Asian Paints products.",
        hits=hits,
    )
    assert result.valid is False
    assert "PET VG panels" in result.fallback_content or "Reds" in result.fallback_content
