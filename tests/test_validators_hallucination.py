import pytest
import re

_hmod = pytest.importorskip("app.validators.hallucination")
_rmod = pytest.importorskip("app.validators.response")
validate_response = _hmod.validate_response
initialize_validator = _hmod.initialize_validator
ValidationResult = _rmod.ValidationResult

ALL_BRANDS = ["Rubio Monocoat", "Nuacoustics", "Nutoy", "Nupanel", "Nuwork"]
ALL_TOKENS = {"Rubio", "Monocoat", "Nuacoustics", "Nutoy", "Nupanel", "Nuwork"}

NUTOY_PRODUCTS = [
    "Waldorf Stacker",
    "Building Block Series",
    "Poplar Wood Set",
    "On Wheels Collection",
]

NUTOY_PRODUCTS_HYPHENATED = [
    "Nutoy-On Wheels-Rabbit",
    "Nutoy-On Wheels-Swan-Black",
    "Nutoy-On Wheels-Duck",
]

# Regex mirroring webhook._ENTITY_TOKEN_RE — used by HALT-FP-01 test
_ENTITY_TOKEN_RE = re.compile(r'\b[A-Z][a-z]{2,}(?:[A-Z][a-z]*)?\b')


def _vr(content, user_query="hi", allowed_entities=None):
    initialize_validator(ALL_BRANDS)
    return validate_response(
        content=content,
        user_query=user_query,
        allowed_entities=allowed_entities or set(),
        response_plan={"message_type": "text"},
        context_data={"history": [], "hits": []},
    )


def test_severe_violation():
    result = _vr("We also offer Berger products")
    assert result.valid is False
    assert result.severity == "severe"
    assert result.recovery_strategy == "use_generic_fallback"
    assert result.fallback_content is not None


def test_recoverable_violation():
    result = _vr("Rubio Monocoat is similar to Asian Paints")
    assert result.valid is False
    assert result.severity == "recoverable"
    assert result.recovery_strategy == "use_context_fallback"


def test_user_introduced_entity():
    initialize_validator(ALL_BRANDS)
    result = validate_response(
        content="Thanks for your interest in Rubio Monocoat from Asian Paints",
        user_query="I am from Asian Paints and want Rubio Monocoat",
        allowed_entities=set(),
        response_plan={"message_type": "text"},
        context_data={"history": [], "hits": []},
    )
    assert result.valid is True


def test_authorized_brands_pass():
    result = _vr("Nutoy and Nuacoustics are great products from Numobel")
    assert result.valid is True


def test_no_capitalized_tokens():
    result = _vr("this product works well for flooring")
    assert result.valid is True


def test_safe_terms_pass():
    result = _vr("The product is good for wooden surfaces")
    assert result.valid is True


# ─── Regression tests for product-name false-positive bug ────────────────────

def test_initialize_validator_accepts_product_names():
    """initialize_validator(brands, product_names) tokenizes product names into AUTHORIZED_BRANDS."""
    initialize_validator(ALL_BRANDS, NUTOY_PRODUCTS)
    ab = _hmod.AUTHORIZED_BRANDS
    # Each word of each product name should be in the set
    assert "Waldorf" in ab
    assert "Building" in ab
    assert "Poplar" in ab
    assert "Wheels" in ab
    assert "Collection" in ab


def test_product_name_tokens_not_flagged_when_initialized_with_products():
    """HALLUCINATION regression: product names loaded at startup must not be flagged.

    Scenario: 'on wheels series' query → LLM responds mentioning Waldorf, Building,
    Poplar — all real Nutoy product names. With product_names passed to
    initialize_validator they should be in AUTHORIZED_BRANDS and pass validation.
    """
    initialize_validator(ALL_BRANDS, NUTOY_PRODUCTS)
    result = validate_response(
        content=(
            "Nutoy offers several wooden toy lines. The Waldorf Stacker and "
            "Building Block Series are popular choices. The Poplar Wood Set is "
            "also available for younger children."
        ),
        user_query="on wheels series",
        allowed_entities=set(),
        response_plan={"message_type": "text"},
        context_data={"history": [], "hits": []},
    )
    assert result.valid is True, (
        "HALLUCINATION false positive: real product name tokens were flagged. "
        f"violations={result.violations}"
    )


def test_product_name_tokens_still_flagged_without_product_init():
    """Baseline: without product_names, unknown capitalized tokens ARE flagged.

    This test documents the pre-fix behaviour so we know the guard works.
    """
    initialize_validator(ALL_BRANDS)  # no product_names
    result = validate_response(
        content="The Waldorf Stacker is a great Nutoy product.",
        user_query="nutoy toys",
        allowed_entities=set(),
        response_plan={"message_type": "text"},
        context_data={"history": [], "hits": []},
    )
    # "Waldorf" and "Stacker" are not in AUTHORIZED_BRANDS or KNOWN_SAFE_TERMS
    assert result.valid is False
    assert any(t in result.violations for t in ("Waldorf", "Stacker"))


def test_initialize_validator_idempotent_with_product_names():
    """initialize_validator clears and repopulates on every call — no accumulation."""
    initialize_validator(ALL_BRANDS, NUTOY_PRODUCTS)
    first_count = len(_hmod.AUTHORIZED_BRANDS)

    initialize_validator(ALL_BRANDS, NUTOY_PRODUCTS)
    second_count = len(_hmod.AUTHORIZED_BRANDS)

    assert first_count == second_count


def test_initialize_validator_product_names_default_none():
    """initialize_validator(brands) with no product_names arg is backwards-compatible."""
    initialize_validator(ALL_BRANDS)  # must not raise
    assert "Nutoy" in _hmod.AUTHORIZED_BRANDS
    assert "Waldorf" not in _hmod.AUTHORIZED_BRANDS


def test_hit_entity_tokens_added_to_allowed_entities():
    """allowed_entities from hit metadata supplements AUTHORIZED_BRANDS in validate_response.

    Simulates the webhook passing _hit_entity_tokens(hits) as allowed_entities.
    All capitalized tokens appearing in the response must be present in the
    combined allowed set (AUTHORIZED_BRANDS | allowed_entities | user entities).
    """
    initialize_validator(ALL_BRANDS)  # no product names at startup

    # Tokens as _hit_entity_tokens() would extract them from hit metadata —
    # includes every word of "Waldorf Stacker" and "Poplar Wood Set"
    hit_tokens = {"Waldorf", "Stacker", "Poplar", "Wood", "Set"}

    result = validate_response(
        content="The Waldorf Stacker and Poplar Wood Set are great Nutoy products.",
        user_query="nutoy toys",
        allowed_entities=hit_tokens,
        response_plan={"message_type": "text"},
        context_data={"history": [], "hits": []},
    )
    assert result.valid is True, (
        "Hit-derived tokens passed as allowed_entities should suppress false positives. "
        f"violations={result.violations}"
    )


def test_hyphenated_product_names_tokenized():
    """HALT-13: DB product names use hyphens — 'Nutoy-On Wheels-Rabbit' must yield 'Rabbit'."""
    initialize_validator(ALL_BRANDS, NUTOY_PRODUCTS_HYPHENATED)
    ab = _hmod.AUTHORIZED_BRANDS
    # Each hyphen-separated segment must be an individual authorized token
    assert "Rabbit" in ab, "Rabbit must be a standalone token from 'Nutoy-On Wheels-Rabbit'"
    assert "Swan" in ab
    assert "Duck" in ab
    assert "Wheels" in ab


def test_hyphenated_product_no_false_positive():
    """HALT-14: 'Rabbit' in LLM response must not flag when product is in catalogue."""
    initialize_validator(ALL_BRANDS, NUTOY_PRODUCTS_HYPHENATED)
    result = validate_response(
        content="The Nutoy On Wheels Rabbit is a charming wooden pull toy for toddlers.",
        user_query="on wheels rabbit",
        allowed_entities=set(),
        response_plan={"message_type": "text"},
        context_data={"history": [], "hits": []},
    )
    assert result.valid is True, (
        "Rabbit is a real Nutoy product token — must not be flagged. "
        f"violations={result.violations}"
    )


# ─── Bug 2 regression: context fallback uses product_name metadata key ────────

def test_context_fallback_uses_product_name_key():
    """Bug 2 regression: _build_context_fallback must use 'product_name' (ChromaDB metadata key).

    ChromaDB product chunks store the product name under 'product_name', not 'name'.
    Before the fix, _build_context_fallback looked for 'name' and found nothing,
    then fell back to the generic out-of-scope message even when hits were present.
    """
    initialize_validator(ALL_BRANDS)
    # Simulate the hit metadata shape that ChromaDB returns for product chunks
    hits = [{"metadata": {"product_name": "Nutoy Stacker Rainbow", "brand": "Nutoy"}}]
    result = validate_response(
        content="Rubio Monocoat Both products are available from Asian Paints",
        user_query="stacker rainbow",
        allowed_entities=set(),
        response_plan={"message_type": "media"},
        context_data={"history": [], "hits": hits},
    )
    assert result.valid is False
    # The fallback must use the product info from the hit, NOT the generic out-of-scope message
    assert "Nutoy Stacker Rainbow" in result.fallback_content, (
        "Bug 2 regression: context fallback did not use product_name from hit metadata. "
        f"Got fallback_content={result.fallback_content!r}"
    )
    assert "Could you ask about one of our brands" not in result.fallback_content, (
        "Bug 2 regression: context fallback fell through to generic message — "
        "product_name key was not found in hit metadata."
    )


def test_context_fallback_brand_only_when_no_product_name():
    """Context fallback uses brand-only sentence when hit has brand but no product_name."""
    initialize_validator(ALL_BRANDS)
    hits = [{"metadata": {"brand": "Nutoy"}}]  # no product_name field
    result = validate_response(
        content="Rubio Monocoat Both products are available from Asian Paints",
        user_query="nutoy",
        allowed_entities=set(),
        response_plan={"message_type": "text"},
        context_data={"history": [], "hits": hits},
    )
    assert result.valid is False
    assert "Nutoy" in result.fallback_content
    assert "Could you ask about one of our brands" not in result.fallback_content


def test_context_fallback_generic_when_hits_empty():
    """Context fallback uses generic message when hits list is empty."""
    initialize_validator(ALL_BRANDS)
    result = validate_response(
        content="Rubio Monocoat is similar to Asian Paints",
        user_query="rubio",
        allowed_entities=set(),
        response_plan={"message_type": "text"},
        context_data={"history": [], "hits": []},
    )
    assert result.valid is False
    assert "Could you ask about one of our brands" in result.fallback_content


# ─── HALT-FP-01: _hit_entity_tokens used wrong key 'document' instead of 'text' ─

def test_context_fallback_uses_chunk_text_when_available():
    """_build_context_fallback prefers chunk text excerpt over metadata product-name sentence.

    When hits contain a 'text' key (as returned by retrieve()), the fallback must
    extract an excerpt from the chunk text rather than falling back to the generic
    name+brand sentence. This provides a grounded, informative response.
    """
    initialize_validator(ALL_BRANDS)
    hits = [{
        "text": (
            "Numobel PET acoustic panels are designer choice for vibrant interior looks. "
            "The base material is Non-woven Polyester with Grooved and Embossed surface finishes."
        ),
        "metadata": {"product_name": "Numobel Acoustics-PET VG-1", "brand": "Nuacoustics"},
    }]
    result = validate_response(
        content="Rubio Monocoat is similar to Asian Paints",
        user_query="pet vg panels",
        allowed_entities=set(),
        response_plan={"message_type": "text"},
        context_data={"history": [], "hits": hits},
    )
    assert result.valid is False
    # Fallback must use the chunk text excerpt, not just the metadata product name sentence
    assert "Numobel PET acoustic panels" in result.fallback_content, (
        f"Expected chunk text excerpt in fallback. Got: {result.fallback_content!r}"
    )


def test_nuacoustics_material_terms_not_flagged_when_in_hit_text():
    """HALT-FP-01: material/finish terms in product chunk text must not be flagged.

    Root cause: _hit_entity_tokens() used hit.get('document') but retrieve() returns
    hits with key 'text'. This prevented Non, Polyester, Grooved, Embossed from entering
    allowed_entities even though they appear in the retrieved chunk text.

    This test simulates what the webhook does: passes allowed_entities from
    _hit_entity_tokens(hits) where hits use the 'text' key (the correct key).
    """
    initialize_validator(ALL_BRANDS)

    hits = [{
        "text": (
            "The base material is Non-woven, Polyester strands compressed together. "
            "There are many design variations - V Grooved and 3D Embossed."
        ),
        "metadata": {"product_name": "Numobel Acoustics-PET VG-1", "brand": "Nuacoustics"},
    }]

    # Replicate what webhook._hit_entity_tokens() does after the fix (uses 'text' key)
    tokens: set[str] = set()
    for hit in hits:
        meta = hit.get("metadata", {})
        for field in ("name", "brand", "product_line"):
            value = meta.get(field) or ""
            for token in value.split():
                tokens.add(token)
        doc = hit.get("text", "") or hit.get("document", "") or ""
        tokens.update(_ENTITY_TOKEN_RE.findall(doc))

    # All four formerly-flagged tokens must now be in allowed_entities
    for term in ("Non", "Polyester", "Grooved", "Embossed"):
        assert term in tokens, (
            f"{term!r} must be extracted from hit 'text' into allowed_entities. "
            f"Got tokens={sorted(tokens)}"
        )

    result = validate_response(
        content=(
            "The PET VG panels feature Non-woven Polyester material with Grooved and Embossed "
            "surface finish options."
        ),
        user_query="what is the difference between the two",
        allowed_entities=tokens,
        response_plan={"message_type": "media"},
        context_data={"history": [], "hits": hits},
    )
    assert result.valid is True, (
        "HALT-FP-01 regression: material/finish terms from chunk text were still flagged. "
        f"violations={result.violations}"
    )
