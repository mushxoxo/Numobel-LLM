import pytest

_hmod = pytest.importorskip("app.validators.hallucination")
_rmod = pytest.importorskip("app.validators.response")
validate_response = _hmod.validate_response
initialize_validator = _hmod.initialize_validator
ValidationResult = _rmod.ValidationResult

ALL_BRANDS = ["Rubio Monocoat", "Nuacoustics", "Nutoy", "Nupanel", "Nuwork"]
ALL_TOKENS = {"Rubio", "Monocoat", "Nuacoustics", "Nutoy", "Nupanel", "Nuwork"}


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
