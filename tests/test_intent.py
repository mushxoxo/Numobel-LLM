"""
Tests for app/intent.py — IntentEnum, Layer 1 rule classifier, Layer 2 centroid classifier.

Import app.intent inside fixtures/tests only — the module has module-level state (_centroids,
_intent_ready) that must be reset between tests via the autouse fixture.
"""

import pytest
import numpy as np
from unittest.mock import patch, MagicMock

PHONE = "919999999999"
SAMPLE_EMBEDDING = [0.1] * 1024  # mxbai-embed-large is 1024-dim


def _history_with_intent(intent_value: str) -> list[dict]:
    return [{"role": "user", "content": "rubio monocoat price", "intent": intent_value}]


@pytest.fixture(autouse=True)
def reset_intent_state(monkeypatch):
    import app.intent as intent_module
    monkeypatch.setattr(intent_module, "_centroids", {})
    monkeypatch.setattr(intent_module, "_intent_ready", False)
    monkeypatch.setattr(intent_module, "_entity_index_ready", False)
    monkeypatch.setattr(intent_module, "_entity_brand_names", frozenset())
    monkeypatch.setattr(intent_module, "_entity_brand_tokens", frozenset())
    monkeypatch.setattr(intent_module, "_entity_product_line_names", frozenset())
    monkeypatch.setattr(intent_module, "_entity_product_tokens", frozenset())
    yield
    monkeypatch.setattr(intent_module, "_centroids", {})
    monkeypatch.setattr(intent_module, "_intent_ready", False)
    monkeypatch.setattr(intent_module, "_entity_index_ready", False)
    monkeypatch.setattr(intent_module, "_entity_brand_names", frozenset())
    monkeypatch.setattr(intent_module, "_entity_brand_tokens", frozenset())
    monkeypatch.setattr(intent_module, "_entity_product_line_names", frozenset())
    monkeypatch.setattr(intent_module, "_entity_product_tokens", frozenset())


def test_classify_intent_returns_required_keys_INTENT01():
    """INTENT-01: classify_intent returns dict with intent, confidence, layer keys."""
    from app.intent import classify_intent, IntentEnum
    result = classify_intent("hello")
    assert set(result.keys()) >= {"intent", "confidence", "layer"}
    assert isinstance(result["intent"], IntentEnum)
    assert isinstance(result["confidence"], float)
    assert isinstance(result["layer"], str)


def test_greeting_rule_layer_INTENT02():
    """INTENT-02: Layer 1 classifies 'hi' as GREETING with layer='rule'."""
    from app.intent import classify_intent, IntentEnum
    result = classify_intent("hi")
    assert result == {"intent": IntentEnum.GREETING, "confidence": 1.0, "layer": "rule"}


def test_oos_rule_layer_INTENT02():
    """INTENT-02: Layer 1 classifies return intent as OUT_OF_SCOPE."""
    from app.intent import classify_intent, IntentEnum
    result = classify_intent("I want to return this product")
    assert result["intent"] == IntentEnum.OUT_OF_SCOPE
    assert result["layer"] == "rule"
    assert result["confidence"] == 1.0


def test_chitchat_rule_layer_INTENT02():
    """INTENT-02: Layer 1 classifies 'thanks' as CHITCHAT."""
    from app.intent import classify_intent, IntentEnum
    result = classify_intent("thanks")
    assert result == {"intent": IntentEnum.CHITCHAT, "confidence": 1.0, "layer": "rule"}


def test_greeting_anchored_not_fired_INTENT02():
    """INTENT-02: Anchored greeting regex must NOT fire on longer messages."""
    from app.intent import classify_intent, IntentEnum
    result = classify_intent("hi I need rubio monocoat pricing")
    assert result["intent"] != IntentEnum.GREETING


def test_embedding_layer_when_ready_INTENT03(monkeypatch):
    """INTENT-03: Layer 2 returns embedding layer when _intent_ready=True and centroid matches."""
    import app.intent as intent_module
    from app.intent import classify_intent, IntentEnum
    # Inject centroid with similarity 1.0 to SAMPLE_EMBEDDING
    monkeypatch.setattr(intent_module, "_intent_ready", True)
    monkeypatch.setattr(intent_module, "_centroids", {
        "general_qna": np.array(SAMPLE_EMBEDDING)
    })
    result = classify_intent("how do I apply rubio monocoat", embedding=SAMPLE_EMBEDDING)
    assert result["layer"] == "embedding"
    assert result["confidence"] >= 0.75


def test_fallback_when_not_ready_INTENT03():
    """INTENT-03: Returns GENERAL_QNA fallback when _intent_ready=False."""
    from app.intent import classify_intent, IntentEnum
    result = classify_intent("anything", embedding=SAMPLE_EMBEDDING)
    assert result == {"intent": IntentEnum.GENERAL_QNA, "confidence": 0.0, "layer": "fallback"}


def test_tentative_inherits_prior_intent_INTENT06(monkeypatch):
    """INTENT-06: Short message with tentative confidence inherits prior intent from history."""
    import app.intent as intent_module
    from app.intent import classify_intent, IntentEnum
    # Create a centroid that produces tentative similarity (~0.625) with SAMPLE_EMBEDDING
    # cosine([0.1]*1024, [0.2]*400 + [0.0]*624) = 8.0 / (3.2 * 4.0) = 0.625
    tentative_centroid = np.array([0.2] * 400 + [0.0] * 624)
    monkeypatch.setattr(intent_module, "_intent_ready", True)
    monkeypatch.setattr(intent_module, "_centroids", {
        "brand_deep_dive": tentative_centroid
    })
    result = classify_intent(
        "tell me more",  # short text, 3 words < 15, not in chitchat exact set
        embedding=SAMPLE_EMBEDDING,
        history=_history_with_intent("brand_deep_dive"),
    )
    assert result["intent"] == IntentEnum.BRAND_DEEP_DIVE
    assert result["layer"] in ("embedding_inherit", "embedding")


def test_tentative_no_inherit_without_history_INTENT06(monkeypatch):
    """INTENT-06: Tentative with no prior history does not inherit."""
    import app.intent as intent_module
    from app.intent import classify_intent, IntentEnum
    tentative_centroid = np.array([0.2] * 400 + [0.0] * 624)
    monkeypatch.setattr(intent_module, "_intent_ready", True)
    monkeypatch.setattr(intent_module, "_centroids", {
        "brand_deep_dive": tentative_centroid
    })
    result = classify_intent("tell me more", embedding=SAMPLE_EMBEDDING, history=[])
    # No prior intent in empty history — should return best match without inherit
    assert result["intent"] == IntentEnum.BRAND_DEEP_DIVE
    assert result["layer"] == "embedding"


def test_no_internal_embedding_call_when_provided_INTENT07():
    """INTENT-07: classify_intent does NOT call get_embedding() when embedding is provided."""
    from app.intent import classify_intent
    with patch("app.rag.get_embedding") as mock_emb:
        classify_intent("rubio monocoat price", embedding=SAMPLE_EMBEDDING)
        mock_emb.assert_not_called()


def test_compute_centroids_sets_ready_on_success(monkeypatch):
    """compute_centroids() sets _intent_ready=True on success."""
    import app.intent as intent_module
    with patch("app.rag.get_embedding", return_value=SAMPLE_EMBEDDING):
        from app.intent import compute_centroids
        compute_centroids()
    assert intent_module._intent_ready is True
    assert len(intent_module._centroids) == 8


def test_compute_centroids_graceful_on_failure(monkeypatch):
    """compute_centroids() logs warning and sets _intent_ready=False on failure."""
    import app.intent as intent_module
    with patch("app.rag.get_embedding", side_effect=RuntimeError("ollama down")):
        from app.intent import compute_centroids
        compute_centroids()  # must not raise
    assert intent_module._intent_ready is False


# ─── Regression tests for greeting-misclassification bug ─────────────────────

def test_greeting_prior_not_inherited_on_product_query_INTENT08(monkeypatch):
    """INTENT-08 regression: greeting prior must NOT be inherited for product queries.

    Scenario: user says 'hi' (classified as greeting), then sends 'tell me about nutoys'.
    The second message has tentative embedding score but must NOT inherit greeting intent.
    """
    import app.intent as intent_module
    from app.intent import classify_intent, IntentEnum

    # Centroid produces tentative similarity (~0.625) — below confidence threshold (0.75)
    # but above tentative threshold (0.60), triggering the inherit branch.
    tentative_centroid = np.array([0.2] * 400 + [0.0] * 624)
    monkeypatch.setattr(intent_module, "_intent_ready", True)
    monkeypatch.setattr(intent_module, "_centroids", {
        "brand_deep_dive": tentative_centroid
    })

    # Prior history shows greeting intent
    history = [{"role": "user", "content": "hi", "intent": "greeting"}]

    result = classify_intent(
        "tell me about nutoys",
        embedding=SAMPLE_EMBEDDING,
        history=history,
    )

    # Must NOT be greeting — greeting prior is non-inheritable
    assert result["intent"] != IntentEnum.GREETING, (
        "Bug regression: greeting prior was inherited onto a product query. "
        f"Got intent={result['intent']} layer={result['layer']}"
    )
    # Should resolve to the best embedding match (brand_deep_dive) instead
    assert result["intent"] == IntentEnum.BRAND_DEEP_DIVE


def test_chitchat_prior_not_inherited_INTENT08(monkeypatch):
    """INTENT-08 regression: chitchat prior must NOT be inherited."""
    import app.intent as intent_module
    from app.intent import classify_intent, IntentEnum

    tentative_centroid = np.array([0.2] * 400 + [0.0] * 624)
    monkeypatch.setattr(intent_module, "_intent_ready", True)
    monkeypatch.setattr(intent_module, "_centroids", {
        "brand_deep_dive": tentative_centroid
    })

    history = [{"role": "user", "content": "thanks", "intent": "chitchat"}]

    result = classify_intent(
        "tell me about nutoys",
        embedding=SAMPLE_EMBEDDING,
        history=history,
    )

    assert result["intent"] != IntentEnum.CHITCHAT
    assert result["intent"] == IntentEnum.BRAND_DEEP_DIVE


def test_out_of_scope_prior_not_inherited_INTENT08(monkeypatch):
    """INTENT-08 regression: out_of_scope prior must NOT be inherited."""
    import app.intent as intent_module
    from app.intent import classify_intent, IntentEnum

    tentative_centroid = np.array([0.2] * 400 + [0.0] * 624)
    monkeypatch.setattr(intent_module, "_intent_ready", True)
    monkeypatch.setattr(intent_module, "_centroids", {
        "brand_deep_dive": tentative_centroid
    })

    history = [{"role": "user", "content": "where is my order", "intent": "out_of_scope"}]

    result = classify_intent(
        "tell me about nutoys",
        embedding=SAMPLE_EMBEDDING,
        history=history,
    )

    assert result["intent"] != IntentEnum.OUT_OF_SCOPE
    assert result["intent"] == IntentEnum.BRAND_DEEP_DIVE


def test_no_embedding_returns_fallback_not_zero_layer_INTENT08(monkeypatch):
    """INTENT-08: classify_intent with no embedding skips Layer 2, returns fallback_no_embedding."""
    import app.intent as intent_module
    from app.intent import classify_intent, IntentEnum

    monkeypatch.setattr(intent_module, "_intent_ready", True)
    monkeypatch.setattr(intent_module, "_centroids", {
        "brand_deep_dive": np.array(SAMPLE_EMBEDDING)
    })

    # Passing embedding=None should NOT run Layer 2 with a zero vector
    result = classify_intent("tell me about nutoys", embedding=None)
    assert result["layer"] == "fallback_no_embedding"
    assert result["intent"] == IntentEnum.GENERAL_QNA


# ─── Regression tests for divergent-prior inherit bug (INTENT-09) ─────────────

def test_general_qna_prior_not_inherited_when_embedding_is_brand_deep_dive_INTENT09(monkeypatch):
    """INTENT-09 regression: general_qna prior must NOT override brand_deep_dive embedding match.

    Scenario: user previously asked a general question (general_qna prior), then sends a
    short follow-up with tentative embedding top-match = brand_deep_dive. The prior is
    inheritable (not in _NON_INHERITABLE) but diverges from the embedding best-match —
    the embedding should win. Uses a non-brand-name query so Layer 1 does not intercept.
    """
    import app.intent as intent_module
    from app.intent import classify_intent, IntentEnum

    # brand_deep_dive centroid: tentative similarity with SAMPLE_EMBEDDING (~0.625)
    # general_qna centroid: slightly lower similarity
    # cosine([0.1]*1024, [0.2]*400 + [0.0]*624) ≈ 0.625  (brand_deep_dive — best)
    # cosine([0.1]*1024, [0.1]*300 + [0.0]*724) ≈ 0.541  (general_qna — below best)
    brand_deep_dive_centroid = np.array([0.2] * 400 + [0.0] * 624)
    general_qna_centroid     = np.array([0.1] * 300 + [0.0] * 724)

    monkeypatch.setattr(intent_module, "_intent_ready", True)
    monkeypatch.setattr(intent_module, "_centroids", {
        "brand_deep_dive": brand_deep_dive_centroid,
        "general_qna":     general_qna_centroid,
    })

    # Prior from history is general_qna — inheritable but diverges from best embedding
    history = [{"role": "user", "content": "what products do you have", "intent": "general_qna"}]

    result = classify_intent(
        "what about it",  # 4 words, not a brand name — reaches Layer 2
        embedding=SAMPLE_EMBEDDING,
        history=history,
    )

    assert result["intent"] == IntentEnum.BRAND_DEEP_DIVE, (
        "INTENT-09 regression: general_qna prior overrode brand_deep_dive embedding match. "
        f"Got intent={result['intent'].value} layer={result['layer']}"
    )
    assert result["layer"] in ("embedding", "embedding_inherit")


def test_short_non_brand_query_resolves_to_best_embedding_INTENT09(monkeypatch):
    """INTENT-09: Short non-brand query with no prior resolves to best embedding match via Layer 2.

    No inherit branch at all (no history) — embedding alone should determine intent.
    Uses a non-brand-name query so Layer 1 does not intercept.
    """
    import app.intent as intent_module
    from app.intent import classify_intent, IntentEnum

    brand_deep_dive_centroid = np.array([0.2] * 400 + [0.0] * 624)

    monkeypatch.setattr(intent_module, "_intent_ready", True)
    monkeypatch.setattr(intent_module, "_centroids", {
        "brand_deep_dive": brand_deep_dive_centroid,
    })

    result = classify_intent("what about it", embedding=SAMPLE_EMBEDDING, history=[])

    assert result["intent"] == IntentEnum.BRAND_DEEP_DIVE
    assert result["layer"] == "embedding"


def test_inherit_still_fires_when_prior_matches_best_intent_INTENT09(monkeypatch):
    """INTENT-09: Inherit IS applied when prior == best_intent (desired multi-turn behaviour).

    Scenario: user was in a brand_deep_dive conversation, sends an ambiguous short
    follow-up. Prior = brand_deep_dive, best embedding = brand_deep_dive. Inherit
    should fire and return embedding_inherit layer.
    """
    import app.intent as intent_module
    from app.intent import classify_intent, IntentEnum

    tentative_centroid = np.array([0.2] * 400 + [0.0] * 624)

    monkeypatch.setattr(intent_module, "_intent_ready", True)
    monkeypatch.setattr(intent_module, "_centroids", {
        "brand_deep_dive": tentative_centroid,
    })

    history = [{"role": "user", "content": "tell me about nutoy", "intent": "brand_deep_dive"}]

    result = classify_intent(
        "more",  # 1 word, tentative, prior == best_intent
        embedding=SAMPLE_EMBEDDING,
        history=history,
    )

    assert result["intent"] == IntentEnum.BRAND_DEEP_DIVE
    assert result["layer"] == "embedding_inherit"


def test_product_prior_beats_chitchat_best_INTENT10(monkeypatch):
    """INTENT-10: Product-relevant prior overrides chitchat best-match on short follow-ups.

    Scenario: user asked "rubio monocoat" (prior=general_qna), then sends "what does it do"
    (embedding best=chitchat, tentative). The product prior should inherit, not chitchat.
    """
    import app.intent as intent_module
    from app.intent import classify_intent, IntentEnum

    # chitchat scores slightly higher than general_qna in the tentative band
    chitchat_centroid  = np.array([0.2] * 400 + [0.0] * 624)   # will score ~0.71
    general_qna_centroid = np.array([0.15] * 400 + [0.0] * 624)  # will score slightly lower

    monkeypatch.setattr(intent_module, "_intent_ready", True)
    monkeypatch.setattr(intent_module, "_centroids", {
        "chitchat":    chitchat_centroid,
        "general_qna": general_qna_centroid,
    })

    history = [{"role": "user", "content": "rubio monocoat", "intent": "general_qna"}]

    result = classify_intent(
        "what does it do",
        embedding=SAMPLE_EMBEDDING,
        history=history,
    )

    assert result["intent"] == IntentEnum.GENERAL_QNA, (
        "INTENT-10 regression: chitchat overrode general_qna prior on product follow-up. "
        f"Got intent={result['intent'].value} layer={result['layer']}"
    )
    assert result["layer"] == "embedding_inherit"


# ─── Regression tests for brand-name Layer 1 rule (INTENT-11) ────────────────

def test_brand_name_rubio_monocoat_layer1_INTENT11():
    """INTENT-11: 'rubio monocoat' alone classifies as BRAND_DEEP_DIVE via Layer 1 rule."""
    from app.intent import classify_intent, IntentEnum
    result = classify_intent("rubio monocoat")
    assert result == {"intent": IntentEnum.BRAND_DEEP_DIVE, "confidence": 1.0, "layer": "rule"}, (
        f"Bug regression: 'rubio monocoat' was not caught by Layer 1. Got {result}"
    )


def test_brand_name_nuacoustics_layer1_INTENT11():
    """INTENT-11: 'nuacoustics' alone classifies as BRAND_DEEP_DIVE via Layer 1 rule."""
    from app.intent import classify_intent, IntentEnum
    result = classify_intent("nuacoustics")
    assert result["intent"] == IntentEnum.BRAND_DEEP_DIVE
    assert result["layer"] == "rule"
    assert result["confidence"] == 1.0


def test_brand_name_nutoy_layer1_INTENT11():
    """INTENT-11: 'nutoy' alone classifies as BRAND_DEEP_DIVE via Layer 1 rule."""
    from app.intent import classify_intent, IntentEnum
    result = classify_intent("nutoy")
    assert result["intent"] == IntentEnum.BRAND_DEEP_DIVE
    assert result["layer"] == "rule"
    assert result["confidence"] == 1.0


def test_brand_name_nupanel_layer1_INTENT11():
    """INTENT-11: 'nupanel' alone classifies as BRAND_DEEP_DIVE via Layer 1 rule."""
    from app.intent import classify_intent, IntentEnum
    result = classify_intent("nupanel")
    assert result["intent"] == IntentEnum.BRAND_DEEP_DIVE
    assert result["layer"] == "rule"
    assert result["confidence"] == 1.0


def test_brand_name_nuwork_layer1_INTENT11():
    """INTENT-11: 'nuwork' alone classifies as BRAND_DEEP_DIVE via Layer 1 rule."""
    from app.intent import classify_intent, IntentEnum
    result = classify_intent("nuwork")
    assert result["intent"] == IntentEnum.BRAND_DEEP_DIVE
    assert result["layer"] == "rule"
    assert result["confidence"] == 1.0


def test_brand_name_case_insensitive_INTENT11():
    """INTENT-11: Brand name regex is case-insensitive."""
    from app.intent import classify_intent, IntentEnum
    for query in ("Rubio Monocoat", "RUBIO MONOCOAT", "Nutoy", "NuPanel", "NUWORK"):
        result = classify_intent(query)
        assert result["intent"] == IntentEnum.BRAND_DEEP_DIVE, (
            f"Brand name '{query}' not caught by Layer 1 rule. Got {result}"
        )
        assert result["layer"] == "rule"


def test_brand_name_with_punctuation_INTENT11():
    """INTENT-11: Brand name with trailing punctuation still matches Layer 1 rule."""
    from app.intent import classify_intent, IntentEnum
    result = classify_intent("rubio monocoat?")
    assert result["intent"] == IntentEnum.BRAND_DEEP_DIVE
    assert result["layer"] == "rule"


def test_brand_name_mid_sentence_not_matched_INTENT11():
    """INTENT-11: Brand name mid-sentence does NOT trigger Layer 1 brand rule (fullmatch only)."""
    from app.intent import classify_intent, IntentEnum
    # This should fall through to Layer 2, not be caught as a brand-name-only query
    result = classify_intent("tell me about rubio monocoat products")
    # Layer 1 brand rule must NOT fire — it's a multi-word sentence
    assert result["layer"] != "rule" or result["intent"] != IntentEnum.BRAND_DEEP_DIVE or \
        result["confidence"] < 1.0 or result["layer"] == "fallback", (
        "Brand name mid-sentence incorrectly matched Layer 1 fullmatch rule"
    )
    # More directly: the layer should be fallback (no centroids loaded) not 'rule'
    # because _BRAND_NAME_RE uses fullmatch and the sentence has extra words
    assert result["layer"] in ("fallback", "fallback_no_embedding", "embedding", "embedding_inherit")


# ─── Regression tests for specificity-inherit bug (INTENT-12) ────────────────

def test_brand_deep_dive_prior_beats_brand_discovery_best_INTENT12(monkeypatch):
    """INTENT-12 regression: brand_deep_dive prior must override tentative brand_discovery match.

    Scenario (Bug 1): user asked about Rubio Monocoat (prior=brand_deep_dive), then sends
    "what all does it have" (5 words, embedding best=brand_discovery at 0.708, tentative).
    The more-specific prior (brand_deep_dive) should inherit, not vague brand_discovery.
    """
    import app.intent as intent_module
    from app.intent import classify_intent, IntentEnum

    # brand_discovery centroid scores slightly higher in tentative band
    # brand_deep_dive scores slightly lower
    brand_discovery_centroid = np.array([0.2] * 400 + [0.0] * 624)   # ~0.625
    brand_deep_dive_centroid = np.array([0.18] * 400 + [0.0] * 624)  # slightly lower

    monkeypatch.setattr(intent_module, "_intent_ready", True)
    monkeypatch.setattr(intent_module, "_centroids", {
        "brand_discovery": brand_discovery_centroid,
        "brand_deep_dive": brand_deep_dive_centroid,
    })

    history = [{"role": "user", "content": "tell me about rubio monocoat", "intent": "brand_deep_dive"}]

    result = classify_intent(
        "what all does it have",  # 5 words, tentative, prior=brand_deep_dive, best=brand_discovery
        embedding=SAMPLE_EMBEDDING,
        history=history,
    )

    assert result["intent"] == IntentEnum.BRAND_DEEP_DIVE, (
        "INTENT-12 regression: brand_discovery overrode brand_deep_dive prior on follow-up. "
        f"Got intent={result['intent'].value} layer={result['layer']}"
    )
    assert result["layer"] == "embedding_inherit"


def test_brand_deep_dive_prior_beats_general_qna_best_INTENT12(monkeypatch):
    """INTENT-12: brand_deep_dive prior overrides tentative general_qna match."""
    import app.intent as intent_module
    from app.intent import classify_intent, IntentEnum

    general_qna_centroid     = np.array([0.2] * 400 + [0.0] * 624)
    brand_deep_dive_centroid = np.array([0.18] * 400 + [0.0] * 624)

    monkeypatch.setattr(intent_module, "_intent_ready", True)
    monkeypatch.setattr(intent_module, "_centroids", {
        "general_qna":     general_qna_centroid,
        "brand_deep_dive": brand_deep_dive_centroid,
    })

    history = [{"role": "user", "content": "rubio monocoat", "intent": "brand_deep_dive"}]

    result = classify_intent(
        "what does it do",
        embedding=SAMPLE_EMBEDDING,
        history=history,
    )

    assert result["intent"] == IntentEnum.BRAND_DEEP_DIVE
    assert result["layer"] == "embedding_inherit"


def test_specific_product_prior_beats_brand_discovery_best_INTENT12(monkeypatch):
    """INTENT-12: specific_product prior (non-vague) overrides tentative brand_discovery match."""
    import app.intent as intent_module
    from app.intent import classify_intent, IntentEnum

    brand_discovery_centroid = np.array([0.2] * 400 + [0.0] * 624)
    specific_product_centroid = np.array([0.18] * 400 + [0.0] * 624)

    monkeypatch.setattr(intent_module, "_intent_ready", True)
    monkeypatch.setattr(intent_module, "_centroids", {
        "brand_discovery":  brand_discovery_centroid,
        "specific_product": specific_product_centroid,
    })

    history = [{"role": "user", "content": "rubio monocoat pure", "intent": "specific_product"}]

    result = classify_intent(
        "how much does it cost",
        embedding=SAMPLE_EMBEDDING,
        history=history,
    )

    assert result["intent"] == IntentEnum.SPECIFIC_PRODUCT
    assert result["layer"] == "embedding_inherit"


def test_vague_prior_does_not_inherit_over_specific_best_INTENT12(monkeypatch):
    """INTENT-12: A vague prior (brand_discovery) does NOT override a more-specific best-match.

    brand_discovery is in _VAGUE_PRODUCT_INTENTS. When the embedding best-match is
    brand_deep_dive (specific) and the prior is brand_discovery (vague), inherit must NOT
    fire — the more-specific embedding match wins.

    Centroids: brand_deep_dive scores ~0.625 (400 active dims);
               brand_discovery scores ~0.541 (300 active dims) — clearly below.
    """
    import app.intent as intent_module
    from app.intent import classify_intent, IntentEnum

    # brand_deep_dive clearly wins; brand_discovery is below tentative threshold
    brand_deep_dive_centroid = np.array([0.2] * 400 + [0.0] * 624)   # ~0.625
    brand_discovery_centroid = np.array([0.2] * 300 + [0.0] * 724)   # ~0.541

    monkeypatch.setattr(intent_module, "_intent_ready", True)
    monkeypatch.setattr(intent_module, "_centroids", {
        "brand_deep_dive": brand_deep_dive_centroid,
        "brand_discovery": brand_discovery_centroid,
    })

    # Prior is the vague intent; embedding top-match is the specific intent
    history = [{"role": "user", "content": "which brands do you have", "intent": "brand_discovery"}]

    result = classify_intent(
        "tell me more about it",  # short, tentative; best=brand_deep_dive, prior=brand_discovery
        embedding=SAMPLE_EMBEDDING,
        history=history,
    )

    # brand_deep_dive is best (not vague) and prior is vague — embedding wins, no inherit
    assert result["intent"] == IntentEnum.BRAND_DEEP_DIVE
    assert result["layer"] == "embedding"


# ─── INTENT-13: Layer 1.5 catalog entity pre-pass ────────────────────────────

def _load_sample_entity_index(monkeypatch):
    """Helper: load a representative entity index into the intent module."""
    import app.intent as intent_module
    from app.intent import load_entity_index
    brands        = ["Rubio Monocoat", "Nuacoustics", "Nutoy", "Nupanel", "Nuwork"]
    product_lines = [
        "On Wheels", "Stacker", "Montessori", "Building Block",
        "Board Games", "Balancing", "Learning", "Furniture",
        "MDF Perforated", "Open Work Panel", "Panelsys",
        "PET Ceiling", "PET Light", "PET Plain", "PET VG",
        "Components", "Storage",
    ]
    product_names = [
        "Nutoy-On Wheels-Duck", "Nutoy-On Wheels-Rabbit", "Nutoy-On Wheels-Swan-Black",
        "Nutoy-Stacker-Rainbow", "Nutoy-Montessori-Cylinder",
    ]
    load_entity_index(brands, product_lines, product_names)
    assert intent_module._entity_index_ready is True


def test_entity_index_loaded_INTENT13(monkeypatch):
    """INTENT-13: load_entity_index populates index and sets _entity_index_ready=True."""
    import app.intent as intent_module
    _load_sample_entity_index(monkeypatch)
    assert intent_module._entity_index_ready is True
    assert "on wheels" in intent_module._entity_product_line_names
    assert "stacker" in intent_module._entity_product_line_names
    assert "duck" in intent_module._entity_product_tokens


def test_product_line_query_via_catalog_entity_INTENT13(monkeypatch):
    """INTENT-13: Exact product line name 'on wheels' → product_line_query, layer=catalog_entity."""
    from app.intent import classify_intent, IntentEnum
    _load_sample_entity_index(monkeypatch)
    result = classify_intent("on wheels")
    assert result["intent"] == IntentEnum.PRODUCT_LINE_QUERY
    assert result["layer"] == "catalog_entity"
    assert result["confidence"] == 1.0


def test_product_line_query_case_insensitive_INTENT13(monkeypatch):
    """INTENT-13: Product line match is case-insensitive."""
    from app.intent import classify_intent, IntentEnum
    _load_sample_entity_index(monkeypatch)
    for query in ("On Wheels", "ON WHEELS", "on wheels", "Stacker", "STACKER"):
        result = classify_intent(query)
        assert result["layer"] == "catalog_entity", (
            f"Query {query!r} did not hit catalog_entity layer. Got {result}"
        )


def test_product_line_query_with_trailing_punctuation_INTENT13(monkeypatch):
    """INTENT-13: Product line name with trailing punctuation still matches."""
    from app.intent import classify_intent, IntentEnum
    _load_sample_entity_index(monkeypatch)
    result = classify_intent("on wheels?")
    assert result["intent"] == IntentEnum.PRODUCT_LINE_QUERY
    assert result["layer"] == "catalog_entity"


def test_specific_product_via_token_match_INTENT13(monkeypatch):
    """INTENT-13: Query containing a known product token → specific_product, layer=catalog_entity."""
    from app.intent import classify_intent, IntentEnum
    _load_sample_entity_index(monkeypatch)
    # 'duck' is a token from 'Nutoy-On Wheels-Duck' (len >= 4)
    result = classify_intent("duck toy")
    assert result["intent"] == IntentEnum.SPECIFIC_PRODUCT
    assert result["layer"] == "catalog_entity"
    assert result["confidence"] == 1.0


def test_specific_product_short_query_INTENT13(monkeypatch):
    """INTENT-13: Short query 'duck' alone → specific_product via token match."""
    from app.intent import classify_intent, IntentEnum
    _load_sample_entity_index(monkeypatch)
    result = classify_intent("duck")
    assert result["intent"] == IntentEnum.SPECIFIC_PRODUCT
    assert result["layer"] == "catalog_entity"


def test_specific_product_image_request_INTENT13(monkeypatch):
    """INTENT-13: 'image pls' without entity tokens does NOT trigger catalog_entity.

    This query has no token >= 4 chars that matches the product index,
    so Layer 1.5 must NOT fire. The query reaches Layer 2 (or fallback).
    """
    from app.intent import classify_intent, IntentEnum
    _load_sample_entity_index(monkeypatch)
    # 'image' is not a product token; 'pls' < 4 chars
    result = classify_intent("image pls")
    assert result["layer"] != "catalog_entity"


def test_entity_index_disabled_without_load_INTENT13():
    """INTENT-13: Layer 1.5 is silently skipped when entity index not loaded (_entity_index_ready=False)."""
    from app.intent import classify_intent, IntentEnum
    # autouse fixture ensures _entity_index_ready=False
    # 'duck' would match if the index were loaded, but must fall through without it
    result = classify_intent("duck")
    # Should degrade to fallback (no centroids), not catalog_entity
    assert result["layer"] != "catalog_entity"
    assert result["intent"] == IntentEnum.GENERAL_QNA


def test_layer1_fires_before_entity_index_INTENT13(monkeypatch):
    """INTENT-13: Layer 1 (greeting) takes priority over entity index."""
    from app.intent import classify_intent, IntentEnum
    _load_sample_entity_index(monkeypatch)
    # 'hi' matches Layer 1 greeting — must NOT reach Layer 1.5
    result = classify_intent("hi")
    assert result["intent"] == IntentEnum.GREETING
    assert result["layer"] == "rule"


def test_load_entity_index_graceful_on_empty_INTENT13(monkeypatch):
    """INTENT-13: load_entity_index with empty lists sets index ready with empty sets."""
    import app.intent as intent_module
    from app.intent import load_entity_index
    load_entity_index([], [], [])
    assert intent_module._entity_index_ready is True
    assert len(intent_module._entity_brand_names) == 0
    assert len(intent_module._entity_product_line_names) == 0
    assert len(intent_module._entity_product_tokens) == 0


def test_short_noise_tokens_not_matched_INTENT13(monkeypatch):
    """INTENT-13: Tokens shorter than _PRODUCT_TOKEN_MIN_LEN (4) are not indexed."""
    import app.intent as intent_module
    from app.intent import load_entity_index
    # 'On' (2), 'a' (1) would be noise — only 'Duck' (4) should be indexed
    load_entity_index([], [], ["On Wheels-Duck"])
    # 'on' has len=2 < 4, 'duck' has len=4 — only 'duck' and 'wheels' should be in index
    # (both are >= 4 chars)
    assert "duck" in intent_module._entity_product_tokens
    assert "wheels" in intent_module._entity_product_tokens
    # 'on' should NOT be in the token index
    assert "on" not in intent_module._entity_product_tokens


# ─── Bug 1 regression: brand name queries via Layer 1.5 ──────────────────────

def test_brand_query_in_sentence_not_specific_product_INTENT15(monkeypatch):
    """INTENT-15 regression: 'what is rubio monocoat' must NOT classify as specific_product.

    Layer 1 fullmatch doesn't fire (sentence has extra words). Layer 1.5 must detect
    that 'rubio'+'monocoat' are brand tokens (not product model tokens) and return
    brand_deep_dive, not specific_product.
    """
    from app.intent import classify_intent, load_entity_index, IntentEnum
    load_entity_index(
        brands=["Rubio Monocoat", "Nuacoustics", "Nutoy", "Nupanel", "Nuwork"],
        product_lines=["On Wheels", "Stacker"],
        product_names=["Rubio Monocoat-Pure", "Rubio Monocoat-Oil Plus 2C"],
    )
    result = classify_intent("what is rubio monocoat")
    assert result["intent"] == IntentEnum.BRAND_DEEP_DIVE, (
        "INTENT-15 regression: brand name sentence wrongly classified as specific_product. "
        f"Got intent={result['intent'].value} layer={result['layer']}"
    )
    assert result["layer"] == "catalog_entity"


def test_brand_query_tell_me_about_INTENT15(monkeypatch):
    """INTENT-15: 'tell me about nuacoustics' → brand_deep_dive via Layer 1.5."""
    from app.intent import classify_intent, load_entity_index, IntentEnum
    load_entity_index(
        brands=["Rubio Monocoat", "Nuacoustics", "Nutoy", "Nupanel", "Nuwork"],
        product_lines=["PET Ceiling", "MDF Perforated"],
        product_names=["Nuacoustics-PET Ceiling-White", "Nuacoustics-PET Plain-Grey"],
    )
    result = classify_intent("tell me about nuacoustics")
    assert result["intent"] == IntentEnum.BRAND_DEEP_DIVE
    assert result["layer"] == "catalog_entity"


def test_mixed_brand_and_product_token_stays_specific_product_INTENT15(monkeypatch):
    """INTENT-15: When query contains both brand tokens AND product-model tokens,
    specific_product wins (not all matched tokens are brand tokens).

    'rubio monocoat pure' matches 'rubio'+'monocoat' (brand tokens) AND 'pure'
    (product model token). Since not ALL matched tokens are brand tokens, it is
    a specific product query.
    """
    from app.intent import classify_intent, load_entity_index, IntentEnum
    load_entity_index(
        brands=["Rubio Monocoat", "Nutoy"],
        product_lines=["Stacker"],
        product_names=["Rubio Monocoat-Pure", "Rubio Monocoat-Oil Plus 2C"],
    )
    # 'pure' is a product model token; 'rubio'+'monocoat' are brand tokens.
    # Since 'pure' is NOT a brand token, NOT all matched tokens are brand tokens.
    result = classify_intent("rubio monocoat pure")
    # Layer 1 fullmatch doesn't fire (3 words including 'pure')
    # Layer 1.5 step 3 fires; matched_tokens = {rubio, monocoat, pure}
    # 'pure' is not a brand token → specific_product
    assert result["intent"] == IntentEnum.SPECIFIC_PRODUCT
    assert result["layer"] == "catalog_entity"


def test_brand_token_index_populated_by_load_entity_index_INTENT15(monkeypatch):
    """INTENT-15: load_entity_index populates _entity_brand_tokens from brand names."""
    import app.intent as intent_module
    from app.intent import load_entity_index
    load_entity_index(
        brands=["Rubio Monocoat", "Nutoy"],
        product_lines=[],
        product_names=[],
    )
    assert "rubio" in intent_module._entity_brand_tokens
    assert "monocoat" in intent_module._entity_brand_tokens
    assert "nutoy" in intent_module._entity_brand_tokens


# ─── Bug 3 regression: context-anchored follow-up queries ────────────────────

def test_what_colors_after_specific_product_anchors_to_general_qna_INTENT14(monkeypatch):
    """INTENT-14 regression: 'what colors do you have' after specific_product → general_qna.

    The embedding for 'what colors do you have' scores highest for brand_discovery (~0.785).
    With a specific_product prior in context, the strong context-anchor rule must override
    and return general_qna (anchored to the prior product context).
    """
    import app.intent as intent_module
    from app.intent import classify_intent, IntentEnum

    # brand_discovery centroid scores above CONFIDENCE_THRESHOLD with SAMPLE_EMBEDDING
    # to simulate the observed 0.785 score
    brand_discovery_centroid = np.array([0.25] * 400 + [0.0] * 624)  # will score ~0.79

    monkeypatch.setattr(intent_module, "_intent_ready", True)
    monkeypatch.setattr(intent_module, "_centroids", {
        "brand_discovery": brand_discovery_centroid,
    })

    history = [{"role": "user", "content": "stacker rainbow", "intent": "specific_product"}]

    result = classify_intent(
        "what colors do you have",  # 5 words, brand_discovery at high confidence
        embedding=SAMPLE_EMBEDDING,
        history=history,
    )

    assert result["intent"] == IntentEnum.GENERAL_QNA, (
        "INTENT-14 regression: brand_discovery overrode specific_product context for follow-up. "
        f"Got intent={result['intent'].value} layer={result['layer']}"
    )
    assert result["layer"] == "embedding_inherit"


def test_what_sizes_after_product_line_query_anchors_to_general_qna_INTENT14(monkeypatch):
    """INTENT-14: 'what sizes do you have' after product_line_query → general_qna."""
    import app.intent as intent_module
    from app.intent import classify_intent, IntentEnum

    brand_discovery_centroid = np.array([0.25] * 400 + [0.0] * 624)

    monkeypatch.setattr(intent_module, "_intent_ready", True)
    monkeypatch.setattr(intent_module, "_centroids", {
        "brand_discovery": brand_discovery_centroid,
    })

    history = [{"role": "user", "content": "acoustic panels", "intent": "product_line_query"}]

    result = classify_intent(
        "what sizes do you have",
        embedding=SAMPLE_EMBEDDING,
        history=history,
    )

    assert result["intent"] == IntentEnum.GENERAL_QNA
    assert result["layer"] == "embedding_inherit"


def test_what_colors_without_product_prior_stays_brand_discovery_INTENT14(monkeypatch):
    """INTENT-14: 'what colors do you have' with NO prior → brand_discovery (no override)."""
    import app.intent as intent_module
    from app.intent import classify_intent, IntentEnum

    brand_discovery_centroid = np.array([0.25] * 400 + [0.0] * 624)

    monkeypatch.setattr(intent_module, "_intent_ready", True)
    monkeypatch.setattr(intent_module, "_centroids", {
        "brand_discovery": brand_discovery_centroid,
    })

    result = classify_intent(
        "what colors do you have",
        embedding=SAMPLE_EMBEDDING,
        history=[],
    )

    # No prior product context — brand_discovery stands
    assert result["intent"] == IntentEnum.BRAND_DISCOVERY
    assert result["layer"] == "embedding"


def test_what_colors_after_non_inheritable_prior_stays_brand_discovery_INTENT14(monkeypatch):
    """INTENT-14: brand_discovery is NOT overridden when prior is greeting (non-inheritable)."""
    import app.intent as intent_module
    from app.intent import classify_intent, IntentEnum

    brand_discovery_centroid = np.array([0.25] * 400 + [0.0] * 624)

    monkeypatch.setattr(intent_module, "_intent_ready", True)
    monkeypatch.setattr(intent_module, "_centroids", {
        "brand_discovery": brand_discovery_centroid,
    })

    history = [{"role": "user", "content": "hi", "intent": "greeting"}]

    result = classify_intent(
        "what colors do you have",
        embedding=SAMPLE_EMBEDDING,
        history=history,
    )

    # Greeting prior is non-inheritable — brand_discovery stands
    assert result["intent"] == IntentEnum.BRAND_DISCOVERY
    assert result["layer"] == "embedding"


# ─── Bug 1 regression: "show me" / "can you show me" → specific_product ────────

def test_show_me_resolves_to_specific_product_layer1():
    """Bug 1 regression: 'can you show me' must hit the Layer 1 visual-request rule
    and return specific_product — never fall through to brand_discovery_anchor."""
    from app.intent import classify_intent, IntentEnum

    result = classify_intent("can you show me", embedding=None, history=[])
    assert result["intent"] == IntentEnum.SPECIFIC_PRODUCT
    assert result["layer"] == "rule"
    assert result["confidence"] == 1.0


def test_show_me_question_mark_resolves_to_specific_product_layer1():
    """Bug 1 regression: 'can you show me?' (trailing punctuation) must also match."""
    from app.intent import classify_intent, IntentEnum

    result = classify_intent("can you show me?", embedding=None, history=[])
    assert result["intent"] == IntentEnum.SPECIFIC_PRODUCT
    assert result["layer"] == "rule"


def test_show_me_photo_resolves_to_specific_product_layer1():
    """'show me a photo' — explicit image request resolves to specific_product."""
    from app.intent import classify_intent, IntentEnum

    result = classify_intent("show me a photo", embedding=None, history=[])
    assert result["intent"] == IntentEnum.SPECIFIC_PRODUCT
    assert result["layer"] == "rule"


def test_show_me_bypasses_brand_discovery_anchor(monkeypatch):
    """Bug 1 regression: 'can you show me' must resolve to specific_product even when
    history contains a product prior that would trigger brand_discovery_anchor."""
    import app.intent as intent_module
    from app.intent import classify_intent, IntentEnum

    # Set up centroids that would normally give brand_discovery — anchor should not fire
    brand_discovery_centroid = np.array([0.25] * 400 + [0.0] * 624)
    monkeypatch.setattr(intent_module, "_intent_ready", True)
    monkeypatch.setattr(intent_module, "_centroids", {"brand_discovery": brand_discovery_centroid})

    history = [{"role": "user", "content": "tell me about cloud hexagon panel",
                "intent": "specific_product"}]

    result = classify_intent("can you show me", embedding=SAMPLE_EMBEDDING, history=history)
    # Layer 1 rule fires first — brand_discovery_anchor never runs
    assert result["intent"] == IntentEnum.SPECIFIC_PRODUCT
    assert result["layer"] == "rule"


def test_show_me_named_product_resolves_to_specific_product_layer1():
    """'show me the Cloud Hexagon' should resolve to specific_product via visual rule."""
    from app.intent import classify_intent, IntentEnum

    result = classify_intent("show me the Cloud Hexagon", embedding=None, history=[])
    assert result["intent"] == IntentEnum.SPECIFIC_PRODUCT
    assert result["layer"] == "rule"
