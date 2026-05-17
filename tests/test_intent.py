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
    yield
    monkeypatch.setattr(intent_module, "_centroids", {})
    monkeypatch.setattr(intent_module, "_intent_ready", False)


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
