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
