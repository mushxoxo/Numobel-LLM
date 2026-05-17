"""Regression tests for media image_url hallucination fix.

Covers:
- is_blocked_image_url() blocklist matching
- _resolve_media_image_url() substitution and downgrade logic
- webhook end-to-end: example.com URL replaced by real Wixstatic URL from hits
- webhook end-to-end: no real image → downgrade to text

Bug: LLM hallucinated https://example.com/... as image_url when message_type=media
because real image URLs live in ChromaDB metadata (not chunk text) and were
never visible to the LLM during generation.
"""
import os
import sys
import pytest
from unittest.mock import patch, MagicMock

from app.validators.hallucination import is_blocked_image_url
from app.intent import IntentEnum
from app.validators.response import ValidationResult


# ─── is_blocked_image_url ─────────────────────────────────────────────────────

def test_blocked_example_com():
    assert is_blocked_image_url("https://example.com/Nutoy-On-Wheels-Duck.jpg") is True


def test_blocked_placeholder_com():
    assert is_blocked_image_url("https://placeholder.com/200x200") is True


def test_blocked_via_placeholder():
    assert is_blocked_image_url("https://via.placeholder.com/300") is True


def test_blocked_dummyimage():
    assert is_blocked_image_url("https://dummyimage.com/600x400/000/fff") is True


def test_blocked_case_insensitive():
    assert is_blocked_image_url("https://EXAMPLE.COM/img.jpg") is True


def test_not_blocked_real_wixstatic_url():
    real = "https://static.wixstatic.com/media/c3551d_d01b6634c519423cb6445651e11fef58~mv2.jpg"
    assert is_blocked_image_url(real) is False


def test_not_blocked_numobel_url():
    assert is_blocked_image_url("https://www.numobel.in/product-page/nutoy-on-wheels-duck") is False


def test_not_blocked_none():
    assert is_blocked_image_url(None) is False


def test_not_blocked_empty_string():
    assert is_blocked_image_url("") is False


# ─── _resolve_media_image_url ─────────────────────────────────────────────────

REAL_URL   = "https://static.wixstatic.com/media/abc123~mv2.jpg"
BLOCKED_URL = "https://example.com/Nutoy-On-Wheels-Duck.jpg"

_HIT_WITH_IMAGE = {"metadata": {"images": REAL_URL}}


def _make_result(message_type="media", image_url=None):
    return {
        "message_type":      message_type,
        "content":           "Here is the duck toy.",
        "buttons":           None,
        "image_url":         image_url,
        "prompt_tokens":     100,
        "completion_tokens": 50,
    }


def test_resolve_passes_through_non_media_unchanged():
    from app.webhook import _resolve_media_image_url
    result = _make_result(message_type="text", image_url=None)
    out = _resolve_media_image_url(result, [_HIT_WITH_IMAGE])
    assert out["message_type"] == "text"
    assert out["image_url"] is None


def test_resolve_passes_through_valid_image_url():
    from app.webhook import _resolve_media_image_url
    result = _make_result(message_type="media", image_url=REAL_URL)
    out = _resolve_media_image_url(result, [])
    assert out["image_url"] == REAL_URL
    assert out["message_type"] == "media"


def test_resolve_replaces_blocked_url_with_hit_image():
    """Core regression: example.com URL replaced by real URL from hits metadata."""
    from app.webhook import _resolve_media_image_url
    result = _make_result(message_type="media", image_url=BLOCKED_URL)
    out = _resolve_media_image_url(result, [_HIT_WITH_IMAGE])
    assert out["image_url"] == REAL_URL
    assert out["message_type"] == "media"


def test_resolve_fills_missing_image_url_from_hits():
    """When LLM omits image_url entirely, first hit image is used."""
    from app.webhook import _resolve_media_image_url
    result = _make_result(message_type="media", image_url=None)
    out = _resolve_media_image_url(result, [_HIT_WITH_IMAGE])
    assert out["image_url"] == REAL_URL
    assert out["message_type"] == "media"


def test_resolve_downgrades_to_text_when_no_hits():
    """No hits and blocked URL → downgrade to text, never send fake URL."""
    from app.webhook import _resolve_media_image_url
    result = _make_result(message_type="media", image_url=BLOCKED_URL)
    out = _resolve_media_image_url(result, [])
    assert out["message_type"] == "text"
    assert out["image_url"] is None


def test_resolve_downgrades_to_text_when_hits_have_no_images():
    """Hits present but no images field → downgrade to text."""
    from app.webhook import _resolve_media_image_url
    hit_no_image = {"metadata": {"images": "", "product_link": "https://numobel.in/duck"}}
    result = _make_result(message_type="media", image_url=None)
    out = _resolve_media_image_url(result, [hit_no_image])
    assert out["message_type"] == "text"
    assert out["image_url"] is None


def test_resolve_does_not_mutate_original_result():
    """_resolve_media_image_url must return a new dict, not modify the input."""
    from app.webhook import _resolve_media_image_url
    result = _make_result(message_type="media", image_url=BLOCKED_URL)
    original_url = result["image_url"]
    _resolve_media_image_url(result, [_HIT_WITH_IMAGE])
    assert result["image_url"] == original_url  # original unchanged


def test_resolve_picks_first_image_from_pipe_separated_list():
    """When images field has multiple URLs, first one is used."""
    from app.webhook import _resolve_media_image_url
    second_url = "https://static.wixstatic.com/media/second~mv2.jpg"
    hit = {"metadata": {"images": f"{REAL_URL}|{second_url}"}}
    result = _make_result(message_type="media", image_url=None)
    out = _resolve_media_image_url(result, [hit])
    assert out["image_url"] == REAL_URL


# ─── Webhook integration: media image_url resolution ─────────────────────────

INCOMING_PHOTO = {
    "contact": {"phone_number": "919999999999"},
    "message": {"body": "share photos", "whatsapp_message_id": "msg-photo-001"},
}

MEDIA_RESULT_HALLUCINATED = {
    "message_type":      "media",
    "content":           "Here is the Nutoy-On Wheels-Duck.",
    "buttons":           None,
    "image_url":         "https://example.com/Nutoy-On-Wheels-Duck.jpg",
    "prompt_tokens":     200,
    "completion_tokens": 50,
}

DUCK_HIT = {
    "text":     "Product: Nutoy-On Wheels-Duck\nBrand: Nutoy",
    "metadata": {
        "product_name": "Nutoy-On Wheels-Duck",
        "brand":        "Nutoy",
        "images":       REAL_URL,
        "product_link": "https://www.numobel.in/product-page/nutoy-on-wheels-duck",
    },
    "distance": 0.05,
}


@pytest.fixture
def photo_client():
    os.environ.setdefault("WA2MATION_API_KEY", "test")
    os.environ.setdefault("WA2MATION_VENDOR_UID", "test")
    with patch("app.startup.run_startup"):
        sys.modules.pop("app.webhook", None)
        import app.webhook as webhook_module

        webhook_module._startup._ready = True
        with patch("app.webhook.rag.get_collection", return_value=MagicMock()), \
             patch("app.webhook.validate_whatsapp_response", side_effect=lambda x: x), \
             patch("app.webhook.validate_response", return_value=ValidationResult(valid=True)):
            app_obj = webhook_module.app
            app_obj.config["TESTING"] = True
            with app_obj.test_client() as c:
                yield c


def test_webhook_replaces_hallucinated_image_url_for_media_response(photo_client):
    """Regression: 'share photos' must dispatch a real Wixstatic URL, never example.com."""
    emb = [0.1] * 1024
    with patch("app.webhook.load_history", return_value=[{"role": "user", "content": "tell me about duck on wheels"}]), \
         patch("app.webhook.rag.get_embedding", return_value=emb), \
         patch("app.webhook.classify_intent", return_value={"intent": IntentEnum.SPECIFIC_PRODUCT, "confidence": 0.85, "layer": "embedding"}), \
         patch("app.webhook.rag.get_collection") as mock_col, \
         patch("app.webhook.rag.rewrite_query", return_value="Can you share photos of the Nutoy-On Wheels-Duck?"), \
         patch("app.webhook.rag.retrieve", return_value=[DUCK_HIT]), \
         patch("app.webhook.plan_response", return_value="media"), \
         patch("app.webhook.rag.generate_answer", return_value=MEDIA_RESULT_HALLUCINATED), \
         patch("app.webhook.dispatch") as mock_dispatch, \
         patch("app.webhook.save_history"):
        mock_col.return_value.query.return_value = {"distances": [[1.0]], "documents": [[]], "metadatas": [[]]}
        resp = photo_client.post("/webhook", json=INCOMING_PHOTO)

    assert resp.status_code == 200
    assert resp.json["status"] == "success"
    mock_dispatch.assert_called_once()
    _, dispatched_result, _ = mock_dispatch.call_args[0]
    assert dispatched_result["message_type"] == "media"
    assert dispatched_result["image_url"] == REAL_URL, (
        f"Expected real Wixstatic URL, got: {dispatched_result['image_url']!r}"
    )
    assert "example.com" not in (dispatched_result["image_url"] or ""), (
        "Hallucinated example.com URL must never reach dispatch"
    )


def test_webhook_downgrades_media_to_text_when_no_real_image(photo_client):
    """When no image URL exists in hits and LLM hallucinates, response becomes text."""
    emb = [0.1] * 1024
    hit_no_image = {
        "text":     "Product: SomeProduct\nBrand: Nutoy",
        "metadata": {"product_name": "SomeProduct", "brand": "Nutoy", "images": "", "product_link": ""},
        "distance": 0.05,
    }
    with patch("app.webhook.load_history", return_value=[]), \
         patch("app.webhook.rag.get_embedding", return_value=emb), \
         patch("app.webhook.classify_intent", return_value={"intent": IntentEnum.SPECIFIC_PRODUCT, "confidence": 0.85, "layer": "embedding"}), \
         patch("app.webhook.rag.get_collection") as mock_col, \
         patch("app.webhook.rag.rewrite_query", return_value="show me photos"), \
         patch("app.webhook.rag.retrieve", return_value=[hit_no_image]), \
         patch("app.webhook.plan_response", return_value="media"), \
         patch("app.webhook.rag.generate_answer", return_value=MEDIA_RESULT_HALLUCINATED), \
         patch("app.webhook.dispatch") as mock_dispatch, \
         patch("app.webhook.save_history"):
        mock_col.return_value.query.return_value = {"distances": [[1.0]], "documents": [[]], "metadatas": [[]]}
        photo_client.post("/webhook", json=INCOMING_PHOTO)

    _, dispatched_result, _ = mock_dispatch.call_args[0]
    assert dispatched_result["message_type"] == "text", (
        "Without a real image, media must downgrade to text"
    )
    assert dispatched_result["image_url"] is None
