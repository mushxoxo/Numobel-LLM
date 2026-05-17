import pytest

_mod = pytest.importorskip("app.validators.whatsapp")
validate_whatsapp_response = _mod.validate_whatsapp_response


def test_button_overflow_downgrade():
    result = validate_whatsapp_response({"message_type": "interactive", "content": "Pick:", "buttons": ["A", "B", "C", "D"], "image_url": None})
    assert result["message_type"] == "text"
    assert result["buttons"] is None or result["buttons"] == []
    assert "\n1. A" in result["content"]
    assert "\n4. D" in result["content"]


def test_button_label_truncation():
    label = "A" * 25
    result = validate_whatsapp_response({"message_type": "interactive", "content": "Pick:", "buttons": [label, "B", "C"], "image_url": None})
    assert result["buttons"][0].endswith("...")
    assert len(result["buttons"][0]) == 20


def test_body_truncation_text():
    result = validate_whatsapp_response({"message_type": "text", "content": "x" * 5000, "buttons": None, "image_url": None})
    assert len(result["content"]) == 4096
    assert result["content"].endswith("…")


def test_body_truncation_interactive():
    result = validate_whatsapp_response({"message_type": "interactive", "content": "x" * 2000, "buttons": ["A", "B"], "image_url": None})
    assert len(result["content"]) == 1024
    assert result["content"].endswith("…")


def test_body_truncation_carousel():
    result = validate_whatsapp_response({"message_type": "carousel", "content": "y" * 300, "buttons": None, "image_url": "http://x"})
    assert len(result["content"]) == 160
    assert result["content"].endswith("…")


def test_short_content_unchanged():
    result = validate_whatsapp_response({"message_type": "text", "content": "Hello", "buttons": None, "image_url": None})
    assert result["content"] == "Hello"


def test_three_buttons_kept():
    result = validate_whatsapp_response({"message_type": "interactive", "content": "Pick:", "buttons": ["A", "B", "C"], "image_url": None})
    assert result["message_type"] == "interactive"
    assert len(result["buttons"]) == 3


def test_button_label_under_limit_untouched():
    result = validate_whatsapp_response({"message_type": "interactive", "content": "Pick:", "buttons": ["ShortLabel"], "image_url": None})
    assert result["buttons"][0] == "ShortLabel"
