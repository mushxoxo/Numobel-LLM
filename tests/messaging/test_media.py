import pytest
from unittest.mock import patch, MagicMock


@pytest.fixture
def mock_env(monkeypatch):
    monkeypatch.setenv("WA2MATION_API_KEY", "test-key")
    monkeypatch.setenv("WA2MATION_VENDOR_UID", "test-uid")


@pytest.fixture
def mock_post():
    with patch("app.messaging.media.requests.post") as m:
        m.return_value = MagicMock(status_code=200)
        yield m


def test_send_media_posts_to_correct_url(mock_env, mock_post):
    from app.messaging.media import send_media
    send_media("919999999999", "https://example.com/img.jpg")
    assert "contact/send-media-message" in mock_post.call_args[0][0]


def test_send_media_default_type_is_image(mock_env, mock_post):
    from app.messaging.media import send_media
    send_media("919999999999", "https://example.com/img.jpg")
    payload = mock_post.call_args[1]["json"]
    assert payload["media_type"] == "image"


def test_send_media_custom_type(mock_env, mock_post):
    from app.messaging.media import send_media
    send_media("919999999999", "https://example.com/vid.mp4", media_type="video")
    assert mock_post.call_args[1]["json"]["media_type"] == "video"


def test_send_media_caption_included(mock_env, mock_post):
    from app.messaging.media import send_media
    send_media("919999999999", "https://example.com/img.jpg", caption="Nice product")
    assert mock_post.call_args[1]["json"]["caption"] == "Nice product"


def test_send_media_payload_fields(mock_env, mock_post):
    from app.messaging.media import send_media
    send_media("919999999999", "https://example.com/img.jpg", caption="Cap", media_type="image")
    payload = mock_post.call_args[1]["json"]
    assert payload["phone_number"] == "919999999999"
    assert payload["media_url"] == "https://example.com/img.jpg"
