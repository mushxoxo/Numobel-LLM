import pytest
from unittest.mock import patch, MagicMock


@pytest.fixture(autouse=True)
def reset_client(monkeypatch):
    import app.messaging.client as client_mod
    monkeypatch.setattr(client_mod, "_session", None)


@pytest.fixture
def mock_post():
    with patch("app.messaging.media.wa2mation_post") as m:
        m.return_value = MagicMock(status_code=200)
        yield m


def test_send_media_posts_to_correct_endpoint(mock_post):
    from app.messaging.media import send_media
    send_media("919999999999", "https://example.com/img.jpg")
    assert mock_post.call_args[0][0] == "send-media-message"


def test_send_media_default_type_is_image(mock_post):
    from app.messaging.media import send_media
    send_media("919999999999", "https://example.com/img.jpg")
    assert mock_post.call_args[0][1]["media_type"] == "image"


def test_send_media_custom_type(mock_post):
    from app.messaging.media import send_media
    send_media("919999999999", "https://example.com/vid.mp4", media_type="video")
    assert mock_post.call_args[0][1]["media_type"] == "video"


def test_send_media_caption_included(mock_post):
    from app.messaging.media import send_media
    send_media("919999999999", "https://example.com/img.jpg", caption="Nice product")
    assert mock_post.call_args[0][1]["caption"] == "Nice product"


def test_send_media_payload_fields(mock_post):
    from app.messaging.media import send_media
    send_media("919999999999", "https://example.com/img.jpg", caption="Cap", media_type="image")
    payload = mock_post.call_args[0][1]
    assert payload["phone_number"] == "919999999999"
    assert payload["media_url"] == "https://example.com/img.jpg"
