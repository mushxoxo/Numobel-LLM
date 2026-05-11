import pytest
from unittest.mock import patch, MagicMock

CARDS = [
    {"media_type": "IMAGE", "media_url": "https://example.com/1.jpg", "button_type": ["QUICK_REPLY", "URL"]},
    {"media_type": "IMAGE", "media_url": "https://example.com/2.jpg", "button_type": ["QUICK_REPLY", "URL"]},
]


@pytest.fixture(autouse=True)
def reset_client(monkeypatch):
    import app.messaging.client as client_mod
    monkeypatch.setattr(client_mod, "_session", None)


@pytest.fixture
def mock_post():
    with patch("app.messaging.carousel.wa2mation_post") as m:
        m.return_value = MagicMock(status_code=200)
        yield m


def test_send_carousel_correct_endpoint(mock_post):
    from app.messaging.carousel import send_carousel
    send_carousel("919999999999", "nutoy_stacker", CARDS)
    assert mock_post.call_args[0][0] == "send-carousel-template-message"


def test_send_carousel_required_fields(mock_post):
    from app.messaging.carousel import send_carousel
    send_carousel("919999999999", "nutoy_stacker", CARDS, language="en")
    payload = mock_post.call_args[0][1]
    assert payload["phone_number"] == "919999999999"
    assert payload["template_name"] == "nutoy_stacker"
    assert payload["template_language"] == "en"
    assert payload["carousel_templates"] == CARDS


def test_send_carousel_body_var_as_field_1(mock_post):
    from app.messaging.carousel import send_carousel
    send_carousel("919999999999", "nutoy_stacker", CARDS, body_var="Nutoy Stackers")
    assert mock_post.call_args[0][1]["field_1"] == "Nutoy Stackers"


def test_send_carousel_field_1_omitted_when_no_body_var(mock_post):
    from app.messaging.carousel import send_carousel
    send_carousel("919999999999", "nutoy_stacker", CARDS)
    assert "field_1" not in mock_post.call_args[0][1]


def test_send_carousel_default_language_is_en(mock_post):
    from app.messaging.carousel import send_carousel
    send_carousel("919999999999", "nutoy_stacker", CARDS)
    assert mock_post.call_args[0][1]["template_language"] == "en"
