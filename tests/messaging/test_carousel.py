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
    send_carousel("919999999999", "numobel_catalogue_4", CARDS)
    assert mock_post.call_args[0][0] == "send-carousel-template-message"


def test_send_carousel_required_fields(mock_post):
    from app.messaging.carousel import send_carousel
    send_carousel("919999999999", "numobel_catalogue_4", CARDS, language="en")
    payload = mock_post.call_args[0][1]
    assert payload["phone_number"] == "919999999999"
    assert payload["template_name"] == "numobel_catalogue_4"
    assert payload["template_language"] == "en"
    assert payload["carousel_templates"] == CARDS


def test_send_carousel_body_vars_as_field_1_and_field_2(mock_post):
    from app.messaging.carousel import send_carousel
    send_carousel("919999999999", "numobel_catalogue_4", CARDS, body_vars=["Nutoy Stackers", "Handcrafted wooden toys"])
    payload = mock_post.call_args[0][1]
    assert payload["field_1"] == "Nutoy Stackers"
    assert payload["field_2"] == "Handcrafted wooden toys"


def test_send_carousel_single_body_var(mock_post):
    from app.messaging.carousel import send_carousel
    send_carousel("919999999999", "numobel_catalogue_4", CARDS, body_vars=["Nutoy Stackers"])
    payload = mock_post.call_args[0][1]
    assert payload["field_1"] == "Nutoy Stackers"
    assert "field_2" not in payload


def test_send_carousel_fields_omitted_when_no_body_vars(mock_post):
    from app.messaging.carousel import send_carousel
    send_carousel("919999999999", "numobel_catalogue_4", CARDS)
    payload = mock_post.call_args[0][1]
    assert "field_1" not in payload
    assert "field_2" not in payload


def test_send_carousel_default_language_is_en(mock_post):
    from app.messaging.carousel import send_carousel
    send_carousel("919999999999", "numobel_catalogue_4", CARDS)
    assert mock_post.call_args[0][1]["template_language"] == "en"
