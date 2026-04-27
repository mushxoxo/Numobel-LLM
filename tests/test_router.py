import pytest
from unittest.mock import patch, call

PHONE = "919999999999"

# Helpers
def _result(message_type, content="Hello", buttons=None, image_url=None):
    return {"message_type": message_type, "content": content, "buttons": buttons, "image_url": image_url}

def _hit(images="https://example.com/img1.jpg|https://example.com/img2.jpg"):
    return {"metadata": {"images": images}}


# ─── text ─────────────────────────────────────────────────────────────────────

def test_dispatch_text_calls_send_text():
    with patch("app.router.send_text") as mock:
        from app.router import dispatch
        dispatch(PHONE, _result("text", "Some answer"))
        mock.assert_called_once_with(PHONE, "Some answer")


def test_dispatch_unknown_type_falls_back_to_text():
    with patch("app.router.send_text") as mock:
        from app.router import dispatch
        dispatch(PHONE, _result("unknown", "Fallback"))
        mock.assert_called_once_with(PHONE, "Fallback")


# ─── interactive ──────────────────────────────────────────────────────────────

def test_dispatch_interactive_with_buttons():
    with patch("app.router.send_interactive") as mock:
        from app.router import dispatch
        dispatch(PHONE, _result("interactive", "Choose:", buttons=["A", "B", "C"]))
        mock.assert_called_once()
        kwargs = mock.call_args[1]
        assert kwargs["buttons"] == ["A", "B", "C"]
        assert kwargs["body"] == "Choose:"
        assert kwargs["header"] == "Numobel Assistant"
        assert kwargs["footer"] == "numobel.in"


def test_dispatch_interactive_no_buttons_falls_back_to_text():
    with patch("app.router.send_text") as mock_text, \
         patch("app.router.send_interactive") as mock_interactive:
        from app.router import dispatch
        dispatch(PHONE, _result("interactive", "No buttons here", buttons=None))
        mock_text.assert_called_once_with(PHONE, "No buttons here")
        mock_interactive.assert_not_called()


# ─── media ────────────────────────────────────────────────────────────────────

def test_dispatch_media_uses_image_url_from_result():
    with patch("app.router.send_media") as mock:
        from app.router import dispatch
        dispatch(PHONE, _result("media", "Nice product", image_url="https://example.com/img.jpg"))
        mock.assert_called_once_with(PHONE, url="https://example.com/img.jpg", caption="Nice product")


def test_dispatch_media_falls_back_to_hits_when_no_image_url():
    hits = [_hit("https://example.com/from_hits.jpg")]
    with patch("app.router.send_media") as mock:
        from app.router import dispatch
        dispatch(PHONE, _result("media", "Caption"), hits=hits)
        mock.assert_called_once_with(PHONE, url="https://example.com/from_hits.jpg", caption="Caption")


def test_dispatch_media_falls_back_to_text_when_no_image():
    with patch("app.router.send_text") as mock_text, \
         patch("app.router.send_media") as mock_media:
        from app.router import dispatch
        dispatch(PHONE, _result("media", "No image"), hits=None)
        mock_text.assert_called_once_with(PHONE, "No image")
        mock_media.assert_not_called()


# ─── carousel ─────────────────────────────────────────────────────────────────

def test_dispatch_carousel_sends_with_two_images():
    hits = [_hit("https://example.com/img1.jpg|https://example.com/img2.jpg")]
    with patch("app.router.send_carousel") as mock:
        from app.router import dispatch
        dispatch(PHONE, _result("carousel", "Nutoy Stackers"), hits=hits)
        mock.assert_called_once()
        kwargs = mock.call_args[1]
        assert kwargs["template_name"] == "nutoy_stacker"
        assert len(kwargs["cards"]) == 2
        assert kwargs["cards"][0]["media_url"] == "https://example.com/img1.jpg"


def test_dispatch_carousel_caps_at_max_cards():
    hits = [_hit("https://example.com/1.jpg|https://example.com/2.jpg|https://example.com/3.jpg")]
    with patch("app.router.send_carousel") as mock:
        from app.router import dispatch
        dispatch(PHONE, _result("carousel"), hits=hits)
        assert len(mock.call_args[1]["cards"]) == 2


def test_dispatch_carousel_falls_back_to_text_when_fewer_than_2_images():
    hits = [_hit("https://example.com/only_one.jpg")]
    with patch("app.router.send_text") as mock_text, \
         patch("app.router.send_carousel") as mock_carousel:
        from app.router import dispatch
        dispatch(PHONE, _result("carousel", "Fallback content"), hits=hits)
        mock_text.assert_called_once_with(PHONE, "Fallback content")
        mock_carousel.assert_not_called()


def test_dispatch_carousel_falls_back_to_text_when_no_hits():
    with patch("app.router.send_text") as mock_text, \
         patch("app.router.send_carousel") as mock_carousel:
        from app.router import dispatch
        dispatch(PHONE, _result("carousel", "No hits"), hits=None)
        mock_text.assert_called_once_with(PHONE, "No hits")
        mock_carousel.assert_not_called()


def test_dispatch_carousel_body_var_truncated_to_60_chars():
    hits = [_hit("https://example.com/img1.jpg|https://example.com/img2.jpg")]
    long_content = "A" * 100
    with patch("app.router.send_carousel") as mock:
        from app.router import dispatch
        dispatch(PHONE, _result("carousel", long_content), hits=hits)
        assert len(mock.call_args[1]["body_var"]) == 60


# ─── _images_from_hits ────────────────────────────────────────────────────────

def test_images_from_hits_deduplicates():
    hits = [
        _hit("https://example.com/img1.jpg|https://example.com/img2.jpg"),
        _hit("https://example.com/img1.jpg|https://example.com/img3.jpg"),
    ]
    from app.router import _images_from_hits
    images = _images_from_hits(hits)
    assert images.count("https://example.com/img1.jpg") == 1
    assert len(images) == 3


def test_images_from_hits_empty():
    from app.router import _images_from_hits
    assert _images_from_hits([]) == []
    assert _images_from_hits(None) == []
