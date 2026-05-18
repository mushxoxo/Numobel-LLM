"""Media throttling tests for response-quality-regression debug session.

Behavior:
  - First media for a product → send_media called.
  - Repeat media for SAME product (same URL or product_name in recent history)
    → downgraded to send_text.
  - Media for a DIFFERENT product → send_media called again.
  - When throttled AND product_link is in hit metadata AND content does not
    already include the link → "Buy here: <url>" appended to text response.
"""
from unittest.mock import patch

PHONE = "919999999998"  # distinct phone to avoid clobbering test_router.py state


def _result(image_url=None, content="See attached"):
    return {
        "message_type": "media",
        "content": content,
        "buttons": None,
        "image_url": image_url,
    }


def _hit(images, product_name="MDF Perforated", product_link=None):
    meta = {"images": images, "product_name": product_name}
    if product_link:
        meta["product_link"] = product_link
    return {"metadata": meta}


# ─── Throttling logic via load_history mock ──────────────────────────────────

def test_media_not_throttled_when_history_empty():
    """No prior assistant turns → media sent normally."""
    with patch("app.router.load_history", return_value=[]), \
         patch("app.router.send_media") as mock_media, \
         patch("app.router.send_text") as mock_text:
        from app.router import dispatch
        dispatch(PHONE, _result(image_url="https://example.com/a.jpg"))
        mock_media.assert_called_once()
        mock_text.assert_not_called()


def test_media_throttled_when_same_url_in_recent_history():
    """Same image_url in last 4 assistant turns → downgraded to text."""
    history = [
        {"role": "user", "content": "tell me about acoustic panels"},
        {"role": "assistant", "content": "Here is info",
         "image_url": "https://example.com/a.jpg", "product_name": "MDF Perforated"},
    ]
    with patch("app.router.load_history", return_value=history), \
         patch("app.router.send_media") as mock_media, \
         patch("app.router.send_text") as mock_text:
        from app.router import dispatch
        hits = [_hit("https://example.com/a.jpg", product_name="MDF Perforated")]
        dispatch(PHONE, _result(image_url="https://example.com/a.jpg"), hits=hits)
        mock_text.assert_called_once_with(PHONE, "See attached")
        mock_media.assert_not_called()


def test_media_throttled_when_same_product_name_in_history():
    """Same product_name in history (even if URL differs) → downgrade."""
    history = [
        {"role": "user", "content": "show me"},
        {"role": "assistant", "content": "Here is info",
         "image_url": "https://example.com/old.jpg", "product_name": "MDF Perforated"},
    ]
    with patch("app.router.load_history", return_value=history), \
         patch("app.router.send_media") as mock_media, \
         patch("app.router.send_text") as mock_text:
        from app.router import dispatch
        # Different URL but same product_name from hits
        hits = [_hit("https://example.com/new.jpg", product_name="MDF Perforated")]
        dispatch(PHONE, _result(image_url="https://example.com/new.jpg"), hits=hits)
        mock_text.assert_called_once()
        mock_media.assert_not_called()


def test_media_not_throttled_on_product_context_switch():
    """Different product_name AND different URL → media allowed."""
    history = [
        {"role": "user", "content": "mdf panels"},
        {"role": "assistant", "content": "Here is info",
         "image_url": "https://example.com/mdf.jpg", "product_name": "MDF Perforated"},
    ]
    with patch("app.router.load_history", return_value=history), \
         patch("app.router.send_media") as mock_media, \
         patch("app.router.send_text") as mock_text:
        from app.router import dispatch
        # Switch to a different product
        hits = [_hit("https://example.com/oil2c.jpg", product_name="Oil2C")]
        dispatch(PHONE, _result(image_url="https://example.com/oil2c.jpg"), hits=hits)
        mock_media.assert_called_once()
        mock_text.assert_not_called()


def test_media_throttle_window_respects_recent_turns_only():
    """A match outside the recent _MEDIA_THROTTLE_WINDOW does NOT throttle."""
    # Construct history with the matching URL at the very beginning, then 4+
    # later assistant turns (text-only) so the matching turn is outside the window.
    history = [
        {"role": "user", "content": "q1"},
        {"role": "assistant", "content": "first",
         "image_url": "https://example.com/a.jpg", "product_name": "MDF Perforated"},
        {"role": "user", "content": "q2"},
        {"role": "assistant", "content": "t2"},
        {"role": "user", "content": "q3"},
        {"role": "assistant", "content": "t3"},
        {"role": "user", "content": "q4"},
        {"role": "assistant", "content": "t4"},
        {"role": "user", "content": "q5"},
        {"role": "assistant", "content": "t5"},
    ]
    with patch("app.router.load_history", return_value=history), \
         patch("app.router.send_media") as mock_media, \
         patch("app.router.send_text") as mock_text:
        from app.router import dispatch
        hits = [_hit("https://example.com/a.jpg", product_name="MDF Perforated")]
        dispatch(PHONE, _result(image_url="https://example.com/a.jpg"), hits=hits)
        # Old matching turn is outside the last-4 window → media allowed
        mock_media.assert_called_once()
        mock_text.assert_not_called()


def test_text_dispatch_unaffected_by_throttling():
    """Throttling only applies to media; text dispatch is independent of history."""
    history = [
        {"role": "assistant", "content": "x",
         "image_url": "https://example.com/a.jpg", "product_name": "MDF Perforated"},
    ]
    with patch("app.router.load_history", return_value=history), \
         patch("app.router.send_text") as mock_text:
        from app.router import dispatch
        dispatch(PHONE, {"message_type": "text", "content": "plain text answer",
                         "buttons": None, "image_url": None})
        mock_text.assert_called_once_with(PHONE, "plain text answer")


def test_media_throttle_load_history_failure_does_not_break_dispatch():
    """If load_history raises, dispatch must still send media (fail-open)."""
    with patch("app.router.load_history", side_effect=Exception("disk error")), \
         patch("app.router.send_media") as mock_media:
        from app.router import dispatch
        dispatch(PHONE, _result(image_url="https://example.com/a.jpg"))
        mock_media.assert_called_once()


# ─── Bug 2 regression: product link preserved when media is throttled ─────────

def test_throttled_media_appends_product_link_when_available():
    """Bug 2 regression: product_link from hit metadata is appended to throttled text response.

    When the user asks 'where can I buy X' and the media throttle fires (same product
    was shown recently), the product purchase URL must still appear in the text fallback
    so the user gets the link they asked for.
    """
    history = [
        {"role": "user", "content": "tell me about kiddo stool"},
        {"role": "assistant", "content": "Here is the Kiddo Stool",
         "image_url": "https://static.wixstatic.com/kiddo.jpg",
         "product_name": "Nutoy-Kiddo-Desk and Stool Set"},
    ]
    product_url = "https://www.numobel.in/product/nutoy-kiddo-stool"
    with patch("app.router.load_history", return_value=history), \
         patch("app.router.send_media") as mock_media, \
         patch("app.router.send_text") as mock_text:
        from app.router import dispatch
        hits = [_hit(
            "https://static.wixstatic.com/kiddo.jpg",
            product_name="Nutoy-Kiddo-Desk and Stool Set",
            product_link=product_url,
        )]
        dispatch(
            PHONE,
            _result(
                image_url="https://static.wixstatic.com/kiddo.jpg",
                content="You can purchase the Nutoy Kiddo Desk and Stool Set from Numobel.",
            ),
            hits=hits,
        )
        mock_media.assert_not_called()
        mock_text.assert_called_once()
        sent_text = mock_text.call_args[0][1]
        assert product_url in sent_text, (
            f"Bug 2 regression: product link was not appended to throttled text response. "
            f"Got: {sent_text!r}"
        )
        assert "Buy here:" in sent_text


def test_throttled_media_no_link_appended_when_no_product_link():
    """When hit metadata has no product_link, throttled text is sent unchanged."""
    history = [
        {"role": "assistant", "content": "img",
         "image_url": "https://example.com/a.jpg", "product_name": "MDF Perforated"},
    ]
    with patch("app.router.load_history", return_value=history), \
         patch("app.router.send_text") as mock_text:
        from app.router import dispatch
        hits = [_hit("https://example.com/a.jpg", product_name="MDF Perforated")]
        dispatch(PHONE, _result(image_url="https://example.com/a.jpg", content="Here is info"),
                 hits=hits)
        sent_text = mock_text.call_args[0][1]
        assert sent_text == "Here is info", (
            f"No product_link in metadata — content must be unchanged. Got: {sent_text!r}"
        )


def test_throttled_media_no_duplicate_link_when_content_already_has_it():
    """If content already contains the product_link, it must not be appended again."""
    product_url = "https://www.numobel.in/product/nutoy-kiddo-stool"
    content_with_link = f"Buy the Kiddo Stool here: {product_url}"
    history = [
        {"role": "assistant", "content": "prior",
         "image_url": "https://example.com/kiddo.jpg",
         "product_name": "Nutoy-Kiddo-Desk and Stool Set"},
    ]
    with patch("app.router.load_history", return_value=history), \
         patch("app.router.send_text") as mock_text:
        from app.router import dispatch
        hits = [_hit(
            "https://example.com/kiddo.jpg",
            product_name="Nutoy-Kiddo-Desk and Stool Set",
            product_link=product_url,
        )]
        dispatch(PHONE, _result(image_url="https://example.com/kiddo.jpg",
                                content=content_with_link), hits=hits)
        sent_text = mock_text.call_args[0][1]
        # Link must appear exactly once — not duplicated
        assert sent_text.count(product_url) == 1, (
            f"Product link must not be duplicated. Got: {sent_text!r}"
        )


# ─── Bug 1b regression: empty content safety net ────────────────────────────

def test_empty_content_sends_fallback_not_422():
    """Bug 1b regression: dispatch must never forward empty content to send_text.

    wa2mation returns 422 when message_body is empty/whitespace. The safety net
    in the case _ branch must substitute a fallback message instead.
    """
    with patch("app.router.load_history", return_value=[]), \
         patch("app.router.send_text") as mock_text:
        from app.router import dispatch
        dispatch(PHONE, {"message_type": "text", "content": "", "buttons": None, "image_url": None})
        mock_text.assert_called_once()
        sent = mock_text.call_args[0][1]
        assert sent.strip(), "dispatch sent empty content to send_text — would cause 422"


def test_whitespace_only_content_sends_fallback():
    """Whitespace-only content must also trigger the fallback, not be forwarded."""
    with patch("app.router.load_history", return_value=[]), \
         patch("app.router.send_text") as mock_text:
        from app.router import dispatch
        dispatch(PHONE, {"message_type": "text", "content": "   ", "buttons": None, "image_url": None})
        mock_text.assert_called_once()
        sent = mock_text.call_args[0][1]
        assert sent.strip(), "dispatch forwarded whitespace-only content — would cause 422"


# ─── Bug 2 regression: product link injected for purchase-intent text responses ─

def test_purchase_intent_text_appends_product_link():
    """Bug 2 regression: 'where can i buy it?' (non-throttled path) must include product link.

    The throttle downgrade path already appends the link. This test covers the base
    case: a fresh text response where no media was previously sent (throttle never fires)
    but the content contains purchase-intent keywords.
    """
    product_url = "https://www.numobel.in/product/nuacoustics-cloud-hexagon"
    with patch("app.router.load_history", return_value=[]), \
         patch("app.router.send_text") as mock_text:
        from app.router import dispatch
        hits = [_hit("https://static.wixstatic.com/hexagon.jpg",
                     product_name="Nuacoustics-Cloud-Hexagon",
                     product_link=product_url)]
        dispatch(
            PHONE,
            {"message_type": "text",
             "content": "You can purchase the Cloud Hexagon panel from Numobel.",
             "buttons": None, "image_url": None},
            hits=hits,
        )
        mock_text.assert_called_once()
        sent = mock_text.call_args[0][1]
        assert product_url in sent, (
            f"Bug 2 regression: product link not appended for purchase-intent text. Got: {sent!r}"
        )
        assert "Buy here:" in sent


def test_purchase_intent_text_no_duplicate_link():
    """Product link must not be appended if content already contains it."""
    product_url = "https://www.numobel.in/product/nuacoustics-cloud-hexagon"
    content_with_link = f"Buy here: {product_url}"
    with patch("app.router.load_history", return_value=[]), \
         patch("app.router.send_text") as mock_text:
        from app.router import dispatch
        hits = [_hit("https://static.wixstatic.com/hexagon.jpg",
                     product_name="Nuacoustics-Cloud-Hexagon",
                     product_link=product_url)]
        dispatch(
            PHONE,
            {"message_type": "text", "content": content_with_link, "buttons": None, "image_url": None},
            hits=hits,
        )
        sent = mock_text.call_args[0][1]
        assert sent.count(product_url) == 1, f"Product link duplicated. Got: {sent!r}"


def test_non_purchase_text_does_not_get_link_appended():
    """Product info queries without purchase keywords must NOT get the link appended."""
    product_url = "https://www.numobel.in/product/nuacoustics-cloud-hexagon"
    with patch("app.router.load_history", return_value=[]), \
         patch("app.router.send_text") as mock_text:
        from app.router import dispatch
        hits = [_hit("https://static.wixstatic.com/hexagon.jpg",
                     product_name="Nuacoustics-Cloud-Hexagon",
                     product_link=product_url)]
        dispatch(
            PHONE,
            {"message_type": "text",
             "content": "The Cloud Hexagon is a premium acoustic ceiling panel.",
             "buttons": None, "image_url": None},
            hits=hits,
        )
        sent = mock_text.call_args[0][1]
        assert product_url not in sent, (
            f"Product link must not be appended to non-purchase text. Got: {sent!r}"
        )


# ─── Bug 2 regression: time-based TTL prevents stale-turn throttling ──────────

def test_media_not_throttled_when_sent_at_is_beyond_ttl():
    """Bug 2 regression: media sent >5 minutes ago must NOT throttle a fresh request.

    Scenario: Cloud-Hexagon image shown at 15:25. User switches topic at 15:41
    (16 minutes later). The matching turn is within the last-4 window but its
    sent_at exceeds _MEDIA_THROTTLE_TTL_SECONDS (300 s). Throttle must NOT fire.
    """
    import datetime
    old_ts = (datetime.datetime.now() - datetime.timedelta(minutes=16)).isoformat(timespec="seconds")
    history = [
        {"role": "user", "content": "tell me about ceiling decorations"},
        {"role": "assistant", "content": "Here is Cloud-Hexagon",
         "image_url": "https://static.wixstatic.com/hexagon.jpg",
         "product_name": "Numobel Acoustics-PET Ceiling Cloud-Hexagon",
         "sent_at": old_ts},
    ]
    with patch("app.router.load_history", return_value=history),          patch("app.router.send_media") as mock_media,          patch("app.router.send_text") as mock_text:
        from app.router import dispatch
        hits = [_hit(
            "https://static.wixstatic.com/hexagon.jpg",
            product_name="Numobel Acoustics-PET Ceiling Cloud-Hexagon",
        )]
        dispatch(
            PHONE,
            _result(image_url="https://static.wixstatic.com/hexagon.jpg"),
            hits=hits,
        )
        mock_media.assert_called_once(), (
            "Bug 2 regression: throttle must not fire for turns older than the TTL"
        )
        mock_text.assert_not_called()


def test_media_throttled_when_sent_at_is_within_ttl():
    """Throttle MUST fire when the matching turn is within the TTL window."""
    import datetime
    recent_ts = (datetime.datetime.now() - datetime.timedelta(minutes=2)).isoformat(timespec="seconds")
    history = [
        {"role": "user", "content": "show ceiling panel"},
        {"role": "assistant", "content": "Here is the panel",
         "image_url": "https://static.wixstatic.com/hexagon.jpg",
         "product_name": "Numobel Acoustics-PET Ceiling Cloud-Hexagon",
         "sent_at": recent_ts},
    ]
    with patch("app.router.load_history", return_value=history),          patch("app.router.send_media") as mock_media,          patch("app.router.send_text") as mock_text:
        from app.router import dispatch
        hits = [_hit(
            "https://static.wixstatic.com/hexagon.jpg",
            product_name="Numobel Acoustics-PET Ceiling Cloud-Hexagon",
        )]
        dispatch(
            PHONE,
            _result(image_url="https://static.wixstatic.com/hexagon.jpg"),
            hits=hits,
        )
        mock_text.assert_called_once()
        mock_media.assert_not_called()


def test_media_throttle_missing_sent_at_falls_back_to_turn_window():
    """Turns without sent_at (old sessions) fall back to turn-window check only."""
    # No sent_at field — backward compat: throttle fires on matching URL.
    history = [
        {"role": "assistant", "content": "prior",
         "image_url": "https://example.com/a.jpg",
         "product_name": "MDF Perforated"},
        # no sent_at
    ]
    with patch("app.router.load_history", return_value=history),          patch("app.router.send_media") as mock_media,          patch("app.router.send_text") as mock_text:
        from app.router import dispatch
        hits = [_hit("https://example.com/a.jpg", product_name="MDF Perforated")]
        dispatch(PHONE, _result(image_url="https://example.com/a.jpg"), hits=hits)
        mock_text.assert_called_once()
        mock_media.assert_not_called()


# ─── Bug 1 regression: product link injected into normal media caption ──────────

def test_media_caption_gets_product_link_on_purchase_intent():
    """Bug 1 regression: non-throttled media path must append product_link to caption
    when response content contains purchase/link intent keywords.

    Previously, only the throttled-downgrade path and the text path injected the link.
    The normal send_media call silently dropped it.
    """
    product_url = "https://www.numobel.in/product-page/numobelacoustics-mdf-1"
    content = "You can find our Numobel Acoustics products on our official website."
    with patch("app.router.load_history", return_value=[]), \
         patch("app.router.send_media") as mock_media, \
         patch("app.router.send_text") as mock_text:
        from app.router import dispatch
        hits = [_hit(
            "https://static.wixstatic.com/mdf.jpg",
            product_name="Numobel acoustics-MDF Perforated",
            product_link=product_url,
        )]
        dispatch(
            PHONE,
            _result(image_url="https://static.wixstatic.com/mdf.jpg", content=content),
            hits=hits,
        )
        mock_media.assert_called_once()
        mock_text.assert_not_called()
        _, kwargs = mock_media.call_args
        caption = kwargs.get("caption", mock_media.call_args[0][1] if len(mock_media.call_args[0]) > 1 else "")
        assert product_url in caption, (
            f"Bug 1 regression: product_link not appended to media caption. Got: {caption!r}"
        )
        assert "Buy here:" in caption


def test_media_caption_no_link_when_no_purchase_intent():
    """Media caption must NOT get product link appended when content has no purchase intent."""
    product_url = "https://www.numobel.in/product-page/numobelacoustics-mdf-1"
    content = "The MDF Perforated panel absorbs mid to high frequency sound waves."
    with patch("app.router.load_history", return_value=[]), \
         patch("app.router.send_media") as mock_media:
        from app.router import dispatch
        hits = [_hit(
            "https://static.wixstatic.com/mdf.jpg",
            product_name="Numobel acoustics-MDF Perforated",
            product_link=product_url,
        )]
        dispatch(
            PHONE,
            _result(image_url="https://static.wixstatic.com/mdf.jpg", content=content),
            hits=hits,
        )
        mock_media.assert_called_once()
        _, kwargs = mock_media.call_args
        caption = kwargs.get("caption", mock_media.call_args[0][1] if len(mock_media.call_args[0]) > 1 else "")
        assert product_url not in caption, (
            f"Product link must not appear in non-purchase media caption. Got: {caption!r}"
        )


def test_media_caption_no_duplicate_link():
    """If product_link already appears in content, it must not be appended again."""
    product_url = "https://www.numobel.in/product-page/numobelacoustics-mdf-1"
    content = f"Visit our official website at {product_url} for more information."
    with patch("app.router.load_history", return_value=[]), \
         patch("app.router.send_media") as mock_media:
        from app.router import dispatch
        hits = [_hit(
            "https://static.wixstatic.com/mdf.jpg",
            product_name="Numobel acoustics-MDF Perforated",
            product_link=product_url,
        )]
        dispatch(
            PHONE,
            _result(image_url="https://static.wixstatic.com/mdf.jpg", content=content),
            hits=hits,
        )
        mock_media.assert_called_once()
        _, kwargs = mock_media.call_args
        caption = kwargs.get("caption", mock_media.call_args[0][1] if len(mock_media.call_args[0]) > 1 else "")
        assert caption.count(product_url) == 1, (
            f"Product link must not be duplicated in caption. Got: {caption!r}"
        )


def test_media_caption_link_keyword_triggers_injection():
    """'link' keyword in content triggers product link injection into caption."""
    product_url = "https://www.numobel.in/product-page/numobelacoustics-mdf-1"
    content = "Please visit Nupanel for more information. You can find the link below."
    with patch("app.router.load_history", return_value=[]), \
         patch("app.router.send_media") as mock_media:
        from app.router import dispatch
        hits = [_hit(
            "https://static.wixstatic.com/mdf.jpg",
            product_name="Numobel acoustics-MDF Perforated",
            product_link=product_url,
        )]
        dispatch(
            PHONE,
            _result(image_url="https://static.wixstatic.com/mdf.jpg", content=content),
            hits=hits,
        )
        mock_media.assert_called_once()
        _, kwargs = mock_media.call_args
        caption = kwargs.get("caption", mock_media.call_args[0][1] if len(mock_media.call_args[0]) > 1 else "")
        assert product_url in caption, (
            f"Bug 1 regression: 'link' keyword must trigger product_link injection. Got: {caption!r}"
        )
