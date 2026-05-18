import datetime

from app.config import CAROUSEL_TEMPLATE
from app.history import load_history
from app.log import get_logger
from app.messaging import send_text, send_media, send_interactive, send_carousel

log = get_logger()

_CAROUSEL_MAX_CARDS = 4  # numobel_catalogue_4 supports up to 4 cards

# Media throttling window: how many recent assistant turns to scan when deciding
# whether the current product image was already sent recently.
_MEDIA_THROTTLE_WINDOW = 4

# Time-based TTL for media throttling: images shown more than this many seconds
# ago are NOT throttled, even if they fall within the turn window. Prevents
# stale sessions from suppressing images on topic switches minutes later.
_MEDIA_THROTTLE_TTL_SECONDS = 300  # 5 minutes

# Purchase intent keywords — when any of these appear in the text response content,
# the product link from hit metadata is appended so the user gets a direct buy URL.
_PURCHASE_KWS = frozenset({
    "buy", "purchase", "order", "where can i get",
    "where to get", "how to buy", "get it",
    "website", "link", "url", "shop", "store", "official",
})


def _images_from_hits(hits: list[dict]) -> list[str]:
    """Extract unique image URLs from ChromaDB retrieval hits."""
    seen, images = set(), []
    for hit in hits or []:
        img_str = hit.get("metadata", {}).get("images", "")
        for url in img_str.split("|"):
            url = url.strip()
            if url and url not in seen:
                seen.add(url)
                images.append(url)
    return images


def _product_name_from_hits(hits: list[dict]) -> str | None:
    """Return the top hit's product_name, or None when unavailable."""
    if not hits:
        return None
    meta = hits[0].get("metadata", {}) or {}
    return meta.get("product_name") or meta.get("name")


def _product_link_from_hits(hits: list[dict]) -> str | None:
    """Return the top hit's product_link URL, or None when unavailable."""
    if not hits:
        return None
    meta = hits[0].get("metadata", {}) or {}
    return meta.get("product_link") or None


def _has_purchase_intent(content: str) -> bool:
    """Return True when content contains any purchase-intent keyword."""
    c_lower = (content or "").lower()
    return any(kw in c_lower for kw in _PURCHASE_KWS)


def _append_product_link(content: str, product_link: str | None) -> str:
    """Append 'Buy here: <url>' to content if product_link is present and not already included."""
    if product_link and product_link not in (content or ""):
        return f"{content}\n\nBuy here: {product_link}"
    return content


def _recent_media_already_sent(
    phone: str,
    image_url: str | None,
    product_name: str | None,
) -> bool:
    """Return True when an equivalent media message was sent in the recent window.

    A media is considered equivalent when ALL of the following are true for a
    prior assistant turn within `_MEDIA_THROTTLE_WINDOW`:
      1. It carries the same image_url OR the same product_name.
      2. Its `sent_at` timestamp is within `_MEDIA_THROTTLE_TTL_SECONDS` of now
         (turns older than the TTL are skipped — they belong to a different
         conversational context even if technically within the turn window).
         Turns that have no `sent_at` field (old sessions) fall back to the
         turn-window check only (backward compatible).

    Loading history via `load_history` is cheap (single JSON read) and gracefully
    returns [] when no session exists, so this is safe to call unconditionally.
    A None image_url and None product_name means we have nothing to compare —
    return False (no throttling).
    """
    if not image_url and not product_name:
        return False
    try:
        history = load_history(phone)
    except Exception:
        log.debug("DISPATCH | history load failed for media throttling — allowing media")
        return False

    now = datetime.datetime.now()
    assistant_turns = [m for m in history if m.get("role") == "assistant"]
    for turn in assistant_turns[-_MEDIA_THROTTLE_WINDOW:]:
        # Time-based TTL guard: skip turns that are too old to be relevant.
        sent_at_str = turn.get("sent_at")
        if sent_at_str:
            try:
                sent_at = datetime.datetime.fromisoformat(sent_at_str)
                if (now - sent_at).total_seconds() > _MEDIA_THROTTLE_TTL_SECONDS:
                    continue
            except (ValueError, TypeError):
                pass  # malformed sent_at — fall through to URL/name check

        prior_url     = turn.get("image_url")
        prior_product = turn.get("product_name")
        if image_url and prior_url and prior_url == image_url:
            return True
        if product_name and prior_product and prior_product == product_name:
            return True
    return False


def dispatch(phone: str, result: dict, hits: list[dict] = None) -> None:
    """Route a generate_answer() result to the correct wa2mation send function.

    Single responsibility — no LLM calls, no business logic.
    Falls back to send_text when required media/carousel assets are unavailable.

    Media throttling: when the same product image (by URL or product_name) was
    already sent in the last few assistant turns for this phone, downgrade the
    media response to text so the user is not spammed with the same image for
    every follow-up question about the same product. A switch to a different
    product (different URL / product_name) allows media through again.

    Product link injection: when content contains purchase-intent keywords and
    the top ChromaDB hit has a product_link, "Buy here: <url>" is appended to
    the response. This applies to both the normal media caption path and the
    text path (including throttled media downgrades).
    """
    message_type = result.get("message_type", "text")
    content      = result.get("content", "")
    buttons      = result.get("buttons") or []
    image_url    = result.get("image_url")

    log.debug("DISPATCH | type=%s phone=%s", message_type, phone)

    match message_type:
        case "interactive":
            if buttons:
                send_interactive(
                    phone,
                    body=content,
                    buttons=buttons,
                    header="Numobel Assistant",
                    footer="numobel.in",
                )
            else:
                send_text(phone, content)

        case "media":
            url = image_url or ((_images_from_hits(hits) or [None])[0])
            if url:
                product_name = _product_name_from_hits(hits)
                if _recent_media_already_sent(phone, url, product_name):
                    log.info(
                        "DISPATCH | media throttled (same product recently shown) — "
                        "downgrading to text. product=%s url=%s",
                        product_name, url,
                    )
                    # Preserve product link when throttling suppresses the image
                    # so explicit "where can I buy" / link requests still get the URL.
                    product_link = _product_link_from_hits(hits)
                    text_content = _append_product_link(content, product_link)
                    send_text(phone, text_content)
                else:
                    # Bug 1 fix: inject product link into media caption when the
                    # response contains purchase-intent keywords (buy, link, website,
                    # etc.). The throttled path above already did this; the normal
                    # media path was silently dropping the link.
                    if hits and _has_purchase_intent(content):
                        product_link = _product_link_from_hits(hits)
                        caption = _append_product_link(content, product_link)
                        log.debug(
                            "DISPATCH | media caption: purchase intent detected, "
                            "appending product_link product=%s",
                            product_name,
                        )
                    else:
                        caption = content
                    send_media(phone, url=url, caption=caption)
            else:
                log.warning("DISPATCH | media requested but no image available — falling back to text")
                send_text(phone, content)

        case "carousel":
            images = _images_from_hits(hits)
            if len(images) >= 2:
                cards = [
                    {"media_type": "IMAGE", "media_url": img, "button_type": ["QUICK_REPLY", "URL"]}
                    for img in images[:_CAROUSEL_MAX_CARDS]
                ]
                parts = (content or "").split("\n", 1)
                body_vars = [p.strip()[:60] for p in parts if p.strip()]
                send_carousel(
                    phone,
                    template_name=CAROUSEL_TEMPLATE,
                    cards=cards,
                    body_vars=body_vars,
                )
            else:
                log.warning("DISPATCH | carousel needs 2+ images, only %d found — falling back to text", len(images))
                send_text(phone, content)

        case _:
            # Inject product link for purchase-intent text responses.
            # This covers the non-throttled text path; the throttled media path above
            # and the normal media path both handle injection too.
            if content and hits and _has_purchase_intent(content):
                product_link = _product_link_from_hits(hits)
                content = _append_product_link(content, product_link)
            # Safety net — never send empty body (wa2mation returns 422).
            if not (content or "").strip():
                log.warning("DISPATCH | empty content for text response — sending fallback")
                content = "I'm sorry, I couldn't retrieve that information. Please try again."
            send_text(phone, content)
