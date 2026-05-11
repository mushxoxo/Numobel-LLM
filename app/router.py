from app.log import get_logger
from app.messaging import send_text, send_media, send_interactive, send_carousel

log = get_logger()

# Only approved carousel template. Expand as more are approved.
_CAROUSEL_TEMPLATE  = "nutoy_stacker"
_CAROUSEL_MAX_CARDS = 2  # nutoy_stacker supports exactly 2 cards


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


def dispatch(phone: str, result: dict, hits: list[dict] = None) -> None:
    """Route a generate_answer() result to the correct wa2mation send function.

    Single responsibility — no LLM calls, no business logic.
    Falls back to send_text when required media/carousel assets are unavailable.
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
                send_media(phone, url=url, caption=content)
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
                send_carousel(
                    phone,
                    template_name=_CAROUSEL_TEMPLATE,
                    cards=cards,
                    body_var=(content[:60] if content else ""),
                )
            else:
                log.warning("DISPATCH | carousel needs 2+ images, only %d found — falling back to text", len(images))
                send_text(phone, content)

        case _:
            send_text(phone, content)
