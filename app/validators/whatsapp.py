"""WhatsApp message constraint enforcement.

Validates and corrects message content before dispatch:
- Truncates button labels that exceed the WhatsApp character limit
- Downgrades interactive messages with too many buttons to text + numbered list
- Truncates message body content per message type limits
"""
from app.config import (
    WA_TEXT_MAX_CHARS,
    WA_INTERACTIVE_BODY_MAX_CHARS,
    WA_CAROUSEL_CARD_BODY_MAX_CHARS,
    WA_INTERACTIVE_MAX_BUTTONS,
    WA_BUTTON_LABEL_MAX_CHARS,
)
from app.log import get_logger

__all__ = ["validate_whatsapp_response"]

log = get_logger()

_BODY_LIMITS = {
    "text":        WA_TEXT_MAX_CHARS,
    "media":       WA_TEXT_MAX_CHARS,
    "interactive": WA_INTERACTIVE_BODY_MAX_CHARS,
    "carousel":    WA_CAROUSEL_CARD_BODY_MAX_CHARS,
}


def _truncate_label(label: str) -> str:
    """Truncate button label to WA_BUTTON_LABEL_MAX_CHARS using ASCII '...'."""
    if len(label) > WA_BUTTON_LABEL_MAX_CHARS:
        return label[: WA_BUTTON_LABEL_MAX_CHARS - 3] + "..."
    return label


def _truncate_body(s: str, limit: int) -> str:
    """Truncate body string to exactly limit chars using Unicode ellipsis '…'."""
    if len(s) > limit:
        return s[: limit - 1] + "…"
    return s


def validate_whatsapp_response(result: dict) -> dict:
    """Enforce WhatsApp constraints on a generate_answer() result dict.

    Pure function — returns a new dict, does not mutate input.
    """
    r            = dict(result)  # shallow copy
    message_type = r.get("message_type", "text")
    content      = r.get("content", "") or ""
    buttons      = list(r.get("buttons") or [])  # copy so we can compare original
    image_url    = r.get("image_url")

    # Step 1: Truncate each button label (keep original list for overflow numbered list)
    original_buttons = list(buttons)
    buttons = [_truncate_label(b) for b in buttons]

    # Step 2: Button overflow — downgrade interactive to text + numbered list
    if message_type == "interactive" and len(buttons) > WA_INTERACTIVE_MAX_BUTTONS:
        n = len(buttons)
        log.warning(
            "VALIDATE_WHATSAPP | button overflow (%d buttons) — downgrading to text", n
        )
        numbered     = "\n\n" + "\n".join(f"{i + 1}. {b}" for i, b in enumerate(original_buttons))
        content      = content + numbered
        message_type = "text"
        buttons      = None
        image_url    = None

    # Step 3: Truncate body per message type
    limit   = _BODY_LIMITS.get(message_type, WA_TEXT_MAX_CHARS)
    content = _truncate_body(content, limit)

    return {
        "message_type": message_type,
        "content":      content,
        "buttons":      buttons if buttons is not None else None,
        "image_url":    image_url,
        **{k: v for k, v in result.items() if k not in ("message_type", "content", "buttons", "image_url")},
    }
