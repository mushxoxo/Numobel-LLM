import requests

from app.log import get_logger
from app.messaging.client import wa2mation_post

log = get_logger()


def send_interactive(
    phone: str,
    body: str,
    buttons: list[str],
    header: str = "",
    footer: str = "",
) -> requests.Response:
    """Send a button interactive message. buttons is a list of up to 3 label strings."""
    if len(buttons) > 3:
        log.warning("SEND_INTERACTIVE | received %d buttons, truncating to 3 — validator should have caught this", len(buttons))
    truncated = [label for label in buttons if len(label) > 20]
    if truncated:
        log.warning("send_interactive | button label(s) truncated to 20 chars: %s", truncated)
    buttons = [label[:20] for label in buttons]

    payload = {
        "phone_number":     phone,
        "interactive_type": "button",
        "body_text":        body,
        "buttons":          {str(i + 1): label for i, label in enumerate(buttons[:3])},
    }
    if header:
        payload["header_type"] = "text"
        payload["header_text"] = header
    if footer:
        payload["footer_text"] = footer

    resp = wa2mation_post("send-interactive-message", payload)
    if resp.status_code != 200:
        log.warning("send_interactive failed | status=%d body=%s", resp.status_code, resp.text[:200])
    return resp
