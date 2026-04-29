import logging
import os
import requests
from dotenv import load_dotenv

load_dotenv()

log = logging.getLogger('rag_chatbot')

_API_KEY    = os.getenv("WA2MATION_API_KEY")
_VENDOR_UID = os.getenv("WA2MATION_VENDOR_UID")
_URL        = f"https://wa2mation.com/api/{_VENDOR_UID}/contact/send-interactive-message"
_HEADERS    = {"Authorization": f"Bearer {_API_KEY}", "Content-Type": "application/json"}
_TIMEOUT    = 10


def send_interactive(
    phone: str,
    body: str,
    buttons: list[str],
    header: str = "",
    footer: str = "",
) -> requests.Response:
    """Send a button interactive message. buttons is a list of up to 3 label strings."""
    if len(buttons) > 3:
        log.warning("send_interactive | %d buttons provided, truncating to 3", len(buttons))
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
    resp = requests.post(_URL, json=payload, headers=_HEADERS, timeout=_TIMEOUT)
    if resp.status_code != 200:
        log.warning("send_interactive failed | status=%d body=%s", resp.status_code, resp.text[:200])
    return resp
