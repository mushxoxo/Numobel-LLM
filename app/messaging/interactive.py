import os
import requests
from dotenv import load_dotenv

load_dotenv()

_API_KEY    = os.getenv("WA2MATION_API_KEY")
_VENDOR_UID = os.getenv("WA2MATION_VENDOR_UID")
_URL        = f"https://wa2mation.com/api/{_VENDOR_UID}/contact/send-interactive-message"
_HEADERS    = {"Authorization": f"Bearer {_API_KEY}", "Content-Type": "application/json"}


def send_interactive(
    phone: str,
    body: str,
    buttons: list[str],
    header: str = "",
    footer: str = "",
) -> requests.Response:
    """Send a button interactive message. buttons is a list of up to 3 label strings."""
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
    return requests.post(_URL, json=payload, headers=_HEADERS)
