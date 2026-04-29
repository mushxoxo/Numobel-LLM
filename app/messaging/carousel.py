import logging
import os
import requests
from dotenv import load_dotenv

load_dotenv()

log = logging.getLogger('rag_chatbot')

_API_KEY    = os.getenv("WA2MATION_API_KEY")
_VENDOR_UID = os.getenv("WA2MATION_VENDOR_UID")
_URL        = f"https://wa2mation.com/api/{_VENDOR_UID}/contact/send-carousel-template-message"
_HEADERS    = {"Authorization": f"Bearer {_API_KEY}", "Content-Type": "application/json"}
_TIMEOUT    = 10


def send_carousel(
    phone: str,
    template_name: str,
    cards: list[dict],
    language: str = "en",
    body_var: str = "",
) -> requests.Response:
    """Send a carousel template message.

    Each card dict: {"media_url": str, "media_type": "IMAGE"|"VIDEO", "button_type": [...]}
    body_var maps to field_1 in the template.
    """
    payload = {
        "phone_number":       phone,
        "template_name":      template_name,
        "template_language":  language,
        "carousel_templates": cards,
    }
    if body_var:
        payload["field_1"] = body_var
    resp = requests.post(_URL, json=payload, headers=_HEADERS, timeout=_TIMEOUT)
    if resp.status_code != 200:
        log.warning("send_carousel failed | status=%d body=%s", resp.status_code, resp.text[:200])
    return resp
