import logging
import os
import requests
from dotenv import load_dotenv

load_dotenv()

log = logging.getLogger('rag_chatbot')

_TIMEOUT = 10


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
    api_key    = os.getenv("WA2MATION_API_KEY")
    vendor_uid = os.getenv("WA2MATION_VENDOR_UID")
    url     = f"https://wa2mation.com/api/{vendor_uid}/contact/send-carousel-template-message"
    headers = {"Authorization": f"Bearer {api_key}", "Content-Type": "application/json"}
    payload = {
        "phone_number":       phone,
        "template_name":      template_name,
        "template_language":  language,
        "carousel_templates": cards,
    }
    if body_var:
        payload["field_1"] = body_var
    resp = requests.post(url, json=payload, headers=headers, timeout=_TIMEOUT)
    if resp.status_code != 200:
        log.warning("send_carousel failed | status=%d body=%s", resp.status_code, resp.text[:200])
    return resp
