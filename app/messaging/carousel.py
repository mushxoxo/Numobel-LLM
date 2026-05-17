import requests

from app.log import get_logger
from app.messaging.client import wa2mation_post

log = get_logger()


def send_carousel(
    phone: str,
    template_name: str,
    cards: list[dict],
    language: str = "en",
    body_vars: list[str] = None,
) -> requests.Response:
    """
    Send a carousel template message.
    Each card dict: {"media_url": str, "media_type": "IMAGE"|"VIDEO", "button_type": [...]}
    body_vars maps to field_1, field_2, … in the template.
    """
    payload = {
        "phone_number":       phone,
        "template_name":      template_name,
        "template_language":  language,
        "carousel_templates": cards,
    }
    for i, var in enumerate(body_vars or [], 1):
        if var:
            payload[f"field_{i}"] = var

    resp = wa2mation_post("send-carousel-template-message", payload)
    if resp.status_code != 200:
        log.warning("send_carousel failed | status=%d body=%s", resp.status_code, resp.text[:200])
    return resp
