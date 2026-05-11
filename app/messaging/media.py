import requests

from app.log import get_logger
from app.messaging.client import wa2mation_post

log = get_logger()


def send_media(phone: str, url: str, caption: str = "", media_type: str = "image") -> requests.Response:
    resp = wa2mation_post(
        "send-media-message",
        {"phone_number": phone, "media_type": media_type, "media_url": url, "caption": caption},
    )
    if resp.status_code != 200:
        log.warning("send_media failed | status=%d body=%s", resp.status_code, resp.text[:200])
    return resp
