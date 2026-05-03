import logging
import os
import requests
from dotenv import load_dotenv

load_dotenv()

log = logging.getLogger('rag_chatbot')

_TIMEOUT = 10


def send_media(phone: str, url: str, caption: str = "", media_type: str = "image") -> requests.Response:
    api_key    = os.getenv("WA2MATION_API_KEY")
    vendor_uid = os.getenv("WA2MATION_VENDOR_UID")
    endpoint   = f"https://wa2mation.com/api/{vendor_uid}/contact/send-media-message"
    headers    = {"Authorization": f"Bearer {api_key}", "Content-Type": "application/json"}
    resp = requests.post(
        endpoint,
        json={"phone_number": phone, "media_type": media_type, "media_url": url, "caption": caption},
        headers=headers,
        timeout=_TIMEOUT,
    )
    if resp.status_code != 200:
        log.warning("send_media failed | status=%d body=%s", resp.status_code, resp.text[:200])
    return resp
