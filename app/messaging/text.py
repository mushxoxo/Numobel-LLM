import logging
import os
import requests
from dotenv import load_dotenv

load_dotenv()

log = logging.getLogger('rag_chatbot')

_TIMEOUT = 10


def send_text(phone: str, message: str) -> requests.Response:
    api_key    = os.getenv("WA2MATION_API_KEY")
    vendor_uid = os.getenv("WA2MATION_VENDOR_UID")
    url     = f"https://wa2mation.com/api/{vendor_uid}/contact/send-message"
    headers = {"Authorization": f"Bearer {api_key}", "Content-Type": "application/json"}
    resp = requests.post(
        url,
        json={"phone_number": phone, "message_body": message},
        headers=headers,
        timeout=_TIMEOUT,
    )
    if resp.status_code != 200:
        log.warning("send_text failed | status=%d body=%s", resp.status_code, resp.text[:200])
    return resp
