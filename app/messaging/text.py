import logging
import os
import requests
from dotenv import load_dotenv

load_dotenv()

log = logging.getLogger('rag_chatbot')

_API_KEY    = os.getenv("WA2MATION_API_KEY")
_VENDOR_UID = os.getenv("WA2MATION_VENDOR_UID")
_URL        = f"https://wa2mation.com/api/{_VENDOR_UID}/contact/send-message"
_HEADERS    = {"Authorization": f"Bearer {_API_KEY}", "Content-Type": "application/json"}
_TIMEOUT    = 10  # seconds


def send_text(phone: str, message: str) -> requests.Response:
    resp = requests.post(
        _URL,
        json={"phone_number": phone, "message_body": message},
        headers=_HEADERS,
        timeout=_TIMEOUT,
    )
    if resp.status_code != 200:
        log.warning("send_text failed | status=%d body=%s", resp.status_code, resp.text[:200])
    return resp
