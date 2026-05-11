import requests

from app.log import get_logger
from app.messaging.client import wa2mation_post

log = get_logger()


def send_text(phone: str, message: str) -> requests.Response:
    resp = wa2mation_post(
        "send-message",
        {"phone_number": phone, "message_body": message},
    )
    if resp.status_code != 200:
        log.warning("send_text failed | status=%d body=%s", resp.status_code, resp.text[:200])
    return resp
