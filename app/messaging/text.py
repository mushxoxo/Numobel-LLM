import os
import requests
from dotenv import load_dotenv

load_dotenv()

_API_KEY    = os.getenv("WA2MATION_API_KEY")
_VENDOR_UID = os.getenv("WA2MATION_VENDOR_UID")
_URL        = f"https://wa2mation.com/api/{_VENDOR_UID}/contact/send-message"
_HEADERS    = {"Authorization": f"Bearer {_API_KEY}", "Content-Type": "application/json"}


def send_text(phone: str, message: str) -> requests.Response:
    return requests.post(
        _URL,
        json={"phone_number": phone, "message_body": message},
        headers=_HEADERS,
    )
