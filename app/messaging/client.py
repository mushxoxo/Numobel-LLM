"""
Shared wa2mation HTTP client.

Uses a lazy singleton so credentials are validated on first send (not at
import time), allowing training scripts to import messaging modules without
requiring wa2mation credentials.

Tests can reset the singleton between cases by setting _session = None.
"""

import os

import requests

_session: requests.Session | None = None
_TIMEOUT = 10


def _get_session() -> requests.Session:
    global _session
    if _session is None:
        key = os.getenv("WA2MATION_API_KEY")
        uid = os.getenv("WA2MATION_VENDOR_UID")
        if not key or not uid:
            raise RuntimeError(
                "WA2MATION_API_KEY and WA2MATION_VENDOR_UID must be set. "
                "Check your .env file."
            )
        _session = requests.Session()
        _session.headers.update({
            "Authorization": f"Bearer {key}",
            "Content-Type": "application/json",
        })
    return _session


def wa2mation_post(endpoint: str, payload: dict) -> requests.Response:
    """POST to a wa2mation contact endpoint and return the response."""
    vendor_uid = os.getenv("WA2MATION_VENDOR_UID")
    url = f"https://wa2mation.com/api/{vendor_uid}/contact/{endpoint}"
    return _get_session().post(url, json=payload, timeout=_TIMEOUT)
