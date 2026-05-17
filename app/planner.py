"""Deterministic response format planner.

Single responsibility: given an intent and a data_shape describing what's
available (button count, images, product count), return the optimal WhatsApp
message type. No LLM calls, no SQLite, no HTTP.
"""
from app.intent import IntentEnum
from app.config import WA_INTERACTIVE_MAX_BUTTONS
from app.log import get_logger

__all__ = ["plan_response"]

log = get_logger()


def plan_response(intent: IntentEnum, data_shape: dict) -> str:
    """Return the optimal WhatsApp message_type for the given intent and data shape.

    data_shape keys:
      button_count (int): number of available buttons/options
      image_available (bool): whether a product image is available
      product_count (int): number of matching products
    """
    button_count    = data_shape.get("button_count", 0)
    image_available = data_shape.get("image_available", False)
    product_count   = data_shape.get("product_count", 0)

    match intent:
        case IntentEnum.GREETING | IntentEnum.CHITCHAT | IntentEnum.OUT_OF_SCOPE:
            planned = "text"
        case IntentEnum.BRAND_DISCOVERY | IntentEnum.BRAND_DEEP_DIVE:
            planned = "interactive" if button_count <= WA_INTERACTIVE_MAX_BUTTONS else "text"
        case IntentEnum.PRODUCT_LINE_QUERY:
            if image_available and product_count >= 2:
                planned = "carousel"
            elif button_count <= WA_INTERACTIVE_MAX_BUTTONS:
                planned = "interactive"
            else:
                planned = "text"
        case IntentEnum.SPECIFIC_PRODUCT:
            planned = "media" if image_available else "text"
        case _:
            planned = "text"

    log.debug("PLANNER | intent=%s data_shape=%s -> %s", intent.value, data_shape, planned)
    return planned
