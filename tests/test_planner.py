import pytest
from app.intent import IntentEnum

plan_response = pytest.importorskip("app.planner").plan_response


def test_greeting_returns_text():
    assert plan_response(IntentEnum.GREETING, {"button_count": 0, "image_available": False, "product_count": 0}) == "text"

def test_chitchat_returns_text():
    assert plan_response(IntentEnum.CHITCHAT, {"button_count": 0, "image_available": False, "product_count": 0}) == "text"

def test_out_of_scope_returns_text():
    assert plan_response(IntentEnum.OUT_OF_SCOPE, {"button_count": 0, "image_available": False, "product_count": 0}) == "text"

def test_brand_discovery_under_threshold_interactive():
    assert plan_response(IntentEnum.BRAND_DISCOVERY, {"button_count": 3, "image_available": False, "product_count": 0}) == "interactive"

def test_brand_discovery_over_threshold_text():
    assert plan_response(IntentEnum.BRAND_DISCOVERY, {"button_count": 5, "image_available": False, "product_count": 0}) == "text"

def test_brand_deep_dive_buttons_under_limit_interactive():
    assert plan_response(IntentEnum.BRAND_DEEP_DIVE, {"button_count": 2, "image_available": False, "product_count": 0}) == "interactive"

def test_brand_deep_dive_overflow_text():
    assert plan_response(IntentEnum.BRAND_DEEP_DIVE, {"button_count": 7, "image_available": False, "product_count": 0}) == "text"

def test_product_line_with_images_carousel():
    assert plan_response(IntentEnum.PRODUCT_LINE_QUERY, {"button_count": 0, "image_available": True, "product_count": 4}) == "carousel"

def test_product_line_no_images_interactive_when_few_buttons():
    assert plan_response(IntentEnum.PRODUCT_LINE_QUERY, {"button_count": 2, "image_available": False, "product_count": 2}) == "interactive"

def test_product_line_no_images_text_when_many_buttons():
    assert plan_response(IntentEnum.PRODUCT_LINE_QUERY, {"button_count": 5, "image_available": False, "product_count": 5}) == "text"

def test_specific_product_with_image_media():
    assert plan_response(IntentEnum.SPECIFIC_PRODUCT, {"button_count": 0, "image_available": True, "product_count": 1}) == "media"

def test_specific_product_no_image_text():
    assert plan_response(IntentEnum.SPECIFIC_PRODUCT, {"button_count": 0, "image_available": False, "product_count": 1}) == "text"

def test_general_qna_returns_text():
    assert plan_response(IntentEnum.GENERAL_QNA, {"button_count": 0, "image_available": False, "product_count": 0}) == "text"
