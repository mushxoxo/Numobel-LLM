import pytest
from app.refinement.constraints import validate_pair


def _pair(**kwargs):
    base = {'question': 'Q?', 'answer': 'A.', 'message_type': 'text'}
    return {**base, **kwargs}


# ─── Text ─────────────────────────────────────────────────────────────────────

def test_valid_text():
    assert validate_pair(_pair()) == []


def test_text_no_buttons():
    v = validate_pair(_pair(buttons=['OK']))
    assert any('button' in x.lower() for x in v)


def test_text_no_image():
    v = validate_pair(_pair(image_url='https://example.com/img.jpg'))
    assert any('image_url' in x.lower() for x in v)


def test_text_too_long():
    v = validate_pair(_pair(answer='x' * 4097))
    assert any('4096' in x for x in v)


# ─── Interactive ──────────────────────────────────────────────────────────────

def test_valid_interactive():
    v = validate_pair(_pair(message_type='interactive', buttons=['Yes', 'No']))
    assert v == []


def test_interactive_no_image():
    v = validate_pair(_pair(
        message_type='interactive',
        buttons=['Yes'],
        image_url='https://example.com/img.jpg',
    ))
    assert any('image_url' in x.lower() for x in v)


def test_interactive_too_many_buttons():
    v = validate_pair(_pair(message_type='interactive', buttons=['A', 'B', 'C', 'D']))
    assert any('3 buttons' in x for x in v)


def test_interactive_no_buttons():
    v = validate_pair(_pair(message_type='interactive', buttons=[]))
    assert any('button' in x.lower() for x in v)


def test_button_label_too_long():
    long_label = 'Order now (product page link)'  # 29 chars — exceeds 20-char limit
    assert len(long_label) > 20
    v = validate_pair(_pair(message_type='interactive', buttons=[long_label]))
    assert any('chars' in x and 'max 20' in x for x in v)


def test_button_label_ok():
    v = validate_pair(_pair(message_type='interactive', buttons=['Contact Us']))
    assert v == []


def test_button_label_url_https():
    v = validate_pair(_pair(message_type='interactive', buttons=['https://numobel.in']))
    assert any('URL' in x for x in v)


def test_button_label_url_www():
    v = validate_pair(_pair(message_type='interactive', buttons=['Visit www.numobel.in']))
    assert any('URL' in x for x in v)


def test_button_label_bare_domain():
    v = validate_pair(_pair(message_type='interactive', buttons=['Order at numobel.in']))
    assert any('URL' in x for x in v)


def test_button_label_bitly():
    v = validate_pair(_pair(message_type='interactive', buttons=['bit.ly/xyz']))
    assert any('URL' in x for x in v)


def test_button_label_safe_word_with_dot():
    # "View" is not a URL
    v = validate_pair(_pair(message_type='interactive', buttons=['Order Now']))
    assert v == []


def test_interactive_body_too_long():
    v = validate_pair(_pair(message_type='interactive', buttons=['OK'], answer='x' * 1025))
    assert any('1024' in x for x in v)


# ─── Media ────────────────────────────────────────────────────────────────────

def test_valid_media():
    v = validate_pair(_pair(message_type='media', image_url='https://example.com/img.jpg'))
    assert v == []


def test_media_no_image():
    v = validate_pair(_pair(message_type='media'))
    assert any('image_url' in x.lower() for x in v)


def test_media_http_not_https():
    v = validate_pair(_pair(message_type='media', image_url='http://example.com/img.jpg'))
    assert any('https' in x.lower() for x in v)


def test_media_no_buttons():
    v = validate_pair(_pair(
        message_type='media',
        image_url='https://example.com/img.jpg',
        buttons=['Buy'],
    ))
    assert any('button' in x.lower() for x in v)


def test_media_caption_too_long():
    v = validate_pair(_pair(
        message_type='media',
        image_url='https://example.com/img.jpg',
        answer='x' * 1025,
    ))
    assert any('1024' in x for x in v)


# ─── Carousel ────────────────────────────────────────────────────────────────

def test_valid_carousel():
    v = validate_pair(_pair(message_type='carousel'))
    assert v == []


def test_carousel_no_buttons():
    v = validate_pair(_pair(message_type='carousel', buttons=['A', 'B']))
    assert any('button' in x.lower() for x in v)


def test_carousel_no_image():
    v = validate_pair(_pair(message_type='carousel', image_url='https://x.com/img.jpg'))
    assert any('image_url' in x.lower() for x in v)


# ─── Unknown type ─────────────────────────────────────────────────────────────

def test_unknown_type():
    v = validate_pair(_pair(message_type='video'))
    assert any('unknown' in x.lower() for x in v)
