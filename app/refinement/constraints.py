"""WhatsApp constraint constants and pair validation."""

import re

INTERACTIVE_MAX_BUTTONS = 3
INTERACTIVE_MIN_BUTTONS = 1
BUTTON_LABEL_MAX_CHARS  = 20
BODY_MAX_CHARS          = 1024
HEADER_MAX_CHARS        = 60
FOOTER_MAX_CHARS        = 60
TEXT_MAX_CHARS          = 4096
CAPTION_MAX_CHARS       = 1024

VALID_TYPES = {"text", "interactive", "media", "carousel"}

# Detects http/https URLs, www., bit.ly, and bare TLD patterns like "numobel.in"
_URL_IN_LABEL_RE = re.compile(
    r'https?://|www\.|bit\.ly|\.[a-z]{2,4}(/|$)',
    re.IGNORECASE,
)


def _label_looks_like_url(label: str) -> bool:
    return bool(_URL_IN_LABEL_RE.search(label))


def validate_pair(pair: dict) -> list[str]:
    """
    Validate a Q&A pair against WhatsApp constraints.
    Returns human-readable violation messages (empty list = valid).
    """
    violations = []
    mt        = pair.get('message_type', 'text')
    buttons   = pair.get('buttons') or []
    image_url = pair.get('image_url') or ''
    answer    = pair.get('answer', '')

    if mt not in VALID_TYPES:
        violations.append(
            f"Unknown message_type '{mt}'. Must be one of: {', '.join(sorted(VALID_TYPES))}"
        )
        return violations

    if mt == 'interactive':
        if image_url:
            violations.append("Interactive messages cannot have an image_url.")
        if not buttons:
            violations.append("Interactive messages require at least 1 button.")
        elif len(buttons) > INTERACTIVE_MAX_BUTTONS:
            violations.append(
                f"Interactive messages support at most {INTERACTIVE_MAX_BUTTONS} buttons; "
                f"got {len(buttons)}."
            )
        for idx, label in enumerate(buttons, 1):
            if len(label) > BUTTON_LABEL_MAX_CHARS:
                violations.append(
                    f"Button {idx} label '{label}' is {len(label)} chars, max {BUTTON_LABEL_MAX_CHARS}."
                )
            if _label_looks_like_url(label):
                violations.append(
                    f"Button {idx} label '{label}' looks like a URL. "
                    "WhatsApp reply buttons cannot contain links."
                )
        if len(answer) > BODY_MAX_CHARS:
            violations.append(
                f"Interactive body is {len(answer)} chars, max {BODY_MAX_CHARS}."
            )

    elif mt == 'media':
        if not image_url:
            violations.append("Media messages require an image_url.")
        elif not image_url.startswith('https://'):
            violations.append(
                f"image_url must start with 'https://'; got '{image_url[:40]}'."
            )
        if buttons:
            violations.append("Media messages cannot have buttons.")
        if len(answer) > CAPTION_MAX_CHARS:
            violations.append(
                f"Media caption is {len(answer)} chars, max {CAPTION_MAX_CHARS}."
            )

    elif mt == 'text':
        if buttons:
            violations.append("Text messages cannot have buttons.")
        if image_url:
            violations.append("Text messages cannot have an image_url.")
        if len(answer) > TEXT_MAX_CHARS:
            violations.append(
                f"Text message is {len(answer)} chars, max {TEXT_MAX_CHARS}."
            )

    elif mt == 'carousel':
        if buttons:
            violations.append("Carousel messages cannot have top-level buttons.")
        if image_url:
            violations.append("Carousel messages cannot have a top-level image_url.")

    return violations
