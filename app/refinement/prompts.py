"""System prompt templates for the refinement chat and pattern extraction."""

import textwrap

_CONSTRAINT_BLOCK = textwrap.dedent("""\
    WhatsApp hard rules you MUST enforce — push back with a concrete fix suggestion when violated:
    - interactive: body ≤1024 chars; 1–3 buttons; each label ≤20 chars; labels CANNOT contain \
URLs (no http://, www., .com/.in, etc.); NO image_url.
    - media: requires image_url starting with https://; NO buttons; caption ≤1024 chars.
    - text: ≤4096 chars; NO buttons; NO image_url.
    - carousel: NO top-level buttons; NO top-level image_url.

    Examples of REJECTED button labels (with explanations):
      ❌ "Order now (product page link)" — 30 chars, exceeds 20-char limit. Suggest: "Order Now"
      ❌ "Visit numobel.in" — contains domain name, looks like a URL. Suggest: "View Products"
      ❌ "https://numobel.in/buy" — contains URL. Suggest: "Buy Now"
      ❌ "See more at www.x.com" — contains www., looks like URL. Suggest: "See More"
""")

_OUTPUT_CONTRACT = textwrap.dedent("""\
    You MUST reply with ONLY a valid JSON object — no markdown, no extra text:
    {"reply": "<your response to the admin>", "pair": <full updated pair object or null>}

    When "pair" is non-null it must be complete:
    {"question": "...", "answer": "...", "message_type": "text|interactive|media|carousel", \
"buttons": [...or null], "image_url": "...or null"}

    Include "pair" only when you have a changed version. Use null if no change.
""")


def build_system_prompt(
    original_pair: dict,
    admin_prefs: list[dict],
    style_examples: list[dict],
) -> str:
    pair_block = (
        f"  question: {original_pair.get('question', '')}\n"
        f"  answer: {original_pair.get('answer', '')}\n"
        f"  message_type: {original_pair.get('message_type', 'text')}\n"
        f"  buttons: {original_pair.get('buttons')}\n"
        f"  image_url: {original_pair.get('image_url')}"
    )

    prefs_block = ""
    if admin_prefs:
        lines = "\n".join(f"  - {p['text']}" for p in admin_prefs)
        prefs_block = f"\nAdmin style preferences (apply when relevant):\n{lines}\n"

    examples_block = ""
    if style_examples:
        parts = []
        for ex in style_examples[:3]:
            parts.append(
                f"  Q: {ex.get('question', '')}\n"
                f"  A: {ex.get('answer', '')}\n"
                f"  type={ex.get('message_type', 'text')} buttons={ex.get('buttons')}"
            )
        examples_block = "\nApproved style examples:\n" + "\n---\n".join(parts) + "\n"

    return textwrap.dedent(f"""\
        You are a WhatsApp chatbot training assistant for Numobel, an Indian home-materials company.
        Help the admin refine a Q&A pair for the product chatbot.
        Enforce WhatsApp constraints firmly — explain violations and suggest compliant alternatives.

        Current Q&A pair being refined:
        {pair_block}
        {prefs_block}{examples_block}
        {_CONSTRAINT_BLOCK}
        {_OUTPUT_CONTRACT}
        Be concise. Do not repeat the full pair unless you have a changed version.
    """).strip()


_PATTERN_EXTRACTION_PROMPT = textwrap.dedent("""\
    Review the following admin refinement conversation. Extract corrections that express a
    GENERALIZABLE rule — e.g., currency format, tone, terminology, label style.
    Do NOT extract corrections specific to the product being refined (name, price, etc.).

    Conversation:
    {history}

    Reply with ONLY a JSON array. Each item: {{"text": "<rule in one sentence>", "evidence": "<brief quote>"}}
    If no generalizable patterns exist, return: []
""")


def build_pattern_extraction_prompt(chat_history: list[dict]) -> str:
    lines = []
    for msg in chat_history:
        role = "Admin" if msg['role'] == 'user' else "Assistant"
        lines.append(f"{role}: {msg['content']}")
    return _PATTERN_EXTRACTION_PROMPT.format(history="\n".join(lines))
