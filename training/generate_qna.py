# Step 8 — Auto-generates Q&A pairs from data/clean_products.json
# LLM choice at startup: [1] llama3.2 via Ollama  [2] claude-sonnet-4-6 via API
# Output: training/qna_pairs/pending.jsonl

import json
import logging
import os
import re
import sys
import time
from pathlib import Path

from dotenv import load_dotenv

load_dotenv()

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import ollama

log = logging.getLogger(__name__)

_PROJECT_ROOT  = Path(__file__).resolve().parent.parent
_DATA_FILE     = _PROJECT_ROOT / 'data' / 'clean_products.json'
_PENDING_PATH  = _PROJECT_ROOT / 'training' / 'qna_pairs' / 'pending.jsonl'
_APPROVED_PATH = _PROJECT_ROOT / 'training' / 'qna_pairs' / 'approved.jsonl'

_VALID_TYPES = {'text', 'interactive', 'media', 'carousel'}

_SYSTEM_PROMPT = (
    "You are a training data generator for Numobel's WhatsApp product chatbot (India). "
    "Given a product, generate 3 to 5 Q&A pairs a real Indian customer might ask on WhatsApp.\n\n"
    "RULES:\n"
    "- Vary message_type across pairs: \"text\", \"interactive\", \"media\", \"carousel\"\n"
    "  (use \"carousel\" only if the product has multiple images or size/color variants)\n"
    "- \"interactive\": include 2-3 short button labels as a JSON array in \"buttons\"\n"
    "- \"media\": set image_url to the product's first image URL (or null if none)\n"
    "- \"text\" and \"carousel\": set buttons to null and image_url to null\n"
    "- Answers must be concise (under 100 words), use ₹ for prices\n"
    "- Never invent specifications not present in the product data\n\n"
    "OUTPUT: Reply with ONLY a valid JSON array — no markdown fences, no commentary."
)


def load_products() -> list[dict]:
    if not _DATA_FILE.exists():
        print(f"ERROR: {_DATA_FILE} not found. Run `python clean_products.py` first.")
        sys.exit(1)
    with open(_DATA_FILE) as f:
        return json.load(f)


def load_jsonl(path: Path) -> list[dict]:
    if not path.exists():
        return []
    pairs = []
    with open(path) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                pairs.append(json.loads(line))
            except json.JSONDecodeError:
                log.warning("Skipping malformed JSONL line in %s: %.80s", path.name, line)
    return pairs


def already_done_products() -> set[str]:
    return {p.get('product', '') for p in load_jsonl(_PENDING_PATH) if p.get('product')}


def select_few_shot_examples(approved: list[dict], n: int = 3) -> list[dict]:
    if not approved:
        return []
    seen_types: set[str] = set()
    examples: list[dict] = []
    for mt in ('text', 'interactive', 'media', 'carousel'):
        if len(examples) >= n:
            break
        for pair in approved:
            if pair.get('message_type') == mt and mt not in seen_types:
                examples.append(pair)
                seen_types.add(mt)
                break
    for pair in approved:
        if len(examples) >= n:
            break
        if pair not in examples:
            examples.append(pair)
    return examples[:n]


def build_generation_prompt(product: dict, few_shot: list[dict]) -> str:
    images = (product.get('media') or {}).get('images', [])
    attrs  = product.get('attributes') or {}
    price_obj = product.get('price') or {}
    price = price_obj.get('discounted') or price_obj.get('original')
    product_line = product.get('product_line') or ''

    product_block = (
        f"Product: {product['name']}\n"
        f"Brand: {product.get('brand', '')}\n"
        + (f"Product line: {product_line}\n" if product_line else "")
        + f"Description: {product.get('description', '')}\n"
        f"Price: ₹{price}\n"
        f"Colors: {', '.join(str(c) for c in attrs.get('colors', [])) or 'N/A'}\n"
        f"Sizes: {', '.join(str(s) for s in attrs.get('size', [])) or 'N/A'}\n"
        f"Specifications: {attrs.get('specifications', '') or 'N/A'}\n"
        f"First image URL: {images[0] if images else 'none'}\n"
        f"Total images: {len(images)}\n"
        f"Product link: {(product.get('metadata') or {}).get('product_link', '')}"
    )

    few_shot_block = ""
    if few_shot:
        lines = ["EXAMPLES (follow this style and JSON format):"]
        for ex in few_shot:
            lines.append(json.dumps(ex, ensure_ascii=False))
        few_shot_block = "\n".join(lines) + "\n\n"

    return f"{few_shot_block}Generate Q&A pairs for this product:\n{product_block}"


def strip_fences(raw: str) -> str:
    raw = raw.strip()
    raw = re.sub(r'^```(?:json)?\s*', '', raw, flags=re.IGNORECASE)
    raw = re.sub(r'\s*```$', '', raw)
    return raw.strip()


def parse_pairs(raw: str, product: dict) -> list[dict]:
    try:
        cleaned = strip_fences(raw)
        parsed = json.loads(cleaned)
    except (json.JSONDecodeError, ValueError):
        log.warning(
            "parse_pairs: JSON decode failed for product %s. Raw: %.120s",
            product.get('name'), raw,
        )
        return []

    if isinstance(parsed, dict):
        parsed = [parsed]
    if not isinstance(parsed, list):
        log.warning(
            "parse_pairs: unexpected JSON type %s for product %s",
            type(parsed).__name__, product.get('name'),
        )
        return []

    result = []
    for item in parsed:
        if not isinstance(item, dict):
            continue
        if not item.get('question') or not item.get('answer'):
            continue
        mt = item.get('message_type', 'text')
        if mt not in _VALID_TYPES:
            log.warning("parse_pairs: dropping pair with invalid message_type %r", mt)
            continue
        buttons = item.get('buttons')
        if isinstance(buttons, list) and len(buttons) > 3:
            buttons = buttons[:3]
        result.append({
            'question':     item['question'],
            'answer':       item['answer'],
            'message_type': mt,
            'buttons':      buttons,
            'image_url':    item.get('image_url'),
            'product':      product['name'],
            'approved':     False,
        })
    return result


def call_llm(prompt: str, model: str, api_key: str | None) -> str:
    if model == 'llama3.2':
        response = ollama.chat(
            model='llama3.2',
            messages=[
                {'role': 'system', 'content': _SYSTEM_PROMPT},
                {'role': 'user',   'content': prompt},
            ],
        )
        return response['message']['content']
    else:
        import anthropic
        client = anthropic.Anthropic(api_key=api_key, max_retries=3)
        message = client.messages.create(
            model='claude-sonnet-4-6',
            max_tokens=1024,
            system=_SYSTEM_PROMPT,
            messages=[{'role': 'user', 'content': prompt}],
        )
        return message.content[0].text


def append_to_pending(pairs: list[dict]) -> None:
    _PENDING_PATH.parent.mkdir(parents=True, exist_ok=True)
    with open(_PENDING_PATH, 'a') as f:
        for pair in pairs:
            f.write(json.dumps(pair, ensure_ascii=False) + '\n')


def choose_model() -> tuple[str, str | None]:
    print("\nWhich model to use for generation?")
    print("  [1] llama3.2       (Ollama — local, free, slower)")
    print("  [2] claude-sonnet-4-6 (Claude API — better quality, uses API key)")
    choice = input("Choice [1/2]: ").strip()

    if choice == '2':
        api_key = os.getenv('ANTHROPIC_API_KEY')
        if not api_key:
            api_key = input("Enter ANTHROPIC_API_KEY: ").strip()
        if not api_key:
            print("No API key provided. Falling back to llama3.2.")
            return 'llama3.2', None
        return 'claude-sonnet-4-6', api_key

    return 'llama3.2', None


def main() -> None:
    logging.basicConfig(level=logging.WARNING, format='%(levelname)s: %(message)s')
    model, api_key = choose_model()

    products = load_products()
    done = already_done_products()
    remaining = [p for p in products if p['name'] not in done]
    print(f"\n({len(done)} already done, generating {len(remaining)} more)\n")

    if not remaining:
        print("Nothing to generate. Use approve_qna_cli.py to review pending pairs.")
        return

    approved = load_jsonl(_APPROVED_PATH)
    few_shot = select_few_shot_examples(approved)
    if few_shot:
        print(f"Using {len(few_shot)} approved pair(s) as few-shot examples.\n")

    failed = 0
    for i, product in enumerate(remaining, 1):
        print(f"[{i}/{len(remaining)}] {product['name']}...", end=' ', flush=True)
        prompt = build_generation_prompt(product, few_shot)
        try:
            raw = call_llm(prompt, model, api_key)
            pairs = parse_pairs(raw, product)
            if pairs:
                append_to_pending(pairs)
                print(f"✓ {len(pairs)} pairs")
            else:
                print("✗ no valid pairs parsed")
                failed += 1
        except Exception as e:
            print(f"✗ error: {e}")
            log.exception("Generation failed for %s", product['name'])
            failed += 1

        if model != 'llama3.2':
            time.sleep(1.2)

    total_ok = len(remaining) - failed
    print(f"\nDone! {total_ok} products generated, {failed} failed.")
    print("Run `python training/approve_qna_cli.py` to review pending pairs.")


if __name__ == '__main__':
    main()
