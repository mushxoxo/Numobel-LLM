# Step 9 — CLI approval loop for Q&A pairs
# [A]pprove → save to JSONL + ingest into ChromaDB via ingest_qna_pair()
# [S]uggest → LLM refines pair (llama3.2) → writes back immediately → show updated → repeat
# [K]skip  [Q]uit
#
# Refinement always uses llama3.2 (Ollama) for fast interactive response.
# The Claude API model choice from generate_qna.py does not carry over here.

import json
import logging
import sys
from pathlib import Path

from dotenv import load_dotenv

load_dotenv()

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import ollama
import rag_chatbot as rag

log = logging.getLogger(__name__)

_PROJECT_ROOT  = Path(__file__).resolve().parent.parent
_PENDING_PATH  = _PROJECT_ROOT / 'training' / 'qna_pairs' / 'pending.jsonl'
_APPROVED_PATH = _PROJECT_ROOT / 'training' / 'qna_pairs' / 'approved.jsonl'


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


def load_unapproved() -> list[dict]:
    return [p for p in load_jsonl(_PENDING_PATH) if not p.get('approved')]


def _rewrite_pending(all_pairs: list[dict]) -> None:
    with open(_PENDING_PATH, 'w') as f:
        for pair in all_pairs:
            f.write(json.dumps(pair, ensure_ascii=False) + '\n')


def mark_approved_in_pending(pair: dict) -> None:
    all_pairs = load_jsonl(_PENDING_PATH)
    for p in all_pairs:
        if p.get('question') == pair.get('question'):
            p['approved'] = True
    _rewrite_pending(all_pairs)


def mark_refined_in_pending(pair: dict) -> None:
    """Write the refined answer back to pending.jsonl to prevent data loss on quit."""
    all_pairs = load_jsonl(_PENDING_PATH)
    for p in all_pairs:
        if p.get('question') == pair.get('question'):
            p['answer'] = pair['answer']
    _rewrite_pending(all_pairs)


def append_approved(pair: dict) -> None:
    _APPROVED_PATH.parent.mkdir(parents=True, exist_ok=True)
    with open(_APPROVED_PATH, 'a') as f:
        f.write(json.dumps(pair, ensure_ascii=False) + '\n')


_REFINE_PROMPT_TEMPLATE = (
    "Refine this WhatsApp chatbot Q&A pair based on the suggestion.\n\n"
    "Current pair:\n"
    "  question: {question}\n"
    "  answer: {answer}\n"
    "  message_type: {message_type}\n"
    "  buttons: {buttons}\n"
    "  image_url: {image_url}\n\n"
    "Suggestion: {suggestion}\n\n"
    "WhatsApp message type rules:\n"
    "  - 'interactive': has buttons (2-3 short labels), no image\n"
    "  - 'media': has image_url, no buttons\n"
    "  - 'text': plain text only, no buttons, no image\n"
    "  - 'carousel': multiple product cards, no buttons, no image\n\n"
    "IMPORTANT: Only change fields the suggestion explicitly asks to change. "
    "Keep unchanged fields exactly as they are.\n\n"
    "Reply with a JSON object: "
    "{{\"answer\": \"...\", \"message_type\": \"text|interactive|media|carousel\", "
    "\"buttons\": [...or null], \"image_url\": \"...or null\"}}"
)


def _normalize_refined(updated: dict, original: dict) -> dict:
    mt        = updated.get('message_type', original.get('message_type', 'text'))
    buttons   = updated.get('buttons',   original.get('buttons'))
    image_url = updated.get('image_url', original.get('image_url'))

    if mt == 'text' and buttons:
        mt = 'interactive'
    if mt == 'text' and image_url:
        mt = 'media'
    if mt == 'interactive':
        image_url = None
    if mt in ('media', 'text', 'carousel'):
        buttons = None

    return {
        **original,
        'answer':       updated.get('answer', original['answer']),
        'message_type': mt,
        'buttons':      buttons,
        'image_url':    image_url,
    }


def refine_with_llm(pair: dict, suggestion: str) -> dict:
    import os
    prompt = _REFINE_PROMPT_TEMPLATE.format(
        question=pair['question'],
        answer=pair['answer'],
        message_type=pair.get('message_type', 'text'),
        buttons=pair.get('buttons'),
        image_url=pair.get('image_url'),
        suggestion=suggestion,
    )

    api_key = os.getenv('ANTHROPIC_API_KEY')
    if api_key:
        import anthropic
        client = anthropic.Anthropic(api_key=api_key)
        message = client.messages.create(
            model='claude-sonnet-4-6',
            max_tokens=512,
            messages=[{'role': 'user', 'content': prompt}],
        )
        raw = message.content[0].text.strip()
    else:
        response = ollama.chat(
            model='llama3.2',
            messages=[{'role': 'user', 'content': prompt}],
            format='json',
        )
        raw = response['message']['content'].strip()

    try:
        updated = json.loads(raw)
        return _normalize_refined(updated, pair)
    except (json.JSONDecodeError, AttributeError):
        return {**pair, 'answer': raw}


def display_pair(pair: dict, idx: int, total: int) -> None:
    buttons = pair.get('buttons')
    print(f"\n[{idx}/{total}]")
    print(f"Product : {pair.get('product', '')}")
    print(f"Type    : {pair.get('message_type', 'text')}")
    print(f"Q       : {pair.get('question', '')}")
    print(f"A       : {pair.get('answer', '')}")
    if buttons:
        print(f"Buttons : {buttons}")
    if pair.get('message_type') == 'media':
        print(f"Image   : {pair.get('image_url') or '(none)'}")


def run_approval_loop(pairs: list[dict], collection) -> tuple[int, int]:
    total = len(pairs)
    approved_count = 0
    skipped_count  = 0

    i = 0
    while i < len(pairs):
        pair = pairs[i]
        display_pair(pair, i + 1, total)

        while True:
            raw = input("\n[A]pprove  [S]uggest  [K]skip  [Q]uit  > ").strip().lower()

            if raw == 'a':
                mark_approved_in_pending(pair)
                approved_pair = {**pair, 'approved': True}
                append_approved(approved_pair)
                rag.ingest_qna_pair(approved_pair, collection)
                print("✅ Saved + ingested into ChromaDB")
                approved_count += 1
                break

            elif raw == 's':
                suggestion = input("Suggestion: ").strip()
                if not suggestion:
                    print("No suggestion entered.")
                    continue
                try:
                    pair = refine_with_llm(pair, suggestion)
                    mark_refined_in_pending(pair)
                    pairs[i] = pair
                    display_pair(pair, i + 1, total)
                except Exception as e:
                    print(f"Refinement failed: {e}. Showing original.")

            elif raw == 'k':
                skipped_count += 1
                break

            elif raw == 'q':
                return approved_count, skipped_count

            else:
                print("Invalid choice. Use A, S, K, or Q.")

        i += 1

    return approved_count, skipped_count


def main() -> None:
    logging.basicConfig(level=logging.WARNING, format='%(levelname)s: %(message)s')

    collection = rag.get_collection()
    if collection.count() == 0:
        print("ChromaDB empty — ingesting products first...")
        rag.ingest_data(collection)

    pairs = load_unapproved()
    if not pairs:
        print("No unapproved pairs found. Run `python training/generate_qna.py` first.")
        return

    print(f"Found {len(pairs)} unapproved pairs.")
    print("Refinement uses llama3.2 (Ollama) — make sure `ollama serve` is running.\n")

    approved, skipped = run_approval_loop(pairs, collection)

    remaining = load_unapproved()
    print(f"\nSession complete: {approved} approved, {skipped} skipped, {len(remaining)} remaining.")


if __name__ == '__main__':
    main()
