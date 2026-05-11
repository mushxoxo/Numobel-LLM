# CLI approval loop for Q&A pairs
# [A]pprove → save to JSONL + ingest into ChromaDB via ingest_qna_pair()
# [S]uggest → multi-turn refinement REPL → type 'approve' or 'cancel' to exit
# [K]skip  [Q]uit
#
# Refinement model chosen at startup: [1] qwen2.5:14b (Ollama)  [2] claude-sonnet-4-6 via API

import json
import os
import sys
from pathlib import Path

from dotenv import load_dotenv

load_dotenv()

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import app.rag as rag
from app.config import PENDING_PATH as _PENDING_PATH
from app.log import get_logger
from app.refinement.engine import chat_turn, extract_patterns
from app.refinement.constraints import validate_pair
from app.refinement.storage import (
    append_admin_prefs,
    mark_approved,
    append_approved,
)
from training.utils import load_jsonl, choose_model

log = get_logger()


def load_unapproved() -> list[dict]:
    return [p for p in load_jsonl(_PENDING_PATH) if not p.get('approved')]


def _rewrite_pending(all_pairs: list[dict]) -> None:
    with open(_PENDING_PATH, 'w') as f:
        for pair in all_pairs:
            f.write(json.dumps(pair, ensure_ascii=False) + '\n')


def mark_refined_in_pending(pair: dict) -> None:
    """Write the refined answer back to pending.jsonl to prevent data loss on quit."""
    all_pairs = load_jsonl(_PENDING_PATH)
    for p in all_pairs:
        if p.get('question') == pair.get('question'):
            p['answer'] = pair['answer']
    _rewrite_pending(all_pairs)


def choose_admin_phone() -> str:
    print("\nEnter admin phone for prefs/state storage (matches WhatsApp admin phones in config.json):")
    phone = input("> ").strip()
    if not phone:
        try:
            username = os.getlogin()
        except Exception:
            username = 'unknown'
        phone = f"cli-{username}"
        print(f"Using fallback phone ID: {phone}")
    return phone


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


def _run_refine_repl(
    pair: dict,
    model: str,
    api_key: str | None,
    phone: str,
) -> dict:
    """
    Interactive multi-turn refinement loop for CLI.
    Type 'approve' or 'cancel' to exit.
    Returns the final pair (possibly unchanged if cancelled).
    """
    state: dict = {
        'original_pair':  dict(pair),
        'current_pair':   dict(pair),
        'chat_history':   [],
        'style_examples': [],
    }
    print("\nRefinement chat started. Type your suggestions, 'approve' to finish, or 'cancel' to abort.")

    while True:
        raw = input("> ").strip()
        if not raw:
            continue

        if raw.lower() == 'approve':
            violations = validate_pair(state['current_pair'])
            if violations:
                print("Cannot approve — fix these issues first:")
                for v in violations:
                    print(f"  • {v}")
                continue
            return state['current_pair']

        if raw.lower() == 'cancel':
            print("Refinement cancelled. Keeping original.")
            return pair

        reply, updated = chat_turn(state, raw, model, api_key, phone)
        if updated:
            state['current_pair'] = updated
        print(f"\nAssistant: {reply}\n")
        if updated:
            display_pair(state['current_pair'], 0, 0)

    return state['current_pair']


def run_approval_loop(
    pairs: list[dict],
    collection,
    model: str,
    api_key: str | None,
    phone: str,
) -> tuple[int, int]:
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
                violations = validate_pair(pair)
                if violations:
                    print("Cannot approve — fix these issues first:")
                    for v in violations:
                        print(f"  • {v}")
                    continue
                identifier = pair.get('pair_id') or pair.get('question', '')
                mark_approved(identifier)
                approved_pair = {**pair, 'approved': True}
                append_approved(approved_pair)
                rag.ingest_qna_pair(approved_pair, collection)
                print("Saved + ingested into ChromaDB")
                approved_count += 1
                break

            elif raw == 's':
                pair = _run_refine_repl(pair, model, api_key, phone)
                mark_refined_in_pending(pair)
                pairs[i] = pair
                display_pair(pair, i + 1, total)
                # Offer pattern extraction after refinement
                state: dict = {
                    'original_pair': pair,
                    'current_pair':  pair,
                    'chat_history':  [],
                }
                patterns = extract_patterns(state.get('chat_history', []), model, api_key)
                if patterns:
                    print("\nDetected style patterns:")
                    for idx, p in enumerate(patterns, 1):
                        print(f"  {idx}. {p['text']}")
                    save_choice = input("Save patterns? [y/n/numbers like 1,3]: ").strip().lower()
                    if save_choice == 'y':
                        append_admin_prefs(phone, patterns)
                        print(f"Saved {len(patterns)} pattern(s).")
                    elif save_choice not in ('n', ''):
                        import re
                        nums = [int(x) for x in re.findall(r'\d+', save_choice)
                                if 1 <= int(x) <= len(patterns)]
                        if nums:
                            chosen = [patterns[n - 1] for n in nums]
                            append_admin_prefs(phone, chosen)
                            print(f"Saved {len(chosen)} pattern(s).")

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
    collection = rag.get_collection()
    if collection.count() == 0:
        print("ChromaDB empty — ingesting products first...")
        rag.ingest_data(collection)

    pairs = load_unapproved()
    if not pairs:
        print("No unapproved pairs found. Run `python training/generate_qna.py` first.")
        return

    print(f"Found {len(pairs)} unapproved pairs.")
    model, api_key = choose_model("refinement")
    phone = choose_admin_phone()
    if model == 'qwen2.5:14b':
        print("Make sure `ollama serve` is running.\n")

    approved, skipped = run_approval_loop(pairs, collection, model, api_key, phone)

    remaining = load_unapproved()
    print(f"\nSession complete: {approved} approved, {skipped} skipped, {len(remaining)} remaining.")


if __name__ == '__main__':
    main()
