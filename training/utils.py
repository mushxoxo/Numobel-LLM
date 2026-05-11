"""Shared utilities for the Numobel training pipeline scripts."""

import json
import os
from pathlib import Path

from app.config import PENDING_PATH, APPROVED_PATH
from app.log import get_logger

log = get_logger()


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


def choose_model(purpose: str = "processing") -> tuple[str, str | None]:
    """Interactive prompt to choose between Ollama and Claude API."""
    print(f"\nWhich model to use for {purpose}?")
    print("  [1] qwen2.5:14b       (Ollama — local, free)")
    print("  [2] claude-sonnet-4-6 (Claude API — best quality, uses API key)")
    choice = input("Choice [1/2]: ").strip()

    if choice == '2':
        api_key = os.getenv('ANTHROPIC_API_KEY')
        if not api_key:
            api_key = input("Enter ANTHROPIC_API_KEY: ").strip()
        if not api_key:
            print("No API key provided. Falling back to qwen2.5:14b.")
            return 'qwen2.5:14b', None
        return 'claude-sonnet-4-6', api_key

    return 'qwen2.5:14b', None
