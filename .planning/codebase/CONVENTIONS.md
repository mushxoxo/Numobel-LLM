# Coding Conventions
_Generated: 2026-05-16_

## Naming Patterns

**Files:**
- `snake_case.py` throughout — e.g., `rag_chatbot.py`, `clean_products.py`, `approve_qna_cli.py`
- Test files prefixed with `test_` matching the module they cover — e.g., `test_router.py` covers `app/router.py`
- Sub-package modules named for their single responsibility — e.g., `carousel.py`, `interactive.py`, `text.py`, `media.py`

**Functions:**
- Public functions: `snake_case` — e.g., `send_carousel()`, `load_history()`, `validate_pair()`, `dispatch()`
- Private helpers: leading underscore `_snake_case` — e.g., `_get_session()`, `_keyword_fallback()`, `_images_from_hits()`, `_atomic_write()`, `_normalize_refined()`
- Boolean predicates: verb form — e.g., `is_admin()`, `is_locked()`, `needs_admin_handling()`, `_is_duplicate()`

**Variables and constants:**
- Constants at module level: `UPPER_SNAKE_CASE` — e.g., `CAROUSEL_TEMPLATE`, `CAROUSEL_MAX_CARDS`, `DEDUP_TTL`, `SESSION_TIMEOUT`, `MAX_PREFS`
- Private module-level constants: leading underscore — e.g., `_CAROUSEL_TEMPLATE`, `_DEDUP_TTL`, `_SESSIONS_DIR`, `_PHONE_RE`, `_session`
- Local variables: `snake_case` — e.g., `message_type`, `user_message`, `body_vars`
- Alignment padding used for multi-variable assignments when visually grouping related items (e.g., `app/config.py`, `app/webhook.py`)

**Classes:**
- `PascalCase` — e.g., `AdminState`, `_JsonFormatter`
- Enums extend `str, Enum` for JSON-serialisable string values — e.g., `AdminState(str, Enum)` in `app/admin.py`

**Test helpers:**
- Private builder functions prefixed with `_` — e.g., `_pair()`, `_hit()`, `_result()`, `_make_pair()`, `_make_product()`, `_make_state()`
- Constants in test files: `UPPER_SNAKE_CASE` — e.g., `PHONE = "919999999999"`, `SAMPLE_PAIR`, `INCOMING`, `RAG_RESULT`

## Code Style

**Module docstrings:**
Every module has a top-level triple-quoted docstring explaining its purpose and key constraints. Examples:
- `app/config.py`: "Central configuration... Does NOT validate credentials — those are checked lazily..."
- `app/messaging/client.py`: "Shared wa2mation HTTP client. Uses a lazy singleton..."
- `app/refinement/storage.py`: "Per-admin refinement state and preferences I/O. All writes are atomic."

**Function docstrings:**
Single-line or short multi-line docstrings for all public functions. Leading with verb phrase. Examples:
- `load_history(phone)`: "Load conversation history for phone. Returns [] if expired or missing."
- `dispatch(phone, result, hits)`: "Route a generate_answer() result to the correct wa2mation send function.\n\nSingle responsibility — no LLM calls, no business logic."
- `validate_pair(pair)`: "Validate a Q&A pair against WhatsApp constraints.\nReturns human-readable violation messages (empty list = valid)."

Private helpers often have no docstring when the name is self-explanatory.

**Section separators:**
Long modules use horizontal rule comments to delineate logical sections:
```python
# ─── ChromaDB ─────────────────────────────────────────────────────────────────
# ─── Text utilities ───────────────────────────────────────────────────────────
```
This pattern appears in `app/rag.py`, `app/refinement/storage.py`, `app/refinement/constraints.py`, and all test files.

**Inline comments:**
Used to explain non-obvious decisions or gotchas, not to restate the code. Examples from `app/webhook.py`:
```python
# In-memory deduplication: {message_id: timestamp}
# Protects against wa2mation retry duplicates within a 60-second window.
```
```python
# Always returns HTTP 200 (even on errors) — returning 500 causes wa2mation to retry
```

## Import Organization

**Order within files:**
1. Standard library (`import os`, `import json`, `import re`, `from pathlib import Path`)
2. Third-party (`import chromadb`, `import ollama`, `from flask import Flask`)
3. Internal app (`from app.config import ...`, `from app.log import get_logger`)

Blank line between each group. No blank line between stdlib and third-party is seen in some files — the pattern is consistent grouping, not strict PEP 8 blank-line enforcement.

**Internal import style:**
- Prefer explicit `from app.X import name` for functions used frequently in a module
- Use `import app.rag as rag` (module alias) when calling multiple functions on the module — e.g., `rag.retrieve()`, `rag.generate_answer()`, `rag.ingest_qna_pair()`
- Never use wildcard imports

**Logger setup — always at module level:**
```python
from app.log import get_logger
log = get_logger()
```
This line appears in every `app/` module that emits logs.

**Test imports:**
- `import pytest` and `from unittest.mock import patch, MagicMock` at top
- Subject module imported inside the test function body when the module has side effects at import time (e.g., `from app.router import dispatch` inside `with patch(...)` block)
- Subject module imported at top-of-file when it has no startup side effects (e.g., `import app.refinement.engine as engine_module`)

## Configuration Management

**Single source of truth: `app/config.py`**
All constants and filesystem paths live in `app/config.py`. `load_dotenv()` is called once at import. No module may hardcode paths, model names, or limits that already appear in `app/config.py`.

**Accessing config:**
```python
from app.config import LLM_MODEL, EMBED_MODEL, MEMORY_LIMIT, TOP_K
```
Named imports, not `from app.config import *`.

**WhatsApp constraint constants — `app/refinement/constraints.py`:**
All validated WhatsApp limits (button count, label length, body length) live here. Modules that need to enforce these import from constraints, not from config. Example: `app/admin.py` does `from app.refinement.constraints import BODY_MAX_CHARS`.

**Environment variables:**
Accessed via `os.getenv()` with explicit fallback or `None`. Credential presence is checked lazily at first use — not at import time. Pattern in `app/messaging/client.py`:
```python
key = os.getenv("WA2MATION_API_KEY")
uid = os.getenv("WA2MATION_VENDOR_UID")
if not key or not uid:
    raise RuntimeError("...")
```

**File-level private constants:**
Module-private path constants use leading underscore and are defined near the top of the module, below imports:
```python
_SESSIONS_DIR    = os.path.join(...)
_SESSION_TIMEOUT = timedelta(minutes=5)
_PHONE_RE        = re.compile(r'...')
```

## Error Handling

**Webhook always returns HTTP 200:**
`app/webhook.py` wraps the entire handler in `try/except Exception` and returns `{"status": "error"}` with HTTP 200 on unhandled failures. Returning 5xx would trigger wa2mation retry storms.

**Graceful degradation pattern:**
Functions that can fail silently return a fallback value rather than raising. Examples:
- `load_history()` returns `[]` on missing/corrupt/expired files
- `load_refine_state()` returns `{}` on missing/corrupt files
- `generate_answer()` calls `_keyword_fallback()` when LLM returns invalid JSON
- `dispatch()` falls back to `send_text()` when required assets (buttons/images) are unavailable

**Specific exception types:**
`except (json.JSONDecodeError, KeyError, ValueError)` is preferred over bare `except Exception` in storage functions. The webhook route uses bare `except Exception` deliberately as a last-resort catch-all.

**Atomic file writes:**
`app/refinement/storage.py` uses write-to-tmp-then-`os.replace()` for all state files:
```python
def _atomic_write(path: Path, data: dict) -> None:
    tmp = path.with_suffix('.tmp')
    tmp.write_text(json.dumps(data, ...), encoding='utf-8')
    os.replace(tmp, path)
```

**File locking:**
`app/history.py` uses `fcntl.flock()` for exclusive locks when writing session files — Linux-only, intentional for cloud VM deployment.

## Logging Conventions

**Logger acquisition — module level:**
```python
from app.log import get_logger
log = get_logger()
```

**Log level conventions (enforced throughout):**
- `log.debug(...)` — internals: token counts, query rewrites, dedup cache hits, session loads
- `log.info(...)` — normal request flow, startup events, state transitions
- `log.warning(...)` — fallbacks triggered, API failures, constraint violations, corrupt files
- `log.error(...)` — unexpected failures (rare; most use `log.exception`)
- `log.critical(...)` — startup failures that degrade service (e.g., ingest failed)
- `log.exception(...)` — for caught exceptions where full traceback is needed (used in webhook catch-all)

**Log message format — `COMPONENT | key=value` style:**
```python
log.debug("DISPATCH | type=%s phone=%s", message_type, phone)
log.info("STARTUP | model=%s embed=%s chroma=%s docs=%d", ...)
log.info("req=%s | phone=%s msg=%.60r", req, phone, user_message)
log.warning("send_carousel failed | status=%d body=%s", resp.status_code, resp.text[:200])
log.debug("HISTORY | expired session deleted for %s", phone)
```
Always use `%`-style formatting (not f-strings) to defer string interpolation until the log record is actually emitted.

**Request correlation:**
Webhook generates a short hex ID per request: `req = uuid.uuid4().hex[:8]`. All log lines within that request include `req=%s` as the first field.

## Messaging Module Pattern

Each send function in `app/messaging/` follows a consistent shape:
1. Apply constraints silently (truncate buttons to 3, cap label to 20 chars) with a `log.warning`
2. Build the `payload` dict
3. Conditionally add optional fields only if non-empty (`if header: payload["header_type"] = ...`)
4. Call `wa2mation_post(endpoint, payload)`
5. Log a warning if `resp.status_code != 200`
6. Return the raw `requests.Response` — callers are not expected to inspect it

## Type Hints

Used consistently on all public function signatures. Parameter types and return types both annotated. Examples:
```python
def dispatch(phone: str, result: dict, hits: list[dict] = None) -> None:
def validate_pair(pair: dict) -> list[str]:
def load_history(phone: str) -> list[dict]:
def send_carousel(phone: str, template_name: str, cards: list[dict], language: str = "en", body_vars: list[str] = None) -> requests.Response:
```

Private helpers may omit type hints when straightforward. No `TypedDict` or `dataclass` usage — all structured data is plain `dict`.
