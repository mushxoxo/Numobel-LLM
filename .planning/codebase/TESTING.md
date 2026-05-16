# Testing Patterns
_Generated: 2026-05-16_

## Test Framework

**Runner:** pytest 9.0.3
- Config: `pytest.ini` at project root — sets `testpaths = tests`
- Assertion library: built-in `assert` (pytest rewriting)
- Mocking: `unittest.mock` (`patch`, `MagicMock`, `mock_open`) + `pytest-mock` 3.15.1 (`monkeypatch`)

**Run Commands:**
```bash
python -m pytest                        # run all 169 tests
python -m pytest tests/test_router.py   # run a single file
python -m pytest -q                     # quiet summary output
python scripts/test_messaging.py        # live integration test (requires .env + config.json)
```

## Test File Organization

**Location:** Separate `tests/` directory, not co-located with source.

**Naming:** `test_<module>.py` mirroring the source path:
- `app/router.py` → `tests/test_router.py`
- `app/webhook.py` → `tests/test_webhook.py`
- `app/admin.py` → `tests/test_admin.py`
- `app/history.py` → `tests/test_history.py`
- `app/refinement/constraints.py` → `tests/test_constraints.py`
- `app/refinement/engine.py` → `tests/test_engine.py`
- `app/refinement/storage.py` → `tests/test_storage.py`
- `app/rag.py` → `tests/test_rag_chatbot.py`
- `training/generate_qna.py` → `tests/test_generate_qna.py`
- `training/approve_qna_cli.py` → `tests/test_approve_qna_cli.py`

**Sub-package:** `tests/messaging/` for messaging layer tests:
- `tests/messaging/test_text.py`
- `tests/messaging/test_media.py`
- `tests/messaging/test_interactive.py`
- `tests/messaging/test_carousel.py`
- `tests/messaging/test_client.py`

Both `tests/` and `tests/messaging/` have `__init__.py` files.

**Structure:** Flat functions only — no `class Test*` grouping. Tests within a file are grouped by the function or concern being tested, separated by the section-divider comment style:
```python
# ─── interactive ──────────────────────────────────────────────────────────────
# ─── media ────────────────────────────────────────────────────────────────────
```

## Test Suite Overview

**Total:** 169 tests collected, 168 passing, 1 failing (`test_call_llm_routes_to_ollama` in `test_generate_qna.py` — `TypeError` in mock setup).

| File | Tests | What is covered |
|---|---|---|
| `test_router.py` | 14 | All 4 dispatch branches, fallback logic, image deduplication, body_var truncation |
| `test_webhook.py` | 10 | HTTP flow, dedup, admin bypass, null-body receipts, phone validation, exception safety |
| `test_admin.py` | 17 | State machine transitions, timeout, approve/refine/cancel flows |
| `test_history.py` | 10 | Load/save round-trip, expiry, corruption handling, MEMORY_LIMIT trimming |
| `test_constraints.py` | 22 | All message types, all constraint rules, URL detection in button labels |
| `test_engine.py` | 13 | `chat_turn`, `extract_patterns`, `_normalize_refined`, LLM fallback |
| `test_storage.py` | 14 | Refine state, preferences FIFO/dedup, pair locking, atomic write |
| `test_rag_chatbot.py` | 10 | JSON parsing, markdown fence stripping, fallback, token counting, ingest |
| `test_generate_qna.py` | 9 | Resume logic, few-shot selection, parse/validate, LLM routing |
| `test_approve_qna_cli.py` | unknown | CLI approval loop |
| `tests/messaging/test_client.py` | 5 | Session singleton, auth headers, URL construction, missing credentials |
| `tests/messaging/test_interactive.py` | 8 | Endpoint, button cap, header/footer omission |
| `tests/messaging/test_carousel.py` | 6 | Endpoint, body_vars mapping, language default |
| `tests/messaging/test_media.py` | varies | Media endpoint, caption, media_type |
| `tests/messaging/test_text.py` | varies | Text endpoint, phone/message fields |

## Mocking Strategy

**Framework:** `unittest.mock.patch` as context manager (not decorator). All patches are applied as `with patch("module.name") as mock:` blocks within the test function body.

**Why context manager over decorator:** Allows conditional patching and keeps the mock scope visually close to the assertion. Some tests use nested `with` blocks for multi-dependency patches:
```python
with patch("app.webhook.load_history", return_value=[]), \
     patch("app.webhook.rag.rewrite_query", return_value="hi"), \
     patch("app.webhook.rag.retrieve", return_value=[]), \
     patch("app.webhook.rag.generate_answer", return_value=RAG_RESULT), \
     patch("app.webhook.dispatch") as mock_dispatch, \
     patch("app.webhook.save_history"):
    client.post("/webhook", json=INCOMING)
```

**What is always mocked:**
- All wa2mation HTTP calls — `wa2mation_post` patched on the messaging module that uses it (e.g., `app.messaging.carousel.wa2mation_post`)
- LLM calls — `app.rag.ollama.chat`, `engine_module._call_llm`
- ChromaDB collection — `MagicMock()` with `.count()` returning a value
- Filesystem paths redirected using `monkeypatch.setattr` (not patched via `patch`)

**What is NOT mocked:**
- Pure logic functions: `validate_pair()`, `_normalize_refined()`, `_keyword_fallback()`, `_images_from_hits()` — tested with real inputs
- JSON serialisation/deserialisation
- `datetime` calculations (history expiry tests use real `timedelta`)

**Singleton reset pattern:**
The `app.messaging.client._session` singleton is reset before each messaging test via `autouse` fixture:
```python
@pytest.fixture(autouse=True)
def reset_client(monkeypatch):
    import app.messaging.client as client_mod
    monkeypatch.setattr(client_mod, "_session", None)
```

**In-memory admin sessions reset pattern:**
```python
@pytest.fixture(autouse=True)
def reset_sessions():
    admin_module._sessions.clear()
    yield
    admin_module._sessions.clear()
```

## Fixtures

**Scope:** All fixtures are function-scoped (default). No session- or module-scoped fixtures.

**`autouse=True` fixtures** for implicit setup/teardown:
- `reset_client` (messaging tests) — resets the HTTP session singleton
- `reset_sessions` (admin tests) — clears in-memory session dict
- `tmp_sessions` (history tests) — redirects `_SESSIONS_DIR` to `tmp_path`
- `tmp_dirs` (storage tests) — redirects all four storage paths to `tmp_path`

**Filesystem isolation with `tmp_path`:**
Tests that touch the filesystem redirect module-level path constants using `monkeypatch.setattr`:
```python
@pytest.fixture(autouse=True)
def tmp_dirs(tmp_path, monkeypatch):
    monkeypatch.setattr(storage_module, '_REFINE_STATE_DIR', tmp_path / 'refine_state')
    monkeypatch.setattr(storage_module, '_PREFS_DIR',        tmp_path / 'prefs')
    monkeypatch.setattr(storage_module, '_PENDING_PATH',     tmp_path / 'pending.jsonl')
    yield tmp_path
```

**Flask test client fixture:**
```python
@pytest.fixture
def client():
    with patch("app.webhook.rag.get_collection"), \
         patch("app.webhook.collection") as mock_col:
        mock_col.count.return_value = 1  # skip ingest
        from app.webhook import app
        app.config["TESTING"] = True
        with app.test_client() as c:
            yield c
```
Import inside fixture body to avoid module-level side effects on `app/webhook.py` startup.

## Test Data / Builders

**Helper functions prefixed with `_` build minimal valid payloads:**
```python
# test_router.py
def _result(message_type, content="Hello", buttons=None, image_url=None):
    return {"message_type": message_type, "content": content, "buttons": buttons, "image_url": image_url}

def _hit(images="https://example.com/img1.jpg|https://example.com/img2.jpg"):
    return {"metadata": {"images": images}}

# test_constraints.py
def _pair(**kwargs):
    base = {'question': 'Q?', 'answer': 'A.', 'message_type': 'text'}
    return {**base, **kwargs}

# test_admin.py
def _active_session(state, pair=None):
    return {"state": state, "current_pair": pair, "last_active": datetime.utcnow()}
```

**Module-level constants for shared payloads:**
```python
PHONE = "919999999999"
SAMPLE_PAIR = {"question": "What is a stacker?", "answer": "A wooden stacking toy.", ...}
INCOMING = {"contact": {"phone_number": "919999999999"}, "message": {"body": "hi there"}}
RAG_RESULT = {"message_type": "interactive", "content": "Welcome!", ...}
```

No shared fixture files or `conftest.py` — all test data is defined locally in each test file.

## Test Types

**Unit tests (all 169 tests):**
All tests are unit tests. Each test exercises one function or one code path in isolation. External I/O (HTTP, filesystem, LLM) is always mocked.

**Integration tests (live, manual):**
`scripts/test_messaging.py` — sends all 4 message types to a real WhatsApp number. Requires `.env` with credentials and `config.json` with a phone number. Not part of the pytest suite. Run manually with `python scripts/test_messaging.py`.

**E2E / load tests:** None.

## Coverage Gaps

**`app/rag.py` — partially tested:**
- `product_to_text()`, `chunk_text()`, `stable_id()`, `get_embedding()`, `ingest_data()`, `rewrite_query()`, `retrieve()` have no dedicated unit tests
- `generate_answer()` and `ingest_qna_pair()` are tested in `test_rag_chatbot.py`
- The "faithful replay" path (copy `buttons`/`image_url` from approved training chunk) has no test

**`app/webhook.py` — dedup logic untested:**
- `_is_duplicate()` eviction of expired entries is not directly tested
- Only the happy path (duplicate blocked) is exercised via mock in `test_webhook.py`

**`training/approve_qna_cli.py` — unknown coverage:**
`test_approve_qna_cli.py` exists but was not observed to have meaningful tests (file not read in full). The CLI is interactive and harder to test.

**Admin state machine — partial coverage:**
- `training_confirm_save` and `training_confirm_prefs` states are not covered in `test_admin.py`
- Pattern extraction flow after refinement is not tested end-to-end

**`app/messaging/text.py` and `app/messaging/media.py`:**
Test files exist (`test_text.py`, `test_media.py`) but were not read in detail. Expected coverage mirrors the `test_interactive.py` / `test_carousel.py` pattern.

**No tests for:**
- `app/log.py` — logging setup, `_JsonFormatter`
- `app/config.py` — constant values, `load_dotenv()` side effects
- `clean_products.py` — CSV parsing pipeline
- `training/utils.py` — `load_jsonl()`, `choose_model()`
- `rag_chatbot.py` — the CLI entry point (`main()`)

## Known Test Issue

`tests/test_generate_qna.py::test_call_llm_routes_to_ollama` fails with a `TypeError`. The test mocks `gqna.ollama.chat` but the `call_llm` function's Ollama path may have changed its call signature. This is the only failing test in the suite as of 2026-05-16.
