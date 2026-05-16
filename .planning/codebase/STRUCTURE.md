# Codebase Structure

_Generated: 2026-05-16_

## Directory Layout

```
numobel/
├── app/                        # Core application package
│   ├── __init__.py
│   ├── config.py               # All constants + shared paths (load_dotenv here)
│   ├── log.py                  # Centralized logger — all modules import from here
│   ├── rag.py                  # RAG core: embed, retrieve, generate, ingest
│   ├── webhook.py              # Flask app + /webhook endpoint
│   ├── admin.py                # Admin WhatsApp state machine
│   ├── router.py               # dispatch() — routes answer to send function
│   ├── history.py              # Per-user conversation history (disk, fcntl)
│   ├── messaging/              # wa2mation HTTP send functions
│   │   ├── __init__.py         # Re-exports send_text, send_media, send_interactive, send_carousel
│   │   ├── client.py           # Lazy singleton requests.Session with auth headers
│   │   ├── text.py             # send_text()
│   │   ├── media.py            # send_media()
│   │   ├── interactive.py      # send_interactive()
│   │   └── carousel.py         # send_carousel()
│   └── refinement/             # LLM-assisted Q&A pair refinement
│       ├── __init__.py         # Re-exports full public API
│       ├── constraints.py      # WhatsApp rule constants + validate_pair()
│       ├── prompts.py          # System prompt builders
│       ├── storage.py          # All JSONL I/O, refine state, prefs, pair locking
│       └── engine.py           # chat_turn(), extract_patterns(), _normalize_refined()
├── training/                   # Offline training pipeline (CLI scripts)
│   ├── __init__.py
│   ├── utils.py                # load_jsonl(), choose_model() — shared by both scripts
│   ├── generate_qna.py         # Batch Q&A generation from product catalogue
│   ├── approve_qna_cli.py      # Interactive CLI approval + refinement
│   ├── admin_refine_state/     # Per-admin refinement state JSON (gitignored)
│   ├── admin_preferences/      # Per-admin style preferences JSON (gitignored)
│   └── qna_pairs/
│       ├── pending.jsonl       # LLM-generated pairs (gitignored, re-generatable)
│       └── approved.jsonl      # Human-approved pairs (tracked in git — recovery source)
├── tests/                      # pytest unit tests
│   ├── __init__.py
│   ├── test_admin.py
│   ├── test_constraints.py
│   ├── test_engine.py
│   ├── test_generate_qna.py
│   ├── test_approve_qna_cli.py
│   ├── test_history.py
│   ├── test_rag_chatbot.py
│   ├── test_router.py
│   ├── test_storage.py
│   ├── test_webhook.py
│   └── messaging/
│       ├── __init__.py
│       ├── test_carousel.py
│       ├── test_client.py
│       ├── test_interactive.py
│       ├── test_media.py
│       └── test_text.py
├── scripts/
│   └── test_messaging.py       # Live integration test — sends all 4 message types
├── docs/
│   └── wa2mation_doc.md        # wa2mation API reference
├── data/                       # Product data (gitignored)
│   ├── Products.csv            # Raw product catalogue
│   └── clean_products.json     # Cleaned JSON output (199 products, 5 brands)
├── chroma_db/                  # ChromaDB persistence (gitignored)
├── sessions/                   # Per-user WhatsApp history JSON (gitignored)
├── logs/                       # Application logs (gitignored)
├── rag_chatbot.py              # CLI entry point — thin re-export of app.rag
├── clean_products.py           # CSV → clean JSON data pipeline
├── Dockerfile
├── .dockerignore
├── requirements.txt
├── pytest.ini
├── .env.example
└── config.example.json
```

## Directory Purposes

**`app/`:**
- Purpose: All runtime application code
- Contains: Flask server, RAG pipeline, messaging, admin state machine, refinement engine
- Key files: `app/webhook.py` (server entry), `app/rag.py` (RAG core), `app/config.py` (all constants)

**`app/messaging/`:**
- Purpose: Thin wrappers over the four wa2mation REST endpoints
- Contains: One module per message type + shared HTTP client singleton
- Key files: `client.py` (auth session), `__init__.py` (barrel re-export)
- Pattern: Import only via `from app.messaging import send_text, send_media, send_interactive, send_carousel`

**`app/refinement/`:**
- Purpose: LLM-assisted Q&A pair refinement used by both admin WhatsApp flow and training CLI
- Contains: Constraints (single source of truth), LLM engine, JSONL storage, prompt builders
- Key files: `constraints.py` (all WhatsApp limits), `storage.py` (all JSONL ops), `engine.py` (LLM calls)

**`training/`:**
- Purpose: Offline pipeline to generate and approve Q&A training pairs
- Contains: Generation script, CLI approval script, shared utilities
- Generated/not committed: `admin_refine_state/`, `admin_preferences/`, `qna_pairs/pending.jsonl`
- Committed: `qna_pairs/approved.jsonl` (human-curated ground truth, git recovery source)

**`tests/`:**
- Purpose: pytest unit tests mirroring the app module structure
- Contains: One test file per app module; `tests/messaging/` mirrors `app/messaging/`

**`sessions/`:**
- Purpose: Per-user conversation history (gitignored, runtime-generated)
- Contains: `{phone}.json` files written by `app/history.py`

**`chroma_db/`:**
- Purpose: ChromaDB vector store persistence (gitignored)
- Generated: Yes, by `rag.ingest_data()` or `python rag_chatbot.py --ingest`
- Recovery: Re-run ingest from `data/clean_products.json` + `training/qna_pairs/approved.jsonl`

## Key File Locations

**Entry Points:**
- `app/webhook.py`: Flask server, run as `python -m app.webhook`
- `rag_chatbot.py`: CLI chat entry point (`python rag_chatbot.py`, `python rag_chatbot.py --ingest`)
- `training/generate_qna.py`: Batch Q&A generation
- `training/approve_qna_cli.py`: CLI approval loop

**Configuration:**
- `app/config.py`: All constants, all shared paths — edit constants here only
- `.env`: wa2mation + Anthropic credentials (gitignored; see `.env.example`)
- `config.json`: Admin phone list + `admin_timeout_minutes` (gitignored; see `config.example.json`)
- `pytest.ini`: pytest configuration

**Core Logic:**
- `app/rag.py`: `rewrite_query()`, `retrieve()`, `generate_answer()`, `ingest_data()`, `ingest_qna_pair()`
- `app/router.py`: `dispatch()` — sole routing logic, no LLM calls
- `app/admin.py`: `handle_admin()`, `AdminState` enum, in-memory `_sessions`
- `app/refinement/constraints.py`: `validate_pair()` and all WhatsApp limit constants
- `app/refinement/storage.py`: All JSONL reads/writes, pair locking, refine state persistence

**Training Data:**
- `training/qna_pairs/approved.jsonl`: Source of truth for human-approved Q&A pairs (committed)
- `training/qna_pairs/pending.jsonl`: LLM-generated pairs awaiting review (gitignored)
- `data/clean_products.json`: 199-product cleaned catalogue (gitignored)

## Naming Conventions

**Files:**
- Modules use `snake_case.py` throughout
- Test files mirror source: `app/rag.py` → `tests/test_rag_chatbot.py`; `app/messaging/text.py` → `tests/messaging/test_text.py`

**Functions:**
- Public functions: `snake_case` (e.g. `send_text`, `get_collection`, `load_history`)
- Private helpers: `_snake_case` prefix (e.g. `_is_duplicate`, `_keyword_fallback`, `_normalize_refined`)

**Variables:**
- Module-level singletons: `_snake_case` (e.g. `_session`, `_sessions`, `_seen_ids`, `_anthropic_client`)
- Constants: `SCREAMING_SNAKE_CASE` (e.g. `EMBED_MODEL`, `TOP_K`, `BODY_MAX_CHARS`)

**Classes:**
- Enums: `PascalCase` extending `str` for JSON serializability (e.g. `AdminState(str, Enum)`)

## Module Dependency Graph

```
app/config.py          ← (no app imports)
app/log.py             ← (no app imports)
app/refinement/
  constraints.py       ← (no app imports)
  prompts.py           ← (no app imports)
  storage.py           ← app/config.py
  engine.py            ← app/log.py, app/refinement/constraints.py,
                         app/refinement/prompts.py, app/refinement/storage.py
app/messaging/
  client.py            ← (no app imports)
  text.py              ← app/log.py, app/messaging/client.py
  media.py             ← app/log.py, app/messaging/client.py
  interactive.py       ← app/log.py, app/messaging/client.py
  carousel.py          ← app/log.py, app/messaging/client.py
  __init__.py          ← .text, .media, .interactive, .carousel
app/history.py         ← app/config.py, app/log.py
app/rag.py             ← app/config.py, app/log.py
app/router.py          ← app/log.py, app/messaging
app/admin.py           ← app/rag.py, app/config.py, app/log.py,
                         app/messaging, app/refinement
app/webhook.py         ← app/rag.py, app/config.py, app/log.py,
                         app/router.py, app/history.py, app/admin.py
training/utils.py      ← app/config.py, app/log.py
training/generate_qna.py ← app/config.py, app/log.py, app/refinement/storage.py,
                           app/refinement/constraints.py, training/utils.py
rag_chatbot.py         ← app/config.py, app/rag.py  (thin re-export)
```

## Public API Surface

**`app/rag.py`** (consumed by `app/webhook.py`, `app/admin.py`, `rag_chatbot.py`):
- `get_collection()` → `chromadb.Collection`
- `ingest_data(collection)` → `int` (chunk count)
- `rewrite_query(query, history)` → `str`
- `retrieve(collection, query, top_k)` → `list[dict]`
- `generate_answer(query, context_chunks, history)` → `dict`
- `ingest_qna_pair(pair, collection)` → `str` (doc_id)

**`app/router.py`** (consumed by `app/webhook.py`):
- `dispatch(phone, result, hits)` → `None`

**`app/messaging/`** (consumed by `app/router.py`, `app/admin.py`):
- `send_text(phone, message)` → `requests.Response`
- `send_media(phone, url, caption, media_type)` → `requests.Response`
- `send_interactive(phone, body, buttons, header, footer)` → `requests.Response`
- `send_carousel(phone, template_name, cards, language, body_vars)` → `requests.Response`

**`app/refinement/`** (consumed by `app/admin.py`, `training/approve_qna_cli.py`):
- `chat_turn(state, admin_msg, model, api_key, phone)` → `(str, dict | None)`
- `extract_patterns(chat_history, model, api_key)` → `list[dict]`
- `validate_pair(pair)` → `list[str]` (violations)
- `ensure_pair_id(pair)` → `dict`
- `load_next_pending(skip_locked)` → `dict | None`
- `mark_approved(identifier)`, `append_approved(pair)`, `load_style_examples(limit)`
- `lock_pair(identifier, phone)`, `unlock_pair(identifier)`, `is_locked(identifier)`
- `load_refine_state(phone)`, `save_refine_state(phone, state)`, `clear_refine_state(phone)`
- `load_admin_prefs(phone)`, `append_admin_prefs(phone, patterns)`

**`app/history.py`** (consumed by `app/webhook.py`):
- `load_history(phone)` → `list[dict]`
- `save_history(phone, history)` → `None`

**`app/admin.py`** (consumed by `app/webhook.py`):
- `is_admin(phone)` → `bool`
- `needs_admin_handling(phone, message)` → `bool`
- `handle_admin(phone, message, collection)` → `None`

## Where to Add New Code

**New message type (e.g. `send_list`):**
- Implement: `app/messaging/list.py` following the pattern in `app/messaging/text.py`
- Export: Add to `app/messaging/__init__.py`
- Route: Add a new `case` branch in `app/router.py:dispatch()`
- Constrain: Add constants and validation to `app/refinement/constraints.py`

**New admin command:**
- Add handling in `app/admin.py:handle_admin()`, following the existing if-chain pattern
- New states: Extend `AdminState` enum in `app/admin.py:28`

**New constant or path:**
- Add exclusively to `app/config.py` — never hardcode paths or limits elsewhere

**New Q&A field:**
- Schema shape is the Q&A pair dict; update `training/generate_qna.py:parse_pairs()`, `app/rag.py:ingest_qna_pair()`, and `app/refinement/constraints.py:validate_pair()` consistently

**New test:**
- Unit tests: `tests/test_{module_name}.py`
- Messaging sub-module tests: `tests/messaging/test_{module_name}.py`
- Follow existing mock patterns in `tests/test_router.py` and `tests/messaging/test_text.py`

## Special Directories

**`chroma_db/`:**
- Purpose: ChromaDB vector store with product chunks and approved Q&A embeddings
- Generated: Yes, by ingest pipeline
- Committed: No (gitignored)
- Recovery: `python rag_chatbot.py --ingest` + re-ingest approved pairs

**`sessions/`:**
- Purpose: Per-user conversation history files (`{phone}.json`)
- Generated: Yes, at runtime by `app/history.py`
- Committed: No (gitignored)
- Recovery: Auto-created on first user message

**`training/admin_refine_state/`:**
- Purpose: Crash-safe disk snapshot of in-progress admin refinement sessions (`{phone}.json`)
- Generated: Yes, after each `chat_turn()` call
- Committed: No (gitignored)
- Recovery: Deleted by `clear_refine_state(phone)` on session completion

**`training/admin_preferences/`:**
- Purpose: Per-admin learned style preferences (`{phone}.json`)
- Generated: Yes, when admin saves patterns during `TRAINING_CONFIRM_PREFS`
- Committed: No (gitignored)

**`training/qna_pairs/`:**
- `pending.jsonl`: LLM-generated pairs, never deleted, gitignored
- `approved.jsonl`: Human-approved ground truth, **committed to git** (recovery source for ChromaDB rebuild)

---

_Structure analysis: 2026-05-16_
