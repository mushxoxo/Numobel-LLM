# Architecture

_Generated: 2026-05-16_

## System Overview

```text
┌──────────────────────────────────────────────────────────────────────┐
│                     WhatsApp / wa2mation.com                         │
│                  (6–8 POST callbacks per user message)               │
└────────────────────────────┬─────────────────────────────────────────┘
                             │ POST /webhook
                             ▼
┌──────────────────────────────────────────────────────────────────────┐
│                    app/webhook.py  (Flask)                           │
│  • req=<8hex> correlation ID    • dedup (60s OrderedDict TTL)        │
│  • phone validation             • always returns HTTP 200            │
└──────────┬────────────────────────────────┬─────────────────────────┘
           │ admin?                         │ user
           ▼                                ▼
┌──────────────────┐            ┌───────────────────────────┐
│  app/admin.py    │            │  app/rag.py               │
│  State machine   │            │  rewrite_query()          │
│  (in-memory      │            │  retrieve()               │
│   _sessions{})   │            │  generate_answer()        │
└────────┬─────────┘            └────────────┬──────────────┘
         │                                   │
         │                          ┌────────▼──────────────┐
         │                          │  app/history.py        │
         │                          │  load/save (sessions/) │
         │                          └────────────────────────┘
         │                                   │
         └──────────────┬────────────────────┘
                        ▼
              ┌─────────────────┐
              │  app/router.py  │
              │  dispatch()     │
              └────────┬────────┘
                       ▼
        ┌──────────────────────────────┐
        │     app/messaging/           │
        │  send_text / send_media /    │
        │  send_interactive /          │
        │  send_carousel               │
        │  (via client.py singleton)   │
        └──────────────────────────────┘
                       │
                       ▼
              wa2mation REST API
              https://wa2mation.com/api/{uid}/contact/...
```

## Component Responsibilities

| Component | Responsibility | File |
|-----------|----------------|------|
| Flask webhook | HTTP entry point, dedup, phone validation, request correlation | `app/webhook.py` |
| RAG core | Embed, retrieve, generate structured answers, ingest | `app/rag.py` |
| Router | Route `generate_answer()` result to correct send function, no LLM calls | `app/router.py` |
| History | Per-user conversation persistence (disk, 5-min expiry, fcntl locks) | `app/history.py` |
| Admin handler | WhatsApp-based admin state machine for Q&A training workflow | `app/admin.py` |
| Messaging client | Lazy singleton `requests.Session` with wa2mation auth headers | `app/messaging/client.py` |
| Messaging senders | One module per message type (text, media, interactive, carousel) | `app/messaging/*.py` |
| Refinement engine | Multi-turn LLM chat for Q&A refinement, pattern extraction | `app/refinement/engine.py` |
| Refinement storage | All JSONL I/O: pending, approved, refine state, prefs, pair locking | `app/refinement/storage.py` |
| Constraints | WhatsApp rule constants and `validate_pair()` — single source of truth | `app/refinement/constraints.py` |
| Config | All constants, shared paths, `load_dotenv()` at import | `app/config.py` |
| Logger | Single named logger `rag_chatbot`, file + stdout | `app/log.py` |

## Pattern Overview

**Overall:** Layered pipeline — HTTP → Admin/RAG → Dispatch → Messaging

**Key Characteristics:**
- Every request returns HTTP 200 (non-200 causes wa2mation retry storms)
- In-memory deduplication via `OrderedDict` TTL at `app/webhook.py:44`
- Single Gunicorn worker enforced — ChromaDB `PersistentClient` is not safe for concurrent writes from multiple processes; gevent handles concurrent callbacks cooperatively
- All LLM calls go through `ollama.chat()` or `anthropic.Anthropic()` — no REST calls managed manually
- Lazy credentials: `messaging/client.py` and `refinement/engine.py` only validate credentials on first use

## Layers

**HTTP Layer:**
- Purpose: Receive wa2mation webhooks, validate, deduplicate, route
- Location: `app/webhook.py`
- Contains: Flask app, dedup cache, startup ingest trigger
- Depends on: `app/rag.py`, `app/router.py`, `app/history.py`, `app/admin.py`
- Used by: wa2mation webhook callbacks

**RAG Layer:**
- Purpose: Query reformulation, vector retrieval, structured answer generation
- Location: `app/rag.py`
- Contains: `rewrite_query()`, `get_embedding()`, `retrieve()`, `generate_answer()`, `ingest_data()`, `ingest_qna_pair()`
- Depends on: Ollama (local LLM + embeddings), ChromaDB, `app/config.py`, `app/log.py`
- Used by: `app/webhook.py`, `app/admin.py`, `rag_chatbot.py` CLI

**Dispatch Layer:**
- Purpose: Route structured answer to correct messaging function
- Location: `app/router.py`
- Contains: `dispatch(phone, result, hits)` — match/case on `message_type`
- Depends on: `app/messaging/`
- Used by: `app/webhook.py`

**Messaging Layer:**
- Purpose: Send WhatsApp messages via wa2mation REST API
- Location: `app/messaging/`
- Contains: `send_text`, `send_media`, `send_interactive`, `send_carousel`
- Depends on: `app/messaging/client.py` (lazy singleton session), `app/log.py`
- Used by: `app/router.py`, `app/admin.py`

**Admin Layer:**
- Purpose: WhatsApp-native state machine for training workflow
- Location: `app/admin.py`
- Contains: `AdminState` enum, `handle_admin()`, in-memory `_sessions` dict
- Depends on: `app/rag.py`, `app/messaging/`, `app/refinement/`
- Used by: `app/webhook.py`

**Refinement Layer:**
- Purpose: LLM-assisted Q&A pair refinement and style preference extraction
- Location: `app/refinement/`
- Contains: `engine.py` (LLM chat), `storage.py` (JSONL ops), `constraints.py` (validation), `prompts.py` (system prompts)
- Depends on: Ollama or Anthropic API, `app/config.py`
- Used by: `app/admin.py`, `training/approve_qna_cli.py`

## Data Flow

### Primary Request Path (user message)

1. wa2mation POSTs to `POST /webhook` (`app/webhook.py:63`)
2. Extract `phone`, `user_message`, `message_id`; validate phone regex (`app/webhook.py:77`)
3. Dedup check: if `message_id` seen within 60s, return `{"status": "ignored"}` (`app/webhook.py:81`)
4. If `needs_admin_handling()` → `handle_admin()` → return (`app/webhook.py:88`)
5. `load_history(phone)` → list of prior turns from `sessions/{phone}.json` (`app/history.py:29`)
6. `rewrite_query(user_message, history)` → standalone question via `llama3.2` (`app/rag.py:193`)
7. `retrieve(collection, search_query)` → embed query with `mxbai-embed-large`, top-5 ChromaDB chunks (`app/rag.py:221`)
8. `generate_answer(search_query, hits, history)` → structured dict with `message_type`, `content`, `buttons`, `image_url` (`app/rag.py:286`)
9. `dispatch(phone, result, hits)` → calls correct send function (`app/router.py:24`)
10. `save_history(phone, history)` → append user + assistant turn to disk (`app/history.py:54`)

### Admin Approval Path

1. Admin sends `:admin on` over WhatsApp → enters `AdminState.MENU`
2. Sends `QnA Review` or `:train` → `load_next_pending(skip_locked=True)` → pair displayed
3. Sends `Refine` → `lock_pair(identifier, phone)` writes `in_review` to `pending.jsonl` → enters `AdminState.TRAINING_CHAT`
4. Each message → `chat_turn(state, msg, model, api_key, phone)` → LLM returns `{reply, pair}` → `_normalize_refined()` applied (`app/refinement/engine.py:72`)
5. Sends `Approve` → `validate_pair()` check → preview sent → `AdminState.TRAINING_CONFIRM_SAVE`
6. Sends `Confirm Save` → `extract_patterns()` from chat history → if patterns found → `AdminState.TRAINING_CONFIRM_PREFS`
7. Saves accepted prefs → `mark_approved()` + `append_approved()` + `rag.ingest_qna_pair()` → loads next pair

### Data Ingestion Path (startup / CLI)

1. `get_collection()` opens/creates ChromaDB at `chroma_db/` (`app/rag.py:28`)
2. `ingest_data(collection)` reads `data/clean_products.json`, calls `product_to_text()`, `chunk_text()`, batch-embeds via Ollama, upserts (`app/rag.py:128`)
3. Chunk IDs are MD5-stable → re-ingest is idempotent (`app/rag.py:112`)

**State Management:**
- User sessions: disk JSON at `sessions/{phone}.json` with 5-minute expiry; fcntl exclusive lock on `.lock` file during writes
- Admin sessions: in-memory `_sessions` dict in `app/admin.py:24`; crash-safe disk snapshot at `training/admin_refine_state/{phone}.json` written after each refinement turn
- Dedup cache: in-memory `OrderedDict` in `app/webhook.py:44`; no persistence (reset on restart)
- Pair locking: `in_review` field written inline into `pending.jsonl` rows

## Key Abstractions

**`generate_answer()` Result Dict:**
- Purpose: Structured output passed from RAG layer through Router to Messaging
- Shape: `{message_type, content, buttons, image_url, prompt_tokens, completion_tokens}`
- Location: `app/rag.py:365`
- Pattern: message_type drives dispatch branch selection in `app/router.py`

**`AdminState` Enum:**
- Purpose: JSON-serializable state machine states (extends `str`)
- Values: `idle`, `menu`, `training_review`, `training_chat`, `training_confirm_save`, `training_confirm_prefs`
- Location: `app/admin.py:28`
- Pattern: State stored in `_sessions[phone]["state"]`, checked in `handle_admin()` via if-chain

**Q&A Pair Schema:**
- Purpose: Unit of training data flowing from generation → approval → ChromaDB
- Fields: `pair_id` (UUID4), `question`, `answer`, `message_type`, `buttons`, `image_url`, `product`, `approved`, `in_review`
- Persisted: `training/qna_pairs/pending.jsonl` (LLM-generated, gitignored) and `training/qna_pairs/approved.jsonl` (human-curated, tracked in git)

## Entry Points

**Flask Server:**
- Location: `app/webhook.py` (run as `python -m app.webhook`)
- Triggers: wa2mation HTTP POST
- Responsibilities: Dedup, routing to admin or RAG pipeline, always return 200

**CLI Chat:**
- Location: `rag_chatbot.py` (thin re-export of `app/rag.main()`)
- Triggers: `python rag_chatbot.py` or `python rag_chatbot.py --ingest`
- Responsibilities: Interactive terminal chat, optional ChromaDB rebuild

**Training Generation:**
- Location: `training/generate_qna.py`
- Triggers: `python training/generate_qna.py`
- Responsibilities: Generate Q&A pairs from product catalogue, append to `pending.jsonl`

## Architectural Constraints

- **Single process required:** ChromaDB `PersistentClient` is not multi-process safe. Gunicorn must run with `--workers 1`. Gevent handles concurrent wa2mation callbacks via cooperative concurrency.
- **Global state:** `_seen_ids` OrderedDict in `app/webhook.py:44`; `_sessions` dict in `app/admin.py:24`; `_session` requests.Session in `app/messaging/client.py:15`; `_anthropic_client` in `app/refinement/engine.py:17`. All are module-level singletons.
- **Linux-only:** `app/history.py` uses `fcntl` for file locking — not portable to Windows.
- **Always HTTP 200:** `app/webhook.py` catches all exceptions and returns 200 to prevent wa2mation retry storms (`app/webhook.py:111`).
- **Faithful replay:** When top ChromaDB hit has `source: approved_training` and LLM agrees on `message_type`, `buttons`/`image_url` are copied verbatim from chunk metadata (`app/rag.py:346`).

## Anti-Patterns

### Direct JSONL manipulation outside `storage.py`

**What happens:** `training/generate_qna.py` calls `ensure_pair_id()` and writes to `pending.jsonl` directly via `append_to_pending()`, bypassing the storage module's `_load_pending_lines()` / `_write_pending_lines()` helpers.
**Why it's wrong:** Two writers to the same JSONL without coordination can interleave lines; also duplicates I/O logic.
**Do this instead:** Add a `append_pending(pair)` function to `app/refinement/storage.py` and call it from `generate_qna.py`.

### Hardcoded carousel template name in router

**What happens:** `_CAROUSEL_TEMPLATE = "numobel_catalogue_4"` is hardcoded at `app/router.py:7`.
**Why it's wrong:** Adding a new approved template requires a code change instead of a config change.
**Do this instead:** Move carousel template name to `app/config.py`.

## Error Handling

**Strategy:** Catch-and-continue with fallback responses. Never surface 500 to wa2mation.

**Patterns:**
- LLM JSON parse failure → `_keyword_fallback(query)` determines type from keywords (`app/rag.py:335`)
- No buttons for interactive → fall back to `send_text` (`app/router.py:48`)
- No image for media/carousel → fall back to `send_text` with warning log (`app/router.py:55,74`)
- Unhandled exception in webhook → log exception, return `{"status": "error", "message": ...}` with HTTP 200 (`app/webhook.py:111`)
- Refinement LLM parse failure → `_FALLBACK_REPLY` string sent to admin (`app/refinement/engine.py:108`)

## Cross-Cutting Concerns

**Logging:** All modules import `from app.log import get_logger; log = get_logger()`. Single named logger `rag_chatbot`, file at `logs/rag_chatbot.log` (DEBUG), stdout (INFO). Request-scoped lines prefixed `req=<8hex>`.

**Validation:** WhatsApp constraints live exclusively in `app/refinement/constraints.py`. Called at generation time (`training/generate_qna.py:192`) and at admin approval time (`app/admin.py:325`).

**Authentication:** wa2mation credentials validated lazily by `app/messaging/client.py` on first HTTP send. Anthropic key validated lazily by `app/refinement/engine.py` on first refinement call.

---

_Architecture analysis: 2026-05-16_
