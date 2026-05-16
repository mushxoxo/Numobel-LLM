# External Integrations
_Generated: 2026-05-16_

## APIs & External Services

### wa2mation (WhatsApp Gateway)

- **What it does:** Bridges the application to the WhatsApp Business API. All outbound WhatsApp messages are sent through wa2mation's REST API, and all inbound messages arrive as webhook POSTs from wa2mation.
- **SDK/Client:** Raw HTTP via `requests.Session` — no official SDK. Singleton session created in `app/messaging/client.py`.
- **Base URL pattern:** `https://wa2mation.com/api/{VENDOR_UID}/contact/{endpoint}`
- **Auth:** Bearer token — `Authorization: Bearer {WA2MATION_API_KEY}` header set once on session creation.
- **Credentials:** `WA2MATION_API_KEY` and `WA2MATION_VENDOR_UID` environment variables.
- **Lazy init:** Session and credentials validated only on first outbound call — training scripts can import `app.config` without requiring wa2mation credentials.

**Outbound endpoints used:**

| Endpoint | Function | File |
|---|---|---|
| `send-message` | `send_text(phone, message)` | `app/messaging/text.py` |
| `send-media-message` | `send_media(phone, url, caption, media_type)` | `app/messaging/media.py` |
| `send-interactive-message` | `send_interactive(phone, body, buttons, header, footer)` | `app/messaging/interactive.py` |
| `send-carousel-template-message` | `send_carousel(phone, template_name, cards, language, body_vars)` | `app/messaging/carousel.py` |

**Inbound webhook:**
- Route: `POST /webhook` in `app/webhook.py`
- wa2mation fires 6–8 callbacks per user message (sent/delivered/read receipts)
- Deduplication: 60-second TTL in-memory `OrderedDict` keyed on `whatsapp_message_id`
- Non-message events (status callbacks with null body) return `{"status": "ignored"}`
- Always returns HTTP 200 — returning 500 causes wa2mation retry/duplicate processing

**Known limitation (discovered 2026-05-15):** wa2mation's carousel endpoint silently drops per-card variable substitution fields (`field_1`, `field_2`, `button_1`). Meta returns `#131008 Required parameter is missing`. Contact wa2mation support for the correct per-card variable API format.

### Anthropic Claude API

- **What it does:** Powers the admin refinement chat loop and optionally the training Q&A generation pipeline.
- **SDK:** `anthropic==0.100.0` (official Python SDK)
- **Client:** Lazy singleton in `app/refinement/engine.py` (`_anthropic_client`), instantiated on first Claude call.
- **Auth:** `ANTHROPIC_API_KEY` environment variable.
- **Models used:**
  - `claude-sonnet-4-6` — refinement chat (`app/refinement/engine.py`) and optional Q&A generation (`training/generate_qna.py`)
- **API surface used:**
  ```python
  client.messages.create(
      model=model,
      max_tokens=1024,
      system=system_content,
      messages=user_messages,
  )
  ```
- **Optional:** If `ANTHROPIC_API_KEY` is unset, the system falls back to Ollama for all LLM calls. The refinement engine checks `model.startswith('claude')` before using the Anthropic client.

### Ollama (Local LLM Inference)

- **What it does:** Runs LLM inference locally. Used for embedding, query rewriting, answer generation, and (optionally) training data generation.
- **SDK:** `ollama==0.5.1` (official Python client)
- **Auth:** None — local service, no credentials required.
- **Endpoint:** Default Ollama server (`http://localhost:11434` or configured via `OLLAMA_HOST`)
- **Models required:**
  - `mxbai-embed-large` — all embedding operations (`app/rag.py`: `get_embedding()`, batch ingest)
  - `llama3.2` — query rewriting (`rewrite_query()`) and answer generation (`generate_answer()`)
  - `qwen2.5:14b` — optional Q&A generation (`training/generate_qna.py`)

**API calls used:**
```python
ollama.embed(model=EMBED_MODEL, input=text_or_list)   # embedding
ollama.chat(model=LLM_MODEL, messages=messages)        # chat generation
ollama.generate(model=LLM_MODEL, prompt=prompt)        # query rewriting
```

## Data Storage

### ChromaDB (Vector Store)

- **Type:** Embedded vector database, persisted on local filesystem
- **Client:** `chromadb==1.0.15`, `chromadb.PersistentClient`
- **Persistence path:** `chroma_db/` (relative to project root; gitignored)
- **Collection:** `numobel_products` (cosine similarity, HNSW index)
- **Connection:** Direct filesystem access — no network, no auth
- **Concurrency constraint:** `PersistentClient` is not safe for concurrent writes across OS processes. Gunicorn must run with `--workers 1`.
- **Usage:** `app/rag.py` — `get_collection()`, `ingest_data()`, `retrieve()`, `ingest_qna_pair()`

### Filesystem (Session History)

- **Type:** JSON files, one per user phone number
- **Path:** `sessions/{phone}.json` (gitignored)
- **Concurrency:** `fcntl.flock` exclusive lock on write — Linux only (`app/history.py`)
- **Expiry:** Sessions older than 5 minutes are deleted on next load

### JSONL Files (Training Data)

- **Pending pairs:** `training/qna_pairs/pending.jsonl` (gitignored — LLM-generated, re-generatable)
- **Approved pairs:** `training/qna_pairs/approved.jsonl` (git-tracked — human-curated ground truth, recovery source)
- **All JSONL I/O:** delegated to `app/refinement/storage.py`

## Authentication & Identity

**wa2mation inbound:** No request signature verification — identity implied by wa2mation-controlled webhook URL. Phone number validation via regex `^\+?[0-9]{7,15}$` in `app/webhook.py`.

**Admin access:** Phone number allowlist in `config.json` (gitignored). Admin state machine in `app/admin.py`. No token/password — admin commands (`:admin on`, `:train`) only work from allowlisted numbers.

**Anthropic API:** API key in `.env`, passed to `anthropic.Anthropic(api_key=...)`.

**wa2mation outbound:** Bearer token in `Authorization` header on all requests.

## WhatsApp Templates

Two approved Meta carousel templates used by `send_carousel()` via `app/router.py`:

| Template name | Cards | Variables | Status |
|---|---|---|---|
| `numobel_catalogue_4` | 4 | `field_1`/`field_2` (body) | Active — used in `app/router.py` |
| `nutoy_stacker` | 2 | `field_1` (body) | Deprecated — replaced by `numobel_catalogue_4` |

Template messages bypass the 24-hour WhatsApp session window restriction.

## Monitoring & Observability

**Error tracking:** None (no Sentry, Datadog, etc.)

**Logs:**
- File: `logs/rag_chatbot.log` (DEBUG level, plain text)
- Stdout: INFO level; JSON format if `LOG_FORMAT=json`
- Per-request correlation: `req=<8hex>` prefix on all request-scoped log lines

## CI/CD & Deployment

**CI pipeline:** None detected (no `.github/workflows/`, no CircleCI, no GitLab CI)

**Hosting:** Cloud VM (Linux), run via gunicorn or Docker

**Tunnel:** ngrok required for development — paste public URL into wa2mation dashboard as webhook callback

## Webhooks

**Incoming:**
- `POST /webhook` — wa2mation fires this for every WhatsApp event (messages, delivery receipts, read receipts)

**Outgoing:**
- All four wa2mation REST endpoints listed above (send-message, send-media-message, send-interactive-message, send-carousel-template-message)

---
_Integration audit: 2026-05-16_
