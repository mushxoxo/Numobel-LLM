# Technology Stack
_Generated: 2026-05-16_

## Language

**Primary:**
- Python 3.12 — all application and training code
- Runtime confirmed in `Dockerfile`: `FROM python:3.12-slim`

## Runtime

**Environment:**
- CPython 3.12 (slim Docker image in production, system Python in dev)
- Linux required — `app/history.py` uses `fcntl` for file locking (POSIX-only)

**Package Manager:**
- pip (no lockfile beyond `requirements.txt`)
- `requirements.txt` — pinned versions, no `pip-compile`/Poetry/uv

## Frameworks

**Web:**
- Flask 3.1.1 — HTTP webhook server (`app/webhook.py`), single route `POST /webhook`

**WSGI / Concurrency:**
- Gunicorn 26.0.0 — production WSGI server
- gevent 24.11.1 — cooperative concurrency worker class (`--worker-class gevent`)
- Single worker required: ChromaDB `PersistentClient` is not safe for multi-process concurrent writes

## Key Dependencies (pinned in `requirements.txt`)

| Package | Version | Purpose |
|---|---|---|
| `anthropic` | 0.100.0 | Anthropic Claude API client — refinement engine + optional Q&A generation |
| `chromadb` | 1.0.15 | Vector store, persisted at `chroma_db/`, cosine similarity via HNSW |
| `flask` | 3.1.3 | WhatsApp webhook HTTP server |
| `gevent` | 24.11.1 | Cooperative concurrency for Gunicorn |
| `gunicorn` | 26.0.0 | Production WSGI server |
| `ollama` | 0.5.1 | Local LLM inference (embedding + generation) |
| `python-dotenv` | 1.1.0 | `.env` loading, called once in `app/config.py` |
| `requests` | 2.32.3 | HTTP client for wa2mation REST API (`app/messaging/client.py`) |

**Dev / Testing:**
| Package | Version | Purpose |
|---|---|---|
| `pytest` | 9.0.3 | Test runner, config in `pytest.ini` (`testpaths = tests`) |
| `pytest-mock` | 3.15.1 | Mocking support in unit tests |

## Local LLM Models (via Ollama — not in requirements.txt)

These models must be pulled separately with `ollama pull <model>`:

| Model | Purpose |
|---|---|
| `mxbai-embed-large` | Embedding model — all vector operations (`EMBED_MODEL` in `app/config.py`) |
| `llama3.2` | Primary chat/generation model (`LLM_MODEL` in `app/config.py`) |
| `qwen2.5:14b` | Optional Q&A generation in training pipeline (`training/generate_qna.py`) |

## Infrastructure / Deployment

**Containerization:**
- Docker — `Dockerfile` at project root, `python:3.12-slim` base
- `.dockerignore` present
- Exposed port: 5000

**Production command:**
```bash
gunicorn --bind 0.0.0.0:5000 --workers 1 --worker-class gevent \
         --worker-connections 100 --timeout 120 app.webhook:app
```

**Tunnel (development):**
- ngrok — exposes local port 5000 to wa2mation webhook callback URL

**Storage:**
- `chroma_db/` — ChromaDB persistence directory (gitignored, local filesystem)
- `sessions/` — Per-user conversation history as JSON files (gitignored)
- `training/qna_pairs/` — JSONL files for pending and approved Q&A pairs

## Configuration

**Environment variables** (loaded via `python-dotenv` in `app/config.py`):
- `WA2MATION_API_KEY` — wa2mation bearer token
- `WA2MATION_VENDOR_UID` — wa2mation vendor identifier (used in API URL)
- `ANTHROPIC_API_KEY` — Anthropic Claude API key (optional; refinement + generation)
- `FLASK_DEBUG` — set `true` to enable Flask debug mode (default: off)
- `LOG_FORMAT` — set `json` for structured JSON log output (default: plain text)

**App constants** (`app/config.py`):
- `EMBED_MODEL`, `LLM_MODEL`, `COLLECTION_NAME`, `TOP_K`, `MEMORY_LIMIT`
- `CHUNK_MAX_CHARS=2400`, `CHUNK_OVERLAP=400`
- All filesystem paths derived from `BASE_DIR = Path(__file__).parent.parent`

**Runtime config file:**
- `config.json` (gitignored) — admin phone list + `admin_timeout_minutes`
- `config.example.json` — committed template

## Logging

**Framework:** Python standard `logging` module, custom setup in `app/log.py`

- Logger name: `rag_chatbot`
- File handler: `logs/rag_chatbot.log` (DEBUG level, auto-created)
- Stream handler: stdout (INFO level; JSON if `LOG_FORMAT=json`)
- Format: `%(asctime)s | %(levelname)-7s | %(message)s`

## Build / Data Pipeline

**No build step** — pure Python, no compilation or bundling.

**Data pipeline** (manual, one-time):
```bash
python clean_products.py        # data/Products.csv → data/clean_products.json
python rag_chatbot.py --ingest  # clean_products.json → ChromaDB
```

**Training pipeline:**
```bash
python training/generate_qna.py    # LLM → pending.jsonl
python training/approve_qna_cli.py # CLI review → approved.jsonl → ChromaDB
```

## Platform Requirements

**Development:**
- Linux/macOS (fcntl dependency in `app/history.py`)
- Ollama running locally (`ollama serve`)
- Python 3.12

**Production:**
- Linux (cloud VM) — fcntl is POSIX-only
- Ollama accessible (same host or reachable endpoint)
- Docker optional but supported

---
_Stack analysis: 2026-05-16_
