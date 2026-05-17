"""
Central configuration for the Numobel RAG chatbot.

Calls load_dotenv() once at import time. Exports all shared constants and
filesystem paths. Does NOT validate credentials — those are checked lazily
by the modules that use them (messaging client, refinement engine).
"""

import os
from pathlib import Path

from dotenv import load_dotenv

load_dotenv()

# ─── Filesystem roots ─────────────────────────────────────────────────────────

BASE_DIR  = Path(__file__).parent.parent
DATA_FILE = BASE_DIR / "data" / "clean_products.json"
CHROMA_DIR = BASE_DIR / "chroma_db"
LOG_DIR    = BASE_DIR / "logs"

# Training data
PENDING_PATH  = BASE_DIR / "training" / "qna_pairs" / "pending.jsonl"
APPROVED_PATH = BASE_DIR / "training" / "qna_pairs" / "approved.jsonl"

# ─── RAG constants ────────────────────────────────────────────────────────────

EMBED_MODEL     = "mxbai-embed-large"
LLM_MODEL       = "llama3.2"
CHUNK_MAX_CHARS = 2400   # ~600 tokens
CHUNK_OVERLAP   = 400    # ~100 tokens overlap
TOP_K           = 5      # chunks retrieved per query
MEMORY_LIMIT    = 5      # conversation turns kept in context
REWRITE_QUERY   = True   # set False to disable query reformulation

# ─── Database ─────────────────────────────────────────────────────────────────

SQLITE_PATH = BASE_DIR / "numobel.db"

# ─── ChromaDB collections ─────────────────────────────────────────────────────

PRODUCTS_COLLECTION = "numobel_products"
QNA_COLLECTION      = "numobel_approved_qna"

# ─── WhatsApp templates ───────────────────────────────────────────────────────

CAROUSEL_TEMPLATE = "numobel_catalogue_4"
