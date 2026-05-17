"""
Thin wrapper — re-exports the full app.rag public API.

Preserves backward compatibility for:
  - `python rag_chatbot.py`           (CLI chat)
  - `python rag_chatbot.py --ingest`  (force rebuild)
  - `import rag_chatbot as rag`       (webhook, admin, training scripts)
"""

from app.config import (
    DATA_FILE, CHROMA_DIR, EMBED_MODEL, LLM_MODEL,
    PRODUCTS_COLLECTION, QNA_COLLECTION,
    CHUNK_MAX_CHARS, CHUNK_OVERLAP, TOP_K, MEMORY_LIMIT, REWRITE_QUERY,
)
from app.rag import (
    get_collection,
    ingest_data,
    product_to_text,
    chunk_text,
    stable_id,
    get_embedding,
    rewrite_query,
    retrieve,
    generate_answer,
    ingest_qna_pair,
    _keyword_fallback,
    SYSTEM_PROMPT,
    main,
)

__all__ = [
    "DATA_FILE", "CHROMA_DIR", "EMBED_MODEL", "LLM_MODEL",
    "PRODUCTS_COLLECTION", "QNA_COLLECTION",
    "CHUNK_MAX_CHARS", "CHUNK_OVERLAP", "TOP_K", "MEMORY_LIMIT", "REWRITE_QUERY",
    "get_collection", "ingest_data", "product_to_text", "chunk_text", "stable_id",
    "get_embedding", "rewrite_query", "retrieve", "generate_answer",
    "ingest_qna_pair", "_keyword_fallback", "SYSTEM_PROMPT", "main",
]

if __name__ == '__main__':
    main()
