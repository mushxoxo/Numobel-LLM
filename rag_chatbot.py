"""
RAG Chatbot — Numobel Product Assistant
========================================
A domain-specific chatbot that answers user queries using only
company product data, powered by:
  • Embeddings : mxbai-embed-large (Ollama)
  • Generation : llama3.2          (Ollama)
  • Vector DB  : ChromaDB (local persistent)

Usage:
    python rag_chatbot.py              # interactive chat (auto-ingests if DB empty)
    python rag_chatbot.py --ingest     # force re-ingest data
"""

import json
import os
import sys
import logging
import textwrap
import hashlib
from datetime import datetime

import chromadb
import ollama


# ─── Configuration ────────────────────────────────────────────────────────────
BASE_DIR        = os.path.dirname(os.path.abspath(__file__))
DATA_FILE       = os.path.join(BASE_DIR, 'rag_products.json')
CHROMA_DIR      = os.path.join(BASE_DIR, 'chroma_db')
LOG_FILE        = os.path.join(BASE_DIR, 'rag_chatbot.log')

EMBED_MODEL     = 'mxbai-embed-large'
LLM_MODEL       = 'llama3.2'
COLLECTION_NAME = 'numobel_products'

CHUNK_MAX_CHARS = 2400          # ~600 tokens (rough 4 chars/token)
CHUNK_OVERLAP   = 400           # ~100 tokens overlap
TOP_K           = 5             # retrieve top-k chunks


# ─── Logging Setup ────────────────────────────────────────────────────────────
def setup_logging():
    """Configure dual logging: console (info) + file (all I/O)."""
    logger = logging.getLogger('rag_chatbot')
    logger.setLevel(logging.DEBUG)

    # File handler — stores everything
    fh = logging.FileHandler(LOG_FILE, encoding='utf-8')
    fh.setLevel(logging.DEBUG)
    fh.setFormatter(logging.Formatter(
        '%(asctime)s | %(levelname)-7s | %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S'
    ))

    # Console handler — minimal
    ch = logging.StreamHandler()
    ch.setLevel(logging.INFO)
    ch.setFormatter(logging.Formatter('%(message)s'))

    logger.addHandler(fh)
    logger.addHandler(ch)
    return logger

log = setup_logging()


# ─── Text Utilities ───────────────────────────────────────────────────────────

def product_to_text(product: dict) -> str:
    """
    Convert a product record into a clean text passage for embedding.
    Includes: Name, Brand, Product Line, Description, Specifications,
              Price, Colors, Sizes, SEO keywords.
    """
    parts = []

    name = product.get('name') or 'Unknown Product'
    brand = product.get('brand') or ''
    product_line = product.get('product_line') or ''

    parts.append(f"Product: {name}")
    if brand:
        parts.append(f"Brand: {brand}")
    if product_line:
        parts.append(f"Product Line: {product_line}")

    desc = product.get('description')
    if desc:
        parts.append(f"Description: {desc}")

    specs = (product.get('attributes') or {}).get('specifications')
    if specs:
        parts.append(f"Specifications: {specs}")

    # Price
    price_info = product.get('price') or {}
    original = price_info.get('original')
    discounted = price_info.get('discounted')
    if original is not None:
        if discounted is not None and discounted != original:
            parts.append(f"Price: ₹{original} (Discounted: ₹{discounted})")
        else:
            parts.append(f"Price: ₹{original}")

    # Colors
    colors = (product.get('attributes') or {}).get('colors', [])
    if colors:
        parts.append(f"Available Colors: {', '.join(str(c) for c in colors)}")

    # Sizes
    sizes = (product.get('attributes') or {}).get('size', [])
    if sizes:
        parts.append(f"Available Sizes: {', '.join(str(s) for s in sizes)}")

    # Weight
    weight = (product.get('attributes') or {}).get('weight')
    if weight:
        parts.append(f"Weight: {weight}")

    # SEO keywords
    keywords = (product.get('seo') or {}).get('keywords', [])
    if keywords:
        parts.append(f"Keywords: {', '.join(keywords)}")

    return '\n'.join(parts)


def chunk_text(text: str, chunk_max: int = CHUNK_MAX_CHARS,
               overlap: int = CHUNK_OVERLAP) -> list[str]:
    """
    Split text into overlapping chunks.
    Tries to break at sentence boundaries ('.') when possible.
    Returns a list of chunk strings.
    """
    if len(text) <= chunk_max:
        return [text]

    chunks = []
    start = 0
    while start < len(text):
        end = start + chunk_max

        # Try to break at a sentence boundary
        if end < len(text):
            # Look backward for a period followed by a space or newline
            boundary = text.rfind('. ', start, end)
            if boundary == -1:
                boundary = text.rfind('\n', start, end)
            if boundary != -1 and boundary > start:
                end = boundary + 1  # include the period

        chunk = text[start:end].strip()
        if chunk:
            chunks.append(chunk)

        # Move forward, minus overlap
        start = end - overlap if end < len(text) else len(text)

    return chunks


def stable_id(text: str, idx: int) -> str:
    """Generate a deterministic document ID for idempotent upserts."""
    h = hashlib.md5(text.encode('utf-8')).hexdigest()[:12]
    return f"chunk_{h}_{idx}"


# ─── Embedding Helper ────────────────────────────────────────────────────────

def get_embedding(text: str) -> list[float]:
    """Get embedding vector from Ollama mxbai-embed-large."""
    response = ollama.embed(model=EMBED_MODEL, input=text)
    return response['embeddings'][0]


# ─── Data Ingestion ──────────────────────────────────────────────────────────

def ingest_data(collection) -> int:
    """
    Load rag_products.json, convert to text, chunk, embed, and store
    in ChromaDB.  Returns total chunks ingested.
    """
    log.info("Loading product data from %s ...", DATA_FILE)
    with open(DATA_FILE, encoding='utf-8') as f:
        products = json.load(f)
    log.info("Loaded %d products.", len(products))

    all_ids        = []
    all_documents  = []
    all_embeddings = []
    all_metadatas  = []

    for i, product in enumerate(products):
        text = product_to_text(product)
        chunks = chunk_text(text)

        brand        = product.get('brand') or 'Unknown'
        product_line = product.get('product_line') or ''
        price_orig   = (product.get('price') or {}).get('original')
        name         = product.get('name') or 'Unknown'

        for j, chunk in enumerate(chunks):
            doc_id = stable_id(chunk, j)
            all_ids.append(doc_id)
            all_documents.append(chunk)
            all_metadatas.append({
                'brand':        brand,
                'product_line': product_line,
                'price':        float(price_orig) if price_orig else 0.0,
                'product_name': name,
                'chunk_index':  j,
            })

        if (i + 1) % 20 == 0 or (i + 1) == len(products):
            log.info("  Processed %d / %d products ...", i + 1, len(products))

    # Batch embed
    log.info("Generating embeddings for %d chunks (this may take a few minutes) ...", len(all_documents))

    BATCH_SIZE = 32
    for batch_start in range(0, len(all_documents), BATCH_SIZE):
        batch_end = min(batch_start + BATCH_SIZE, len(all_documents))
        batch_texts = all_documents[batch_start:batch_end]

        response = ollama.embed(model=EMBED_MODEL, input=batch_texts)
        all_embeddings.extend(response['embeddings'])

        log.info("  Embedded %d / %d chunks ...", batch_end, len(all_documents))

    # Upsert into ChromaDB
    log.info("Upserting %d chunks into ChromaDB ...", len(all_ids))
    collection.upsert(
        ids=all_ids,
        documents=all_documents,
        embeddings=all_embeddings,
        metadatas=all_metadatas,
    )

    log.info("✅ Ingestion complete: %d chunks stored.", len(all_ids))
    return len(all_ids)


# ─── Retrieval ────────────────────────────────────────────────────────────────

def retrieve(collection, query: str, top_k: int = TOP_K) -> list[dict]:
    """
    Embed user query and retrieve top-k relevant chunks from ChromaDB.
    Returns list of dicts with 'text', 'metadata', 'distance'.
    """
    query_embedding = get_embedding(query)

    results = collection.query(
        query_embeddings=[query_embedding],
        n_results=top_k,
        include=['documents', 'metadatas', 'distances'],
    )

    hits = []
    for doc, meta, dist in zip(
        results['documents'][0],
        results['metadatas'][0],
        results['distances'][0],
    ):
        hits.append({
            'text':     doc,
            'metadata': meta,
            'distance': dist,
        })

    return hits


# ─── Generation ───────────────────────────────────────────────────────────────

SYSTEM_PROMPT = textwrap.dedent("""\
    You are a Numobel company assistant. Only answer using the provided context.
    If the answer is not found in the context, say "I don't have that information."
    Prefer structured responses using bullet points or sections when appropriate.
    Always mention the product name and brand when referring to a product.
    If prices are mentioned, use the Indian Rupee symbol (₹).
""")


def generate_answer(query: str, context_chunks: list[dict]) -> str:
    """
    Build the RAG prompt and generate an answer via llama3.2.
    """
    # Assemble context block
    context_parts = []
    for i, chunk in enumerate(context_chunks, 1):
        meta = chunk['metadata']
        header = f"[Source {i}: {meta.get('product_name', '?')} | {meta.get('brand', '?')}]"
        context_parts.append(f"{header}\n{chunk['text']}")

    context_block = '\n\n---\n\n'.join(context_parts)

    user_message = f"CONTEXT:\n{context_block}\n\nUSER QUERY:\n{query}"

    response = ollama.chat(
        model=LLM_MODEL,
        messages=[
            {'role': 'system', 'content': SYSTEM_PROMPT},
            {'role': 'user',   'content': user_message},
        ],
    )

    return response['message']['content']


# ─── Interactive CLI ──────────────────────────────────────────────────────────

WELCOME_BANNER = """
╔══════════════════════════════════════════════════════════════╗
║              🏭  NUMOBEL PRODUCT ASSISTANT  🏭              ║
║                                                              ║
║   Ask me anything about Numobel products!                    ║
║   Models: mxbai-embed-large + llama3.2 (local via Ollama)    ║
║                                                              ║
║   Type 'quit' or 'exit' to leave.                            ║
║   Type 'stats' for database info.                            ║
╚══════════════════════════════════════════════════════════════╝
"""


def print_stats(collection):
    """Print collection statistics."""
    count = collection.count()
    log.info("── Database Stats ──")
    log.info("  Collection : %s", COLLECTION_NAME)
    log.info("  Chunks     : %d", count)
    log.info("  Embed Model: %s", EMBED_MODEL)
    log.info("  LLM Model  : %s", LLM_MODEL)
    log.info("  Log File   : %s", LOG_FILE)


def interactive_loop(collection):
    """Main interactive Q&A loop."""
    print(WELCOME_BANNER)
    print_stats(collection)
    print()

    while True:
        try:
            query = input("📝 You: ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\n👋 Goodbye!")
            log.info("Session ended by user (EOF/Interrupt).")
            break

        if not query:
            continue

        if query.lower() in ('quit', 'exit', 'q'):
            print("👋 Goodbye!")
            log.info("Session ended by user (quit command).")
            break

        if query.lower() == 'stats':
            print_stats(collection)
            continue

        # Log the user query
        log.debug("USER_INPUT  | %s", query)

        # Retrieve
        log.info("🔍 Searching knowledge base ...")
        hits = retrieve(collection, query)

        if not hits:
            answer = "I don't have that information."
        else:
            # Log retrieved sources
            for i, h in enumerate(hits, 1):
                log.debug(
                    "RETRIEVED_%d | dist=%.4f | %s",
                    i, h['distance'], h['metadata'].get('product_name', '?')
                )

            # Generate
            log.info("🤖 Generating answer ...")
            answer = generate_answer(query, hits)

        # Log the output
        log.debug("LLM_OUTPUT  | %s", answer.replace('\n', '\\n'))

        # Display
        print(f"\n🤖 Assistant:\n{answer}\n")
        print("─" * 60)


# ─── Entry Point ──────────────────────────────────────────────────────────────

def main():
    force_ingest = '--ingest' in sys.argv

    # Initialize ChromaDB
    client = chromadb.PersistentClient(path=CHROMA_DIR)
    collection = client.get_or_create_collection(
        name=COLLECTION_NAME,
        metadata={"hnsw:space": "cosine"},   # cosine similarity
    )

    current_count = collection.count()
    log.info("ChromaDB initialized at %s  (%d existing chunks)", CHROMA_DIR, current_count)

    # Ingest if empty or forced
    if current_count == 0 or force_ingest:
        if force_ingest and current_count > 0:
            log.info("Force re-ingest requested. Deleting existing collection ...")
            client.delete_collection(COLLECTION_NAME)
            collection = client.get_or_create_collection(
                name=COLLECTION_NAME,
                metadata={"hnsw:space": "cosine"},
            )
        ingest_data(collection)

    # Enter interactive mode
    interactive_loop(collection)


if __name__ == '__main__':
    main()
