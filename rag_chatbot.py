"""
RAG Chatbot Core
========================================
Core ingestion, retrieval, and generation logic.
Imported by web_app.py and usable standalone via CLI.
"""

import json
import os
import sys
import logging
import textwrap
import hashlib

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

CHUNK_MAX_CHARS = 2400          # ~600 tokens
CHUNK_OVERLAP   = 400           # ~100 tokens overlap
TOP_K           = 5             # retrieve top-k chunks

# ====== USER CONFIGURATION ======
MEMORY_LIMIT    = 5             # Max previous conversation turns to remember
REWRITE_QUERY   = True          # Set False to disable query reformulation
USE_NATIVE_IMAGES = True        # True = Streamlit columns layout; False = Pillow collage
# ================================


# ─── Logging Setup ────────────────────────────────────────────────────────────
def setup_logging():
    logger = logging.getLogger('rag_chatbot')
    if logger.handlers:
        return logger

    logger.setLevel(logging.DEBUG)

    fh = logging.FileHandler(LOG_FILE, encoding='utf-8')
    fh.setLevel(logging.DEBUG)
    fh.setFormatter(logging.Formatter(
        '%(asctime)s | %(levelname)-7s | %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S'
    ))

    ch = logging.StreamHandler()
    ch.setLevel(logging.INFO)
    ch.setFormatter(logging.Formatter('%(message)s'))

    logger.addHandler(fh)
    logger.addHandler(ch)
    return logger

log = setup_logging()


def get_collection() -> chromadb.Collection:
    """Returns the ChromaDB collection, creating it if needed."""
    client = chromadb.PersistentClient(path=CHROMA_DIR)
    return client.get_or_create_collection(
        name=COLLECTION_NAME,
        metadata={"hnsw:space": "cosine"},
    )


# ─── Text Utilities ───────────────────────────────────────────────────────────

def product_to_text(product: dict) -> str:
    """Convert a product record into a single text passage for embedding."""
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

    price_info = product.get('price') or {}
    original = price_info.get('original')
    discounted = price_info.get('discounted')
    if original is not None:
        if discounted is not None and discounted != original:
            parts.append(f"Price: ₹{original} (Discounted: ₹{discounted})")
        else:
            parts.append(f"Price: ₹{original}")

    colors = (product.get('attributes') or {}).get('colors', [])
    if colors:
        parts.append(f"Available Colors: {', '.join(str(c) for c in colors)}")

    sizes = (product.get('attributes') or {}).get('size', [])
    if sizes:
        parts.append(f"Available Sizes: {', '.join(str(s) for s in sizes)}")

    weight = (product.get('attributes') or {}).get('weight')
    if weight:
        parts.append(f"Weight: {weight}")

    keywords = (product.get('seo') or {}).get('keywords', [])
    if keywords:
        parts.append(f"Keywords: {', '.join(keywords)}")

    return '\n'.join(parts)


def chunk_text(text: str, chunk_max: int = CHUNK_MAX_CHARS,
               overlap: int = CHUNK_OVERLAP) -> list[str]:
    """Split text into overlapping chunks at sentence boundaries."""
    if len(text) <= chunk_max:
        return [text]

    chunks = []
    start = 0
    while start < len(text):
        end = start + chunk_max
        if end < len(text):
            boundary = text.rfind('. ', start, end)
            if boundary == -1:
                boundary = text.rfind('\n', start, end)
            if boundary != -1 and boundary > start:
                end = boundary + 1
        chunk = text[start:end].strip()
        if chunk:
            chunks.append(chunk)
        start = end - overlap if end < len(text) else len(text)

    return chunks


def stable_id(text: str, idx: int) -> str:
    """Deterministic document ID for idempotent upserts."""
    h = hashlib.md5(text.encode('utf-8')).hexdigest()[:12]
    return f"chunk_{h}_{idx}"


# ─── Embedding Helper ────────────────────────────────────────────────────────

def get_embedding(text: str) -> list[float]:
    """Get embedding vector from Ollama mxbai-embed-large."""
    response = ollama.embed(model=EMBED_MODEL, input=text)
    return response['embeddings'][0]


# ─── Data Ingestion ──────────────────────────────────────────────────────────

def ingest_data(collection) -> int:
    """Load products, chunk, embed, and store in ChromaDB with image metadata."""
    log.info("Loading product data from %s ...", DATA_FILE)
    with open(DATA_FILE, encoding='utf-8') as f:
        products = json.load(f)
    log.info("Loaded %d products.", len(products))

    all_ids, all_documents, all_embeddings, all_metadatas = [], [], [], []

    for i, product in enumerate(products):
        text = product_to_text(product)
        chunks = chunk_text(text)

        brand        = product.get('brand') or 'Unknown'
        product_line = product.get('product_line') or ''
        price_orig   = (product.get('price') or {}).get('original')
        name         = product.get('name') or 'Unknown'

        # Extract image URLs for metadata (stored as pipe-separated string)
        images_list  = (product.get('media') or {}).get('images', [])
        images_str   = '|'.join(images_list) if images_list else ''
        product_link = (product.get('metadata') or {}).get('product_link') or ''

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
                'images':       images_str,
                'product_link': product_link,
            })

    # Batch embed
    log.info("Generating embeddings for %d chunks ...", len(all_documents))
    BATCH_SIZE = 32
    for batch_start in range(0, len(all_documents), BATCH_SIZE):
        batch_end = min(batch_start + BATCH_SIZE, len(all_documents))
        batch_texts = all_documents[batch_start:batch_end]
        response = ollama.embed(model=EMBED_MODEL, input=batch_texts)
        all_embeddings.extend(response['embeddings'])
        log.info("  Embedded %d / %d chunks ...", batch_end, len(all_documents))

    # Upsert
    log.info("Upserting %d chunks into ChromaDB ...", len(all_ids))
    collection.upsert(
        ids=all_ids,
        documents=all_documents,
        embeddings=all_embeddings,
        metadatas=all_metadatas,
    )

    log.info("✅ Ingestion complete: %d chunks stored.", len(all_ids))
    return len(all_ids)


# ─── Query Reformulation ─────────────────────────────────────────────────────

def format_history_for_rewrite(history: list[dict]) -> str:
    """Format recent messages for the rewrite prompt."""
    parts = []
    for msg in history[-MEMORY_LIMIT:]:
        role = "User" if msg['role'] == 'user' else "Assistant"
        parts.append(f"{role}: {msg['content']}")
    return '\n'.join(parts)


def rewrite_query(query: str, history: list[dict]) -> str:
    """Rewrite a follow-up query into a standalone question using chat history."""
    if not history or not REWRITE_QUERY:
        return query

    history_str = format_history_for_rewrite(history)

    prompt = textwrap.dedent(f"""\
        Given the following conversation history and a follow up question,
        rephrase the follow up question to be a standalone question.

        Chat History:
        {history_str}

        Follow Up Input: {query}

        Standalone question (ONLY print the question, no introductory text):""")

    response = ollama.generate(model=LLM_MODEL, prompt=prompt)
    standalone = response['response'].strip()
    log.debug("QUERY_REWRITTEN | Original: '%s' -> Standalone: '%s'", query, standalone)
    return standalone


# ─── Retrieval ────────────────────────────────────────────────────────────────

def retrieve(collection, query: str, top_k: int = TOP_K) -> list[dict]:
    """Embed user query and retrieve top-k relevant chunks."""
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
        hits.append({'text': doc, 'metadata': meta, 'distance': dist})
    return hits


def collect_images_from_hits(hits: list[dict]) -> list[str]:
    """Extract unique image URLs from retrieval hits."""
    seen = set()
    images = []
    for hit in hits:
        img_str = hit['metadata'].get('images', '')
        if img_str:
            for url in img_str.split('|'):
                url = url.strip()
                if url and url not in seen:
                    seen.add(url)
                    images.append(url)
    return images


# ─── Generation ───────────────────────────────────────────────────────────────

SYSTEM_PROMPT = textwrap.dedent("""\
    You are a Numobel company product assistant. Only answer using the provided context.
    If the answer is not found in the context, say "I don't have that information."

    FORMATTING RULES (you MUST follow these):
    1. Always use **bold** for product names and brands.
    2. Use bullet points (•) for listing features, colors, or sizes.
    3. When presenting specifications or comparing multiple products, use a Markdown TABLE.
       Example table format:
       | Property | Value |
       |----------|-------|
       | Brand    | Rubio Monocoat |
    4. Use headings (### ) to separate sections in longer answers.
    5. Always include the price with the ₹ symbol.
    6. Keep answers concise but complete.
    7. When mentioning a product, state its brand and product line if available.
""")


def generate_answer(query: str, context_chunks: list[dict],
                    history: list[dict] = None) -> dict:
    """
    Build the RAG prompt and generate an answer via llama3.2.
    Returns a dict with 'content', 'prompt_tokens', 'completion_tokens'.
    """
    # Assemble context block
    context_parts = []
    for i, chunk in enumerate(context_chunks, 1):
        meta = chunk['metadata']
        header = f"[Source {i}: {meta.get('product_name', '?')} | {meta.get('brand', '?')}]"
        context_parts.append(f"{header}\n{chunk['text']}")

    context_block = '\n\n---\n\n'.join(context_parts)
    current_prompt = f"CONTEXT:\n{context_block}\n\nUSER QUERY:\n{query}"

    messages = [{'role': 'system', 'content': SYSTEM_PROMPT}]

    # Append memory history
    if history:
        for msg in history[-MEMORY_LIMIT:]:
            messages.append({'role': msg['role'], 'content': msg['content']})

    messages.append({'role': 'user', 'content': current_prompt})

    response = ollama.chat(model=LLM_MODEL, messages=messages)

    content          = response['message']['content']
    prompt_tokens     = response.get('prompt_eval_count', 0)
    completion_tokens = response.get('eval_count', 0)

    log.debug("TOKENS | prompt=%d completion=%d", prompt_tokens, completion_tokens)

    return {
        'content':           content,
        'prompt_tokens':     prompt_tokens,
        'completion_tokens': completion_tokens,
    }


# ─── CLI Run ──────────────────────────────────────────────────────────────────

def main():
    force_ingest = '--ingest' in sys.argv
    collection = get_collection()

    current_count = collection.count()
    if current_count == 0 or force_ingest:
        if force_ingest and current_count > 0:
            client = chromadb.PersistentClient(path=CHROMA_DIR)
            client.delete_collection(COLLECTION_NAME)
            collection = client.get_or_create_collection(
                name=COLLECTION_NAME,
                metadata={"hnsw:space": "cosine"},
            )
        ingest_data(collection)

    print("DB Ready. CLI Chat is active. Type 'quit' to exit.")
    history = []
    while True:
        try:
            q = input("📝 You: ").strip()
            if q.lower() in ('quit', 'exit'):
                break
            if not q:
                continue

            search_query = rewrite_query(q, history)
            hits = retrieve(collection, search_query)
            result = generate_answer(q, hits, history)

            print(f"\n🤖: {result['content']}\n")
            print(f"⚡ Tokens: {result['prompt_tokens']} prompt / {result['completion_tokens']} completion\n")

            history.append({"role": "user", "content": q})
            history.append({"role": "assistant", "content": result['content']})

        except (EOFError, KeyboardInterrupt):
            break


if __name__ == '__main__':
    main()
