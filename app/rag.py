"""
RAG core — embedding, retrieval, and generation logic.

Imported by app/webhook.py (server) and rag_chatbot.py (CLI wrapper).
"""

import json
import os
import re
import sys
import textwrap
import hashlib

import chromadb
import ollama

from app.config import (
    DATA_FILE, CHROMA_DIR, EMBED_MODEL, LLM_MODEL, COLLECTION_NAME,
    CHUNK_MAX_CHARS, CHUNK_OVERLAP, TOP_K, MEMORY_LIMIT, REWRITE_QUERY,
)
from app.log import get_logger

log = get_logger()


# ─── ChromaDB ─────────────────────────────────────────────────────────────────

def get_collection() -> chromadb.Collection:
    """Return the ChromaDB collection, creating it if needed."""
    client = chromadb.PersistentClient(path=str(CHROMA_DIR))
    return client.get_or_create_collection(
        name=COLLECTION_NAME,
        metadata={"hnsw:space": "cosine"},
    )


# ─── Text utilities ───────────────────────────────────────────────────────────

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


# ─── Embedding ────────────────────────────────────────────────────────────────

def get_embedding(text: str) -> list[float]:
    """Get embedding vector from Ollama mxbai-embed-large."""
    response = ollama.embed(model=EMBED_MODEL, input=text)
    return response['embeddings'][0]


# ─── Ingestion ────────────────────────────────────────────────────────────────

def ingest_data(collection) -> int:
    """
    Load products, chunk, embed, and upsert into ChromaDB.

    Raises on embedding failure — callers must handle startup ingest errors.
    """
    log.info("INGEST | loading product data from %s", DATA_FILE)
    with open(DATA_FILE, encoding='utf-8') as f:
        products = json.load(f)
    log.info("INGEST | loaded %d products", len(products))

    all_ids, all_documents, all_metadatas = [], [], []

    for product in products:
        text = product_to_text(product)
        chunks = chunk_text(text)

        brand        = product.get('brand') or 'Unknown'
        product_line = product.get('product_line') or ''
        price_orig   = (product.get('price') or {}).get('original')
        name         = product.get('name') or 'Unknown'
        images_list  = (product.get('media') or {}).get('images', [])
        images_str   = '|'.join(images_list) if images_list else ''
        product_link = (product.get('metadata') or {}).get('product_link') or ''

        for j, chunk in enumerate(chunks):
            all_ids.append(stable_id(chunk, j))
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

    log.info("INGEST | embedding %d chunks ...", len(all_documents))
    all_embeddings = []
    BATCH_SIZE = 32
    try:
        for batch_start in range(0, len(all_documents), BATCH_SIZE):
            batch_end = min(batch_start + BATCH_SIZE, len(all_documents))
            batch_texts = all_documents[batch_start:batch_end]
            response = ollama.embed(model=EMBED_MODEL, input=batch_texts)
            all_embeddings.extend(response['embeddings'])
            log.info("INGEST | embedded %d / %d chunks", batch_end, len(all_documents))
    except Exception:
        log.critical("INGEST | embedding failed — is Ollama running? (ollama serve)")
        raise

    log.info("INGEST | upserting %d chunks into ChromaDB ...", len(all_ids))
    collection.upsert(
        ids=all_ids,
        documents=all_documents,
        embeddings=all_embeddings,
        metadatas=all_metadatas,
    )
    log.info("INGEST | complete — %d chunks stored", len(all_ids))
    return len(all_ids)


# ─── Query reformulation ──────────────────────────────────────────────────────

def rewrite_query(query: str, history: list[dict]) -> str:
    """Rewrite a follow-up query into a standalone question using chat history."""
    if not history or not REWRITE_QUERY:
        return query

    history_str = '\n'.join(
        f"{'User' if m['role'] == 'user' else 'Assistant'}: {m['content']}"
        for m in history[-MEMORY_LIMIT:]
    )
    prompt = textwrap.dedent(f"""\
        Given the following conversation history and a follow up question,
        rephrase the follow up question to be a standalone question.

        Chat History:
        {history_str}

        Follow Up Input: {query}

        Standalone question (ONLY print the question, no introductory text):""")

    response = ollama.generate(model=LLM_MODEL, prompt=prompt)
    standalone = response['response'].strip()
    log.debug("REWRITE | '%s' -> '%s'", query, standalone)
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
    return [
        {'text': doc, 'metadata': meta, 'distance': dist}
        for doc, meta, dist in zip(
            results['documents'][0],
            results['metadatas'][0],
            results['distances'][0],
        )
    ]


# ─── Generation ───────────────────────────────────────────────────────────────

_CAROUSEL_KEYWORDS = {
    'stacker', 'stackers', 'on wheels', 'montessori', 'nutoy',
    'rubio', 'monocoat', 'nuacoustics', 'nupanel', 'nuwork',
    'acoustic', 'panel', 'hardwax',
}

SYSTEM_PROMPT = textwrap.dedent("""\
    You are a Numobel product assistant. Only answer using the provided context.
    If the answer is not in the context, set content to "I don't have that information."

    You MUST respond with ONLY a valid JSON object — no markdown fences, no extra text:
    {
      "message_type": "text" | "interactive" | "media" | "carousel",
      "content": "<your answer here>",
      "buttons": ["<label1>", "<label2>", "<label3>"] or null,
      "image_url": "<url>" or null
    }

    MESSAGE TYPE RULES:
    - "interactive" — use when offering category choices or asking the user to pick an option.
                      Include up to 3 short button labels in "buttons".
    - "carousel"    — use when showcasing 2+ products from the same product line.
    - "media"       — use when the user explicitly asks for an image, photo, or picture.
    - "text"        — use for all other answers (facts, specs, comparisons, prices).

    CONTENT RULES:
    - Use ₹ symbol for all prices.
    - Bold product names and brands using **name**.
    - For greetings, introduce Numobel's 5 brands and offer category buttons.
    - Keep answers concise but complete.
""")


def _keyword_fallback(query: str) -> dict:
    """Determine message_type from keywords when LLM returns invalid JSON."""
    q = query.lower()
    if any(w in q for w in ('hi', 'hello', 'hey', 'what do you have', 'show me', 'list')):
        return {'message_type': 'interactive', 'buttons': ['Wood Finishes', 'Wooden Toys', 'Acoustic Panels']}
    if any(w in q for w in _CAROUSEL_KEYWORDS):
        return {'message_type': 'carousel', 'buttons': None}
    if any(w in q for w in ('image', 'photo', 'picture')):
        return {'message_type': 'media', 'buttons': None}
    return {'message_type': 'text', 'buttons': None}


def generate_answer(query: str, context_chunks: list[dict],
                    history: list[dict] = None) -> dict:
    """
    Build the RAG prompt and generate a structured answer.
    Returns a dict with: message_type, content, buttons, image_url,
    prompt_tokens, completion_tokens.
    """
    context_parts = []
    for i, chunk in enumerate(context_chunks, 1):
        meta = chunk['metadata']
        header = f"[Source {i}: {meta.get('product_name', '?')} | {meta.get('brand', '?')}]"
        body = chunk['text']
        if meta.get('source') == 'approved_training':
            mt = meta.get('message_type', 'text')
            try:
                btns = json.loads(meta.get('buttons', '[]'))
            except (json.JSONDecodeError, ValueError):
                btns = []
            hint = f'message_type="{mt}"'
            if btns:
                hint += f', buttons={json.dumps(btns)}'
            body += f'\n[Approved format: {hint}]'
        context_parts.append(f"{header}\n{body}")

    context_block = '\n\n---\n\n'.join(context_parts)
    current_prompt = f"CONTEXT:\n{context_block}\n\nUSER QUERY:\n{query}"

    messages = [{'role': 'system', 'content': SYSTEM_PROMPT}]
    if history:
        for msg in history[-MEMORY_LIMIT:]:
            messages.append({'role': msg['role'], 'content': msg['content']})
    messages.append({'role': 'user', 'content': current_prompt})

    response = ollama.chat(model=LLM_MODEL, messages=messages)
    raw = response['message']['content']
    prompt_tokens     = response.get('prompt_eval_count', 0)
    completion_tokens = response.get('eval_count', 0)
    log.debug("TOKENS | prompt=%d completion=%d", prompt_tokens, completion_tokens)

    try:
        cleaned = raw.strip()
        if cleaned.startswith('```'):
            cleaned = re.sub(r'^```[a-z]*\n?', '', cleaned).rstrip('`').strip()
        parsed = json.loads(cleaned)
        message_type = parsed.get('message_type', 'text')
        content      = parsed.get('content', '')
        buttons      = parsed.get('buttons') or None
        image_url    = parsed.get('image_url') or None
        log.debug("STRUCTURED | type=%s buttons=%s", message_type, buttons)
    except (json.JSONDecodeError, ValueError):
        log.warning("LLM returned non-JSON — using keyword fallback. Raw: %s", raw[:120])
        fallback     = _keyword_fallback(query)
        message_type = fallback['message_type']
        m = re.search(r'"content"\s*:\s*"([^"]*)"', raw)
        content      = m.group(1) if m else "Sorry, I couldn't process that. Please try again."
        buttons      = fallback['buttons']
        image_url    = None

    # Faithful-replay override: when the top hit is an approved training pair and the LLM
    # agrees on message_type, copy structural fields verbatim from metadata.
    top_hit = context_chunks[0] if context_chunks else None
    if top_hit and top_hit.get('metadata', {}).get('source') == 'approved_training':
        meta = top_hit['metadata']
        approved_mt = meta.get('message_type')
        if approved_mt and message_type == approved_mt:
            if approved_mt == 'interactive':
                try:
                    stored_buttons = json.loads(meta.get('buttons', '[]'))
                except (json.JSONDecodeError, ValueError):
                    stored_buttons = []
                if stored_buttons:
                    buttons = stored_buttons
                    log.debug("REPLAY_FIX | copied buttons from approved metadata")
            elif approved_mt == 'media':
                stored_img = meta.get('image_url') or ''
                if stored_img:
                    image_url = stored_img
                    log.debug("REPLAY_FIX | copied image_url from approved metadata")

    return {
        'message_type':      message_type,
        'content':           content,
        'buttons':           buttons,
        'image_url':         image_url,
        'prompt_tokens':     prompt_tokens,
        'completion_tokens': completion_tokens,
    }


# ─── Training ingestion ───────────────────────────────────────────────────────

def ingest_qna_pair(pair: dict, collection) -> str:
    """
    Upsert a single approved Q&A pair into ChromaDB.
    Returns the document ID.
    """
    text = f"Q: {pair['question']}\nA: {pair['answer']}"
    embedding = get_embedding(text)
    doc_id = stable_id(text, 0)

    collection.upsert(
        ids=[doc_id],
        documents=[text],
        embeddings=[embedding],
        metadatas=[{
            'source':       'approved_training',
            'message_type': pair.get('message_type', 'text'),
            'buttons':      json.dumps(pair.get('buttons') or []),
            'image_url':    pair.get('image_url') or '',
            'product':      pair.get('product', ''),
        }],
    )
    log.info("INGEST_QNA | id=%s product=%s", doc_id, pair.get('product', ''))
    return doc_id


# ─── CLI entry point ──────────────────────────────────────────────────────────

def main():
    force_ingest = '--ingest' in sys.argv
    collection = get_collection()

    current_count = collection.count()
    if current_count == 0 or force_ingest:
        if force_ingest and current_count > 0:
            client = chromadb.PersistentClient(path=str(CHROMA_DIR))
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
            q = input("You: ").strip()
            if q.lower() in ('quit', 'exit'):
                break
            if not q:
                continue

            search_query = rewrite_query(q, history)
            hits = retrieve(collection, search_query)
            result = generate_answer(q, hits, history)

            print(f"\n[{result['message_type']}]: {result['content']}\n")
            if result.get('buttons'):
                print(f"Buttons: {result['buttons']}")
            print(f"Tokens: {result['prompt_tokens']} prompt / {result['completion_tokens']} completion\n")

            history.append({"role": "user", "content": q})
            history.append({"role": "assistant", "content": result['content']})

        except (EOFError, KeyboardInterrupt):
            break
