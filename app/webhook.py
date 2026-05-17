import os
import re
import time
import uuid
from collections import OrderedDict

from flask import Flask, request, jsonify

import json

import app.rag as rag
from app.config import LLM_MODEL, EMBED_MODEL, CHROMA_DIR, PRODUCTS_COLLECTION, QNA_COLLECTION, QNA_OVERRIDE_THRESHOLD
from app.intent import classify_intent, IntentEnum
from app.log import get_logger
from app.router import dispatch, _images_from_hits
from app.planner import plan_response
from app.validators import validate_whatsapp_response, validate_response, AUTHORIZED_BRANDS
from app.validators.hallucination import is_blocked_image_url
from app.history import load_history, save_history
from app.admin import is_admin, needs_admin_handling, handle_admin
from app.startup import run_startup
import app.startup as _startup

log = get_logger()
app = Flask(__name__)

_PHONE_RE = re.compile(r"^\+?[0-9]{7,15}$")

wa2mation_key = "configured" if os.getenv("WA2MATION_API_KEY") else "MISSING"
anthropic_key  = "configured" if os.getenv("ANTHROPIC_API_KEY") else "not set"
flask_debug    = os.getenv("FLASK_DEBUG", "false").lower()
log.info(
    "STARTUP | model=%s embed=%s chroma=%s collection=%s",
    LLM_MODEL, EMBED_MODEL, CHROMA_DIR, PRODUCTS_COLLECTION,
)
log.info(
    "STARTUP | wa2mation=%s anthropic=%s flask_debug=%s",
    wa2mation_key, anthropic_key, flask_debug,
)
run_startup()

# In-memory deduplication: {message_id: timestamp}
# Protects against wa2mation retry duplicates within a 60-second window.
_seen_ids: OrderedDict = OrderedDict()
_DEDUP_TTL = 60  # seconds

_ENTITY_TOKEN_RE = re.compile(r'\b[A-Z][a-z]{2,}(?:[A-Z][a-z]*)?\b')


def _is_duplicate(message_id: str) -> bool:
    """Return True if this message_id was already processed within the TTL window."""
    now = time.monotonic()
    # Evict expired entries
    for mid, ts in list(_seen_ids.items()):
        if now - ts > _DEDUP_TTL:
            del _seen_ids[mid]
        else:
            break
    if message_id in _seen_ids:
        return True
    _seen_ids[message_id] = now
    return False


def _hit_entity_tokens(hits: list[dict]) -> set[str]:
    """Extract capitalized name tokens from ChromaDB hit metadata.

    Adds product and brand names found in the retrieved chunks to the
    allowed-entities set so the hallucination validator never flags terms
    that are grounded in the actual retrieval context.
    """
    tokens: set[str] = set()
    for hit in hits:
        meta = hit.get("metadata", {})
        for field in ("name", "brand", "product_line"):
            value = meta.get(field) or ""
            for token in value.split():
                tokens.add(token)
        # Also capture any capitalized tokens from the document text itself
        doc = hit.get("document", "") or ""
        tokens.update(_ENTITY_TOKEN_RE.findall(doc))
    return tokens


def _resolve_media_image_url(result: dict, hits: list[dict]) -> dict:
    """Ensure media responses carry a real image URL, never a hallucinated placeholder.

    When message_type is 'media' and the LLM-supplied image_url is absent or
    matches a blocked placeholder domain (e.g. example.com), replace it with
    the first real image URL found in the ChromaDB hit metadata.  If no real
    image is available either, downgrade message_type to 'text' so the router
    never forwards a fake URL to WhatsApp.

    Returns a (possibly mutated) copy of result — original dict is not modified.
    """
    if result.get("message_type") != "media":
        return result

    current_url = result.get("image_url")
    if current_url and not is_blocked_image_url(current_url):
        # Already a valid URL — nothing to do.
        return result

    # LLM gave no URL or a blocked placeholder — try hits metadata.
    real_urls = _images_from_hits(hits)
    if real_urls:
        fixed = dict(result)
        fixed["image_url"] = real_urls[0]
        log.info(
            "MEDIA_FIX | replaced blocked/missing image_url=%r with hit url=%r",
            current_url, real_urls[0],
        )
        return fixed

    # No real image available — downgrade to text so router doesn't send a broken URL.
    log.warning(
        "MEDIA_FIX | no real image URL available for media response — downgrading to text. "
        "blocked_url=%r",
        current_url,
    )
    fallback = dict(result)
    fallback["message_type"] = "text"
    fallback["image_url"]    = None
    return fallback


@app.route("/webhook", methods=["POST"])
def webhook():
    req = uuid.uuid4().hex[:8]
    try:
        data = request.get_json(silent=True) or {}
        log.debug("req=%s | incoming: %s", req, data)

        if not _startup._ready:
            log.debug("req=%s | startup not complete — returning 503", req)
            return jsonify({"status": "starting"}), 503

        user_message = ((data.get("message") or {}).get("body") or "").strip()
        phone        = ((data.get("contact") or {}).get("phone_number") or "")
        message_id   = ((data.get("message") or {}).get("whatsapp_message_id") or "")

        if not user_message or not phone:
            return jsonify({"status": "ignored"})

        if not _PHONE_RE.match(phone):
            log.debug("req=%s | invalid phone format '%s' — ignored", req, phone)
            return jsonify({"status": "ignored"})

        if message_id and _is_duplicate(message_id):
            log.debug("req=%s | duplicate message_id=%s — ignored", req, message_id)
            return jsonify({"status": "ignored"})

        log.info("req=%s | phone=%s msg=%.60r", req, phone, user_message)
        collection = rag.get_collection(PRODUCTS_COLLECTION)

        # Admin commands bypass the RAG pipeline entirely
        if needs_admin_handling(phone, user_message):
            log.info("req=%s | admin handler", req)
            handle_admin(phone, user_message, collection)
            return jsonify({"status": "success"})

        history = load_history(phone)

        # Layer 1 classification (no embedding needed)
        clf = classify_intent(text=user_message, embedding=None, history=history)
        query_embedding = None

        if clf["layer"] != "rule":
            # Layer 1 didn't fire — compute embedding once for Layer 2 + QnA override
            query_embedding = rag.get_embedding(user_message)
            clf = classify_intent(text=user_message, embedding=query_embedding, history=history)

        log.info(
            "req=%s | INTENT_DEBUG query=%r rule_intent=%s layer=%s confidence=%.3f",
            req, user_message[:80], clf["intent"].value, clf["layer"], clf["confidence"],
        )

        # Intent-conditional routing
        if clf["intent"] == IntentEnum.GREETING:
            result = {
                "message_type": "text",
                "content": "Welcome to Numobel! We carry Rubio Monocoat, Nuacoustics, Nutoy, Nupanel, and Nuwork. How can I help?",
                "buttons": None, "image_url": None, "prompt_tokens": 0, "completion_tokens": 0,
            }
            hits = []

        elif clf["intent"] == IntentEnum.CHITCHAT:
            result = {
                "message_type": "text",
                "content": "I'm here to help with Numobel products! Ask me about Rubio Monocoat, Nuacoustics, Nutoy, Nupanel, or Nuwork.",
                "buttons": None, "image_url": None, "prompt_tokens": 0, "completion_tokens": 0,
            }
            hits = []

        elif clf["intent"] == IntentEnum.OUT_OF_SCOPE:
            result = {
                "message_type": "text",
                "content": "I can only help with product questions. For orders and delivery, please contact Numobel support directly.",
                "buttons": None, "image_url": None, "prompt_tokens": 0, "completion_tokens": 0,
            }
            hits = []

        elif clf["intent"] == IntentEnum.BRAND_DISCOVERY:
            result = {
                "message_type": "text",
                "content": "We carry 5 brands: Rubio Monocoat, Nuacoustics, Nutoy, Nupanel, and Nuwork. Which brand would you like to know more about?",
                "buttons": None, "image_url": None, "prompt_tokens": 0, "completion_tokens": 0,
            }
            hits = []  # Phase 3 will replace with SQLite-driven interactive message

        else:
            # BRAND_DEEP_DIVE, PRODUCT_LINE_QUERY, SPECIFIC_PRODUCT, GENERAL_QNA
            # — full RAG with QnA override.
            # NOTE: BRAND_DEEP_DIVE intentionally falls through here so brand queries
            # ("Nuacoustics", "tell me about nutoy") get real product information from
            # ChromaDB instead of the Phase 3 stub response that caused the fallback loop.
            hits = []
            result = None

            # QnA override check (only when embedding is available)
            if query_embedding is not None:
                qna_col = rag.get_collection(QNA_COLLECTION)
                qna_results = qna_col.query(
                    query_embeddings=[query_embedding],
                    n_results=1,
                    include=["documents", "metadatas", "distances"],
                )
                if qna_results["distances"][0] and qna_results["distances"][0][0] < QNA_OVERRIDE_THRESHOLD:
                    stored = qna_results["metadatas"][0][0]
                    try:
                        buttons = json.loads(stored.get("buttons", "[]")) or None
                    except (json.JSONDecodeError, TypeError):
                        buttons = None
                    result = {
                        "message_type": stored.get("message_type", "text"),
                        "content": qna_results["documents"][0][0],
                        "buttons": buttons,
                        "image_url": stored.get("image_url") or None,
                        "prompt_tokens": 0,
                        "completion_tokens": 0,
                    }
                    log.info("req=%s | INTENT_DEBUG qna_override=true distance=%.4f", req,
                             qna_results["distances"][0][0])

            if result is None:
                # Full RAG pipeline
                search_query = rag.rewrite_query(user_message, history)
                hits = rag.retrieve(collection, search_query)
                log.info(
                    "req=%s | INTENT_DEBUG qna_override=false search_query=%r hits=%d",
                    req, search_query[:80], len(hits),
                )

                # Compute data shape for planner
                image_available = any(
                    h.get("metadata", {}).get("image_url") or h.get("metadata", {}).get("images")
                    for h in hits
                )
                product_count = len(hits)
                button_count  = min(product_count, 5)
                data_shape = {
                    "button_count":    button_count,
                    "image_available": image_available,
                    "product_count":   product_count,
                }

                # 1. Deterministic format planning
                planned_type = plan_response(clf["intent"], data_shape)

                # 2. LLM generation with planned message type
                result = rag.generate_answer(search_query, hits, history, message_type=planned_type)

                # 3. WhatsApp constraint validation
                result = validate_whatsapp_response(result)

                # 4. Resolve media image URL — strip hallucinated placeholder URLs and
                #    substitute a real image from hits metadata. Downgrades to text when
                #    no real image is available so a fake URL never reaches WhatsApp.
                result = _resolve_media_image_url(result, hits)

                # 5. Hallucination validation
                # allowed_entities supplements AUTHORIZED_BRANDS (already populated at
                # startup with all brand + product names). The hit-derived tokens here
                # provide a per-request safety net for any product names the LLM used
                # that are grounded in the retrieved chunks but not yet in the global set
                # (e.g. if the catalogue was updated after startup).
                allowed_entities = _hit_entity_tokens(hits)
                vr = validate_response(
                    content=result["content"],
                    user_query=user_message,
                    allowed_entities=allowed_entities,
                    response_plan={"message_type": result["message_type"]},
                    context_data={"history": history, "hits": hits},
                )
                if not vr.valid:
                    log.warning(
                        "req=%s | HALLUCINATION | severity=%s violations=%s strategy=%s",
                        req, vr.severity, vr.violations, vr.recovery_strategy,
                    )
                    result["content"] = vr.fallback_content

        log.info(
            "req=%s | response type=%s tokens=%d+%d",
            req, result["message_type"],
            result.get("prompt_tokens", 0), result.get("completion_tokens", 0),
        )
        dispatch(phone, result, hits)
        history.append({"role": "user", "content": user_message, "intent": clf["intent"].value})
        history.append({"role": "assistant", "content": result["content"]})
        save_history(phone, history)

        return jsonify({"status": "success"})

    except Exception:
        log.exception("req=%s | unhandled error", req)
        return jsonify({"status": "error"})


@app.route("/health", methods=["GET"])
def health():
    """Readiness check for load balancers and operator smoke tests."""
    status = {
        "ready": _startup._ready,
        "sqlite": False,
        "ollama": False,
        "chromadb": _startup._ready,
    }
    http_code = 200 if _startup._ready else 503

    try:
        from app.db import get_db

        get_db().execute("SELECT 1")
        status["sqlite"] = True
    except Exception:
        http_code = 503

    try:
        import ollama

        ollama.list()
        status["ollama"] = True
    except Exception:
        http_code = 503

    return jsonify(status), http_code


if __name__ == "__main__":
    debug = os.getenv("FLASK_DEBUG", "false").lower() == "true"
    app.run(host="0.0.0.0", port=5000, debug=debug)
