import logging
import os
import sys
import time
from collections import OrderedDict

# Allow running as `python app/webhook.py` from the project root
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from flask import Flask, request, jsonify
from dotenv import load_dotenv

import rag_chatbot as rag
from app.router import dispatch
from app.history import load_history, save_history
from app.admin import is_admin, needs_admin_handling, handle_admin

load_dotenv()

log = logging.getLogger('rag_chatbot')
app = Flask(__name__)

# Load ChromaDB once at startup; ingest if empty
collection = rag.get_collection()
if collection.count() == 0:
    log.info("ChromaDB empty — running initial ingest...")
    rag.ingest_data(collection)

# In-memory deduplication: {message_id: timestamp}
# Protects against wa2mation retry duplicates within a 60-second window.
_seen_ids: OrderedDict = OrderedDict()
_DEDUP_TTL = 60  # seconds


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


@app.route("/webhook", methods=["POST"])
def webhook():
    data = request.json
    log.debug("WEBHOOK | incoming: %s", data)

    try:
        user_message = ((data.get("message") or {}).get("body") or "").strip()
        phone        = ((data.get("contact") or {}).get("phone_number") or "")
        message_id   = ((data.get("message") or {}).get("whatsapp_message_id") or "")

        if not user_message or not phone:
            return jsonify({"status": "ignored"})

        if message_id and _is_duplicate(message_id):
            log.debug("WEBHOOK | duplicate message_id=%s — ignored", message_id)
            return jsonify({"status": "ignored"})

        # Admin commands bypass the RAG pipeline entirely
        if needs_admin_handling(phone, user_message):
            handle_admin(phone, user_message, collection)
            return jsonify({"status": "success"})

        history      = load_history(phone)
        search_query = rag.rewrite_query(user_message, history)
        hits         = rag.retrieve(collection, search_query)
        # Pass search_query (rewritten) so the LLM sees the unambiguous question
        result       = rag.generate_answer(search_query, hits, history)

        dispatch(phone, result, hits)

        history.append({"role": "user",      "content": user_message})
        history.append({"role": "assistant", "content": result["content"]})
        save_history(phone, history)

        return jsonify({"status": "success"})

    except Exception as e:
        log.exception("WEBHOOK | unhandled error: %s", e)
        return jsonify({"status": "error", "message": str(e)}), 500


if __name__ == "__main__":
    debug = os.getenv("FLASK_DEBUG", "false").lower() == "true"
    app.run(host="0.0.0.0", port=5000, debug=debug)
