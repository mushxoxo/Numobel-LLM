import logging
import os
import sys

# Allow running as `python app/webhook.py` from the project root
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from flask import Flask, request, jsonify
from dotenv import load_dotenv

import rag_chatbot as rag
from app.router import dispatch
from app.history import load_history, save_history

load_dotenv()

log = logging.getLogger('rag_chatbot')
app = Flask(__name__)

# Load ChromaDB once at startup; ingest if empty
collection = rag.get_collection()
if collection.count() == 0:
    log.info("ChromaDB empty — running initial ingest...")
    rag.ingest_data(collection)


@app.route("/webhook", methods=["POST"])
def webhook():
    data = request.json
    log.debug("WEBHOOK | incoming: %s", data)

    try:
        user_message = ((data.get("message") or {}).get("body") or "").strip()
        phone        = ((data.get("contact") or {}).get("phone_number") or "")

        if not user_message or not phone:
            return jsonify({"status": "ignored"})

        history      = load_history(phone)
        search_query = rag.rewrite_query(user_message, history)
        hits         = rag.retrieve(collection, search_query)
        result       = rag.generate_answer(user_message, hits, history)

        dispatch(phone, result, hits)

        history.append({"role": "user",      "content": user_message})
        history.append({"role": "assistant", "content": result["content"]})
        save_history(phone, history)

        return jsonify({"status": "success"})

    except Exception as e:
        log.exception("WEBHOOK | unhandled error: %s", e)
        return jsonify({"status": "error", "message": str(e)}), 500


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000, debug=True)
