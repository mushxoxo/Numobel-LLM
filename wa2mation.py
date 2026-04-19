from flask import Flask, request, jsonify
import requests
import os
from dotenv import load_dotenv

import rag_chatbot as rag

load_dotenv()

app = Flask(__name__)

API_KEY    = os.getenv("WA2MATION_API_KEY")
VENDOR_UID = os.getenv("WA2MATION_VENDOR_UID")
SEND_URL   = f"https://wa2mation.com/api/{VENDOR_UID}/contact/send-message"

# Load ChromaDB once at startup
collection = rag.get_collection()
if collection.count() == 0:
    rag.ingest_data(collection)

# Per-user conversation history keyed by phone number
histories = {}


def send_message(phone_number, message):
    response = requests.post(
        SEND_URL,
        json={"phone_number": phone_number, "message_body": message},
        headers={"Authorization": f"Bearer {API_KEY}", "Content-Type": "application/json"}
    )
    print(f"SENT → {phone_number} | Status: {response.status_code}")


@app.route("/webhook", methods=["POST"])
def webhook():
    data = request.json
    print("INCOMING:", data)

    try:
        user_message = data["message"]["body"]
        user_number  = data["contact"]["phone_number"]

        if not user_message:
            return jsonify({"status": "ignored"})

        history = histories.setdefault(user_number, [])

        search_query = rag.rewrite_query(user_message, history)
        hits         = rag.retrieve(collection, search_query)
        result       = rag.generate_answer(user_message, hits, history)
        answer       = result["content"]

        history.append({"role": "user",      "content": user_message})
        history.append({"role": "assistant", "content": answer})

        # Keep only last MEMORY_LIMIT turns
        if len(history) > rag.MEMORY_LIMIT * 2:
            histories[user_number] = history[-(rag.MEMORY_LIMIT * 2):]

        send_message(user_number, answer)
        return jsonify({"status": "success"})

    except Exception as e:
        print("ERROR:", e)
        return jsonify({"status": "error", "message": str(e)}), 500


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000, debug=True)
