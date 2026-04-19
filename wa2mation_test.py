from flask import Flask, request, jsonify
import requests
import os
from dotenv import load_dotenv

load_dotenv()

app = Flask(__name__)

API_KEY    = os.getenv("WA2MATION_API_KEY")
VENDOR_UID = os.getenv("WA2MATION_VENDOR_UID")
SEND_URL   = f"https://wa2mation.com/api/{VENDOR_UID}/contact/send-message"


def send_message(phone_number, message):
    response = requests.post(
        SEND_URL,
        json={"phone_number": phone_number, "message_body": message},
        headers={"Authorization": f"Bearer {API_KEY}", "Content-Type": "application/json"}
    )
    print(f"SENT → {phone_number}: {message} | Status: {response.status_code}")


@app.route("/webhook", methods=["POST"])
def webhook():
    data = request.json
    print("INCOMING:", data)

    try:
        user_message = data["message"]["body"]
        user_number  = data["contact"]["phone_number"]

        if user_message and user_message.strip().upper() == "TEST":
            send_message(user_number, "TEST TOO")

        return jsonify({"status": "success"})

    except Exception as e:
        print("ERROR:", e)
        return jsonify({"status": "error", "message": str(e)}), 500


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000, debug=True)
