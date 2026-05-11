"""
Manual test — sends all 4 message types to the admin phone from config.json.
Run: python test_messaging.py
"""
import json, sys, time
from app.messaging import send_text, send_media, send_interactive, send_carousel

config = json.load(open("config.json"))
PHONE  = config["admin_phones"][0]

products = json.load(open("data/clean_products.json"))
stacker_imgs = [
    p["media"]["images"][0]
    for p in products
    if "stacker" in p.get("name", "").lower() and p.get("media", {}).get("images")
][:2]


def check(label, resp):
    status = "✅" if resp.status_code == 200 else "❌"
    print(f"{status} {label}: {resp.status_code} — {resp.text[:120]}")
    if resp.status_code != 200:
        sys.exit(1)


print(f"Sending to: {PHONE}\n")

# 1. Text
r = send_text(PHONE, "🔧 *Test 1/4 — Text message*\nMessaging module is live!")
check("send_text", r)
time.sleep(1)

# 2. Media
r = send_media(PHONE, stacker_imgs[0], caption="🔧 Test 2/4 — Media message (Nutoy Stacker)", media_type="image")
check("send_media", r)
time.sleep(1)

# 3. Interactive
r = send_interactive(
    PHONE,
    body="🔧 Test 3/4 — Interactive message\nChoose a category:",
    buttons=["Wood Finishes", "Wooden Toys", "Acoustic Panels"],
    header="Numobel Assistant",
    footer="numobel.in",
)
check("send_interactive", r)
time.sleep(1)

# 4. Carousel
cards = [
    {"media_type": "IMAGE", "media_url": img, "button_type": ["QUICK_REPLY", "URL"]}
    for img in stacker_imgs
]
r = send_carousel(PHONE, template_name="nutoy_stacker", cards=cards, language="en", body_var="Test 4/4 — Carousel")
check("send_carousel", r)

print("\nAll 4 message types sent successfully.")
