# wa2mation API Reference

**Source:** https://documenter.getpostman.com/view/29712127/2sBXiojUpr

---

## Base URL & Auth

```
Base URL  : https://wa2mation.com/api
Vendor UID: WA2MATION_VENDOR_UID  (set in .env)
```

Every request requires a `Bearer` token in the `Authorization` header:

```
Authorization: Bearer WA2MATION_API_KEY  (set in .env)
Content-Type: application/json
```

Phone numbers must be in international format without `+` (e.g. `91XXXXXXXXXX` for India).

---

## Endpoints

### 1. Send Message (Text)

`POST /api/{vendorUid}/contact/send-message`

Sends a plain text WhatsApp message. **This is the primary endpoint for the RAG chatbot.**

**Body:**

| Field | Required | Description |
|---|---|---|
| `phone_number` | Yes | Recipient in international format (e.g. `91XXXXXXXXXX`) |
| `message_body` | Yes | The text to send |
| `from_phone_number_id` | No | Sender phone ID; omit to use the account default |
| `contact` | No | Auto-create the contact if it doesn't exist yet (see Contact Object) |

```json
{
    "phone_number": "91XXXXXXXXXX",
    "message_body": "Hello from Numobel!"
}
```

**curl:**

```bash
curl --location 'https://wa2mation.com/api/{VENDOR_UID}/contact/send-message' \
  --header 'Authorization: Bearer {API_KEY}' \
  --header 'Content-Type: application/json' \
  --data '{
    "phone_number": "91XXXXXXXXXX",
    "message_body": "Hello from Numobel!"
  }'
```

---

### 2. Send Media Message

`POST /api/{vendorUid}/contact/send-media-message`

Sends an image, video, or document. Useful for sending product images alongside a RAG answer.

**Body:**

| Field | Required | Description |
|---|---|---|
| `phone_number` | Yes | Recipient |
| `media_type` | Yes | `image`, `video`, or `document` |
| `media_url` | Yes | Publicly accessible URL of the file |
| `caption` | No | Caption text (for `image` or `video` only) |
| `file_name` | No | Display filename (for `document` only) |
| `from_phone_number_id` | No | Omit to use default |
| `contact` | No | Auto-create contact if missing |

```json
{
    "phone_number": "91XXXXXXXXXX",
    "media_type": "image",
    "media_url": "https://static.wixstatic.com/media/example.jpg",
    "caption": "Rubio Monocoat Exterior Wood Cleaner"
}
```

**curl:**

```bash
curl --location 'https://wa2mation.com/api/{VENDOR_UID}/contact/send-media-message' \
  --header 'Authorization: Bearer {API_KEY}' \
  --header 'Content-Type: application/json' \
  --data '{
    "phone_number": "91XXXXXXXXXX",
    "media_type": "image",
    "media_url": "https://static.wixstatic.com/media/example.jpg",
    "caption": "Rubio Monocoat Exterior Wood Cleaner"
  }'
```

---

### 3. Send Template Message

`POST /api/{vendorUid}/contact/send-template-message`

Sends a pre-approved WhatsApp Business template. Required for outbound messages to users who haven't messaged you in the last 24 hours.

**Body:**

| Field | Required | Description |
|---|---|---|
| `phone_number` | Yes | Recipient |
| `template_name` | Yes | Name of the approved template |
| `template_language` | Yes | e.g. `en` |
| `from_phone_number_id` | No | Omit to use default |
| `header_image` / `header_video` / `header_document` | No | Media URL for the template header |
| `header_document_name` | No | Display name for document header |
| `header_field_1` | No | Variable for header text |
| `field_1` … `field_4` | No | Body variable substitutions |
| `button_0`, `button_1` | No | Button variable substitutions |
| `copy_code` | No | Coupon code for copy-code buttons |
| `location_latitude`, `location_longitude`, `location_name`, `location_address` | No | For location header type |
| `contact` | No | Auto-create contact if missing |

```json
{
    "phone_number": "91XXXXXXXXXX",
    "template_name": "your_template_name",
    "template_language": "en",
    "field_1": "Rubio Monocoat"
}
```

**curl:**

```bash
curl --location 'https://wa2mation.com/api/{VENDOR_UID}/contact/send-template-message' \
  --header 'Authorization: Bearer {API_KEY}' \
  --header 'Content-Type: application/json' \
  --data '{
    "phone_number": "91XXXXXXXXXX",
    "template_name": "your_template_name",
    "template_language": "en"
  }'
```

---

### 4. Send Interactive Message

`POST /api/{vendorUid}/contact/send-interactive-message`

Sends a message with buttons, a CTA link, or a scrollable list. The `interactive_type` field controls which format is used.

**Body:**

| Field | Required | Description |
|---|---|---|
| `phone_number` | Yes | Recipient |
| `interactive_type` | Yes | `button`, `cta_url`, or `list` |
| `header_type` | No | `text`, `image`, `video`, or `document` |
| `header_text` | No | Header text (when `header_type` is `text`) |
| `media_link` | No | Media URL (when `header_type` is `image`, `video`, or `document`) |
| `body_text` | No | Main message body |
| `footer_text` | No | Footer text |
| `buttons` | When `interactive_type` = `button` | Object with keys `"1"`–`"3"` and button label values (max 3) |
| `cta_url` | When `interactive_type` = `cta_url` | Object with `display_text` and `url` |
| `list_data` | When `interactive_type` = `list` | Object with `button_text` and `sections` (see structure below) |
| `from_phone_number_id` | No | Omit to use default |
| `contact` | No | Auto-create contact if missing |

**Button example:**

```json
{
    "phone_number": "91XXXXXXXXXX",
    "interactive_type": "button",
    "header_type": "text",
    "header_text": "Numobel Products",
    "body_text": "How can I help you today?",
    "footer_text": "Numobel Assistant",
    "buttons": {
        "1": "Browse Products",
        "2": "Get a Quote",
        "3": "Contact Us"
    }
}
```

**List example:**

```json
{
    "phone_number": "91XXXXXXXXXX",
    "interactive_type": "list",
    "body_text": "Choose a category",
    "list_data": {
        "button_text": "View Categories",
        "sections": {
            "section_1": {
                "title": "Wood Finishes",
                "id": "section_1",
                "rows": {
                    "row_1": {
                        "id": "row_1",
                        "row_id": "1",
                        "title": "Rubio Monocoat",
                        "description": "Premium wood oils and cleaners"
                    }
                }
            }
        }
    }
}
```

**curl (button type):**

```bash
curl --location 'https://wa2mation.com/api/{VENDOR_UID}/contact/send-interactive-message' \
  --header 'Authorization: Bearer {API_KEY}' \
  --header 'Content-Type: application/json' \
  --data '{
    "phone_number": "91XXXXXXXXXX",
    "interactive_type": "button",
    "header_type": "text",
    "header_text": "Numobel Products",
    "body_text": "How can I help you today?",
    "buttons": {
        "1": "Browse Products",
        "2": "Get a Quote"
    }
  }'
```

---

### 5. Send Carousel Template Message

`POST /api/{vendorUid}/contact/send-carousel-template-message`

Sends a carousel of cards, each with a media item and up to 2 buttons. Requires a pre-approved carousel template. Minimum 2 cards, maximum 10.

**Body:**

| Field | Required | Description |
|---|---|---|
| `phone_number` | Yes | Recipient |
| `template_name` | Yes | Name of the approved carousel template |
| `template_language` | Yes | e.g. `en_us` |
| `field_1` … `field_3` | No | Body-level variable substitutions |
| `carousel_templates` | Yes | Array of card objects (min 2, max 10) |
| `from_phone_number_id` | No | Omit to use default |
| `contact` | No | Auto-create contact if missing |

Each card in `carousel_templates`:

| Field | Description |
|---|---|
| `media_type` | `IMAGE` or `VIDEO` |
| `media_url` | Publicly accessible URL |
| `button_type` | Array of button types: `QUICK_REPLY`, `PHONE_NUMBER`, `URL` |

```json
{
    "phone_number": "91XXXXXXXXXX",
    "template_name": "product_carousel",
    "template_language": "en_us",
    "carousel_templates": [
        {
            "media_type": "IMAGE",
            "media_url": "https://static.wixstatic.com/media/example1.jpg",
            "button_type": ["QUICK_REPLY", "URL"]
        },
        {
            "media_type": "IMAGE",
            "media_url": "https://static.wixstatic.com/media/example2.jpg",
            "button_type": ["QUICK_REPLY", "URL"]
        }
    ]
}
```

---

### 6. Create Contact

`POST /api/{vendorUid}/contact/create`

**Body:**

| Field | Required | Description |
|---|---|---|
| `phone_number` | Yes | In international format |
| `first_name` | No | |
| `last_name` | No | |
| `email` | No | |
| `country` | No | e.g. `india` |
| `language_code` | No | e.g. `en` |
| `groups` | No | Comma-separated group names |
| `custom_fields` | No | Object of arbitrary key-value pairs |

```json
{
    "phone_number": "91XXXXXXXXXX",
    "first_name": "Arushi",
    "country": "india",
    "language_code": "en"
}
```

---

### 7. Update Contact

`POST /api/{vendorUid}/contact/update/{phoneNumber}`

Same fields as Create Contact, plus:

| Field | Description |
|---|---|
| `whatsapp_opt_out` | `true` / `false` |
| `enable_ai_bot` | `true` / `false` |

> Only send fields you want to update — sending a blank value may clear an existing value.

---

### 8. Get Contact

`GET /api/{vendorUid}/contact?phone_number_or_email={value}`

Returns a single contact record by phone number or email.

```bash
curl --location 'https://wa2mation.com/api/{VENDOR_UID}/contact?phone_number_or_email=91XXXXXXXXXX' \
  --header 'Authorization: Bearer {API_KEY}'
```

---

### 9. Get Contacts (List)

`GET /api/{vendorUid}/contacts`

Returns a paginated list of contacts.

| Query Param | Description |
|---|---|
| `page` | Page number (default 1) |
| `page_size` | Results per page |
| `search_term` | Filter by name/phone |

```bash
curl --location 'https://wa2mation.com/api/{VENDOR_UID}/contacts?page=1&page_size=10' \
  --header 'Authorization: Bearer {API_KEY}'
```

---

### 10. Assign Team Member

`POST /api/{vendorUid}/contact/assign-team-member`

Assigns a conversation to a team member and controls bot toggles.

| Field | Description |
|---|---|
| `username_or_email` | Team member's username or email |
| `phone_number` | Contact's phone number |
| `enable_ai_bot` | `1` = on, `0` = off |
| `enable_reply_bot` | `1` = on, `0` = off |

---

## Contact Object (shared across endpoints)

When included in a send request, this will auto-create the contact if they don't already exist in wa2mation.

```json
"contact": {
    "first_name": "Arushi",
    "last_name": "Tiwari",
    "email": "user@example.com",
    "country": "india",
    "language_code": "en",
    "groups": "group1,group2",
    "custom_fields": {
        "BDay": "2025-09-04"
    }
}
```

---

## Incoming Webhook Payload

wa2mation forwards incoming WhatsApp messages to your webhook URL via POST. Structure:

```json
{
    "contact": {
        "status": "existing/updated/new",
        "phone_number": "91XXXXXXXXXX",
        "uid": "...",
        "first_name": "...",
        "last_name": "...",
        "email": "...",
        "language_code": "en",
        "country": "india"
    },
    "message": {
        "whatsapp_business_phone_number_id": "...",
        "whatsapp_message_id": "wamid...",
        "replied_to_whatsapp_message_id": "wamid...",
        "is_new_message": true,
        "body": "the text the user sent",
        "status": null,
        "media": {
            "type": "image",
            "link": "...",
            "caption": null,
            "mime_type": "image/jpeg",
            "file_name": "...",
            "original_filename": "..."
        }
    },
    "whatsapp_webhook_payload": {}
}
```

Key fields to use in code:

| Field | Description |
|---|---|
| `contact.phone_number` | Sender's phone number — use this to reply |
| `message.body` | Text content of the incoming message |
| `message.is_new_message` | `true` for new messages, `false` for status updates |
| `message.media` | Present when the user sends an image/video/document |

---

## Relevant Endpoints for the RAG Chatbot

For the Numobel WhatsApp bot, the two endpoints we actually use are:

| Use Case | Endpoint |
|---|---|
| Reply with RAG text answer | `POST /contact/send-message` |
| Send product image alongside answer | `POST /contact/send-media-message` |
