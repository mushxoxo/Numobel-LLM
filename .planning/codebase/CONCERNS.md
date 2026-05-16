# Codebase Concerns
_Generated: 2026-05-16_

---

## Known Bugs

### Carousel Per-Card Variable Substitution Broken (Blocker)
- **Symptoms:** wa2mation returns `#131008 Required parameter is missing` when carousel cards include per-card body/button variables. Alternative code paths using `carousel_variables` or `components` keys in cards reach Meta but generate `#132012` (empty header parameters).
- **Files:** `app/messaging/carousel.py`, `app/router.py`
- **Root cause:** wa2mation's `send-carousel-template-message` endpoint silently drops all card-level fields (`field_1`, `field_2`, `button_1`, etc.) beyond `media_url`. This is a third-party API limitation discovered 2026-05-15.
- **Workaround:** Current code only sends `media_url` per card (images display but card text/buttons are absent). Unblocking requires contacting wa2mation support for the correct per-card variable format.

### Button Tap Payload Format Untested
- **Symptoms:** When a user taps an interactive button, wa2mation's callback payload format is unknown. The code assumes `message.body` contains the button label text, but this has never been verified against a live button tap.
- **Files:** `app/webhook.py` (lines 70–71)
- **Risk:** Button taps from end-users (including admin `[Approve]`/`[Refine]` buttons) may not be parsed correctly, causing the state machine to fall through to the "Unknown command" fallback.

### In-Memory Dedup Lost on Restart
- **Symptoms:** The `_seen_ids` OrderedDict in `app/webhook.py` is process-local. If gunicorn worker restarts (OOM, crash, deploy), any message IDs received just before the restart will be re-processed on the next callback from wa2mation, which fires 6–8 times per message.
- **Files:** `app/webhook.py` (lines 44–60)
- **Impact:** Duplicate RAG queries and duplicate WhatsApp message sends to users on worker restart.

### Admin Session State Not Persisted Across Worker Restarts
- **Symptoms:** `_sessions` dict in `app/admin.py` is module-level in-memory state. On any restart (including code deploys), all active admin sessions are lost mid-flow. The disk-based `refine_state` rescue path (`load_refine_state`) only partially covers `TRAINING_CHAT`, `TRAINING_CONFIRM_SAVE`, and `TRAINING_CONFIRM_PREFS` states — the `TRAINING_REVIEW` and `MENU` states are not persisted to disk and are lost silently.
- **Files:** `app/admin.py` (line 26), `app/refinement/storage.py`

---

## Security Considerations

### No Webhook Incoming Request Authentication
- **Risk:** The `/webhook` POST endpoint in `app/webhook.py` accepts any request without verifying it originates from wa2mation. An attacker who discovers the endpoint can send arbitrary messages, trigger RAG queries, or inject admin commands for registered admin phone numbers.
- **Files:** `app/webhook.py`
- **Current mitigation:** None. Admin command guard (`is_admin()`) checks the `phone_number` field in the JSON body, but that field is attacker-controlled.
- **Recommendation:** Validate an HMAC signature or shared secret from wa2mation on every inbound request.

### Vendor UID Re-Read From Environment Per Request
- **Risk:** `WA2MATION_VENDOR_UID` is read via `os.getenv()` on every `wa2mation_post()` call (line 39 of `app/messaging/client.py`), even though the session singleton caches the API key. If `vendor_uid` is `None` (missing env var), the URL silently becomes `https://wa2mation.com/api/None/contact/...` and requests are sent with an invalid path rather than raising an error.
- **Files:** `app/messaging/client.py` (lines 39–40)
- **Recommendation:** Cache `vendor_uid` in the session singleton alongside the API key, and raise `RuntimeError` at session creation time if it is absent.

### Admin Phone Numbers Stored in Untracked Config File
- **Risk:** `config.json` is gitignored and must be manually created on each deployment. There is no validation or schema enforcement. A typo in `config.json` causes `_load_config()` to silently return `{"admin_phones": [], ...}`, effectively disabling admin authentication without any warning log.
- **Files:** `app/admin.py` (lines 44–49)
- **Recommendation:** Log a WARNING at startup if `config.json` is missing or has no admin phones, so misconfiguration is surfaced immediately.

### Refinement LLM Receives Full Pair Including Unapproved Content
- **Risk:** The refinement engine (`app/refinement/engine.py`) sends the current Q&A pair's full content in the system prompt to the LLM on every chat turn. A malicious admin could craft pair content that attempts prompt injection against the refinement LLM.
- **Files:** `app/refinement/prompts.py` (lines 37–43), `app/refinement/engine.py`
- **Impact:** Low — admin accounts are trusted by design, but worth noting.

---

## Technical Debt

### `auto-generate` Admin Feature Is a Stub
- **Description:** The "Auto-generate" button in the admin WhatsApp menu sends `"Auto-generate is coming soon. Use python training/generate_qna.py from the CLI for now."` There is no in-app generation flow.
- **Files:** `app/admin.py` (line 247)
- **Impact:** Admins cannot trigger Q&A generation without CLI access to the cloud VM.

### `_load_config()` Re-Reads File on Every Admin Check
- **Description:** `_load_config()` opens and parses `config.json` on every call to `is_admin()`, `needs_admin_handling()`, and `_is_timed_out()`. With 6–8 wa2mation callbacks per message, this is 6–8 file reads per message even for non-admin users.
- **Files:** `app/admin.py` (lines 44–49)
- **Fix approach:** Cache config at startup with an optional reload mechanism (e.g., re-read on SIGHUP or after a TTL).

### `get_collection()` Creates a New ChromaDB Client on Every Call in CLI Mode
- **Description:** `app/rag.py:get_collection()` creates a new `PersistentClient` every time it is called (line 30). In the CLI path (`rag_chatbot.py`), this is called once at startup, which is acceptable. But if called multiple times (e.g., from training scripts), each call instantiates a new SQLite client.
- **Files:** `app/rag.py` (lines 28–34)
- **Fix approach:** Cache the collection as a module-level singleton.

### Datetime Timezone Inconsistency
- **Description:** `app/history.py` uses `datetime.now()` (naive local time) for session timestamps, while `app/admin.py` and `app/refinement/storage.py` use `datetime.utcnow()`. Cross-module comparisons (e.g., between history expiry and admin timeout) are based on different time references. This has no current impact because they are not compared across modules, but adds fragility.
- **Files:** `app/history.py` (lines 40, 68), `app/admin.py` (line 99), `app/refinement/storage.py` (line 60)

### `_keyword_fallback()` Hardcodes Product-Specific Brand Names
- **Description:** `_keyword_fallback()` in `app/rag.py` (line 241) contains a hardcoded set of brand/product keywords (`'rubio'`, `'nuacoustics'`, etc.) to determine carousel message type when LLM returns non-JSON. Adding new brands requires code changes, not data changes.
- **Files:** `app/rag.py` (lines 241–245)

### Jupyter Notebook Checkpoints Checked Into Repo (Sort of)
- **Description:** `.ipynb_checkpoints/` is gitignored but the directory exists with large files (1MB CSV, 1MB notebook). This suggests an earlier period of notebook-based development. The files are not in git but consume disk space and create confusion about what is authoritative source.
- **Files:** `/home/mush/git/github/numobel/.ipynb_checkpoints/` (not in git)

---

## Performance Bottlenecks

### Sequential LLM Call Chain Per Message (High Latency)
- **Description:** Every user message triggers three sequential Ollama calls before responding: `rewrite_query()` (llama3.2 generate), `get_embedding()` (mxbai-embed-large embed), `generate_answer()` (llama3.2 chat). On a local Ollama instance, total latency is 5–15 seconds per message depending on hardware.
- **Files:** `app/webhook.py` (lines 94–96), `app/rag.py`
- **Impact:** Users experience long response delays. WhatsApp has no typing indicator from the bot side.
- **Improvement path:** `REWRITE_QUERY = False` in `app/config.py` eliminates the first LLM call; acceptable for users who send standalone questions.

### ChromaDB Upsert Blocks Webhook During Initial Ingest
- **Description:** If ChromaDB is empty at startup, `ingest_data()` is called synchronously (lines 23–28 of `app/webhook.py`), blocking the Flask app from handling requests until all 199 products are embedded (several minutes). During this window, wa2mation retries accumulate.
- **Files:** `app/webhook.py` (lines 22–28), `app/rag.py` (`ingest_data()`)
- **Improvement path:** Run ingest in a background thread at startup; return 503 for incoming requests until collection is ready.

### `pending.jsonl` Scanned Linearly on Every Admin Review Step
- **Description:** `load_next_pending()`, `mark_approved()`, `lock_pair()`, `unlock_pair()`, and `is_locked()` all perform full linear scans of `pending.jsonl`. For the current dataset (hundreds of pairs) this is acceptable, but degrades as the file grows.
- **Files:** `app/refinement/storage.py`

---

## Platform Constraints

### Linux-Only Deployment (`fcntl`)
- **Description:** `app/history.py` uses `fcntl.flock()` for exclusive file locking. This module does not exist on Windows. Development on Windows/macOS requires workarounds.
- **Files:** `app/history.py` (lines 1, 64)

### Single Gunicorn Worker Required
- **Description:** `chromadb.PersistentClient` (SQLite backend) is not safe for concurrent writes from multiple processes. The gunicorn deployment is constrained to `--workers 1`. Horizontal scaling (multiple VMs or workers) is blocked without migrating ChromaDB to a server mode (e.g., `chromadb.HttpClient` pointing to a ChromaDB server).
- **Files:** `app/rag.py` (line 30), `Dockerfile`, gunicorn configuration in `CLAUDE.md`

### wa2mation 24-Hour Session Window
- **Description:** Non-template messages can only be sent within 24 hours of the user last messaging. If the bot needs to proactively message a user or respond after a long delay, it must use a template message. Only the carousel template path bypasses this limit.
- **Files:** `app/router.py` — no enforcement or detection of this constraint

### Carousel Template Hard-Coded to `numobel_catalogue_4`
- **Description:** `_CAROUSEL_TEMPLATE = "numobel_catalogue_4"` is a module-level constant in `app/router.py`. The older `nutoy_stacker` template is deprecated. Adding new approved carousel templates requires a code change.
- **Files:** `app/router.py` (lines 7–8)

---

## Missing Functionality

### No Webhook Signature Verification (see Security section)

### No Monitoring or Alerting
- **Description:** There is no error tracking service (Sentry, etc.) or uptime monitoring. Application errors are logged to `logs/rag_chatbot.log` (file on the VM), but there is no alerting if the bot goes down, Ollama stops responding, or error rates spike.
- **Impact:** Silent failures — customers get no response and there is no notification to operators.

### No Rate Limiting on User Messages
- **Description:** Any phone number can send unlimited messages, triggering unlimited Ollama LLM calls and ChromaDB queries. There is no per-user throttle or abuse protection beyond the 60-second dedup window for identical message IDs.
- **Files:** `app/webhook.py`

### Greeting Retrieval Pollution (Known, Partially Mitigated)
- **Description:** Sending "hello" retrieves Nutoy product chunks (closest embedding match), producing irrelevant product responses to greetings. The mitigation is to generate and approve greeting Q&A pairs via the training pipeline so they outrank product chunks. Until enough greeting pairs are approved, this affects first-time users.
- **Files:** `app/rag.py` (retrieval path)
- **Status:** Workaround documented; not structurally fixed.

### No Pagination or Scroll for Long Product Answers
- **Description:** The RAG `generate_answer()` has no enforcement that the generated `content` stays under WhatsApp message limits for `text` type (4096 chars). The `constraints.py` limits are only enforced in the training/review pipeline, not on live RAG output.
- **Files:** `app/rag.py` (`generate_answer()`), `app/refinement/constraints.py`

---

## Operational Concerns

### ChromaDB Recovery Requires Manual CLI Intervention
- **Description:** If `chroma_db/` is lost or corrupted (e.g., disk failure, VM reset), recovery requires SSH access to the VM and running two manual Python commands. There is no automated restore script or health check endpoint.
- **Files:** `app/rag.py`, `app/webhook.py`
- **Recovery procedure:** Documented in `CLAUDE.md` — requires `approved.jsonl` (tracked in git) and a running Ollama instance.

### Stale Pair Locks Persist After Crashes
- **Description:** `lock_pair()` writes `in_review: {phone}` to `pending.jsonl` when an admin starts refinement. If Flask crashes during refinement, the lock is never cleared. The `unlock_pair()` call only happens on clean cancellation or approval. Stale locks silently skip pairs when `load_next_pending(skip_locked=True)` is called.
- **Files:** `app/refinement/storage.py` (`lock_pair`, `unlock_pair`)
- **Recovery:** Manually edit `pending.jsonl` to remove `in_review` fields, or restart and send `:admin off` which calls `unlock_pair` for the current pair.

### `data/` Directory Gitignored — No Version Control on Product Catalogue
- **Description:** `data/Products.csv` and `data/clean_products.json` are gitignored. The product catalogue (199 products, 5 brands) has no version history. If the CSV is accidentally corrupted or lost, there is no recovery path from git.
- **Impact:** Requires re-sourcing the original product CSV from an external system.

### No Health Check Endpoint
- **Description:** There is no `/health` or `/ping` endpoint. Load balancers, uptime monitors, and deployment readiness probes have no way to verify the app is ready to serve traffic (including confirming ChromaDB is loaded and Ollama is reachable).
- **Files:** `app/webhook.py`

### Logs Stored Only on VM Disk
- **Description:** Application logs go to `logs/rag_chatbot.log` on the VM's local filesystem (gitignored). There is no log aggregation, rotation configuration, or off-VM log shipping. On VM reset or disk issue, all historical logs are lost.
- **Files:** `app/log.py`

---

## Test Coverage Gaps

### RAG `generate_answer()` Not Integration-Tested Against Real Ollama
- **Description:** `tests/test_rag_chatbot.py` mocks `ollama.chat()` entirely. The prompt construction, JSON parsing, keyword fallback, and faithful-replay logic are tested, but no test exercises the actual LLM response quality or token count reporting.
- **Files:** `tests/test_rag_chatbot.py`, `app/rag.py`
- **Priority:** Medium — functional correctness is covered; LLM output quality is tested manually.

### WhatsApp Button Tap Handling Untested End-to-End
- **Description:** No test exercises what happens when wa2mation sends a button-tap callback. The assumed payload format (`message.body` = button label) is not confirmed against live wa2mation behaviour.
- **Files:** `tests/test_webhook.py`, `app/webhook.py`
- **Priority:** High — admin flow and user interactive buttons both depend on this.

### Admin Session Disk Resume Path Has Limited Coverage
- **Description:** `test_admin.py` does not test the `:admin on` resume path (re-entering an active refinement session from disk state). The `load_refine_state` → `_sessions[phone] = disk` branch in `handle_admin` is untested.
- **Files:** `tests/test_admin.py`, `app/admin.py` (lines 200–218)
- **Priority:** Medium.

### `send_carousel()` wa2mation Response Not Validated in Tests
- **Description:** `tests/messaging/test_carousel.py` verifies the correct payload is constructed and posted, but does not test the response-handling path (non-200 status, Meta error codes).
- **Files:** `tests/messaging/test_carousel.py`, `app/messaging/carousel.py`
- **Priority:** Low.
