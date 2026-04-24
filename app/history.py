# Step 5 — Per-user conversation history (sessions/{phone}.json)
# load_history(phone) — returns [] if last_active > 5 min ago (auto-expire)
# save_history(phone, history) — trims to MEMORY_LIMIT * 2 messages, updates last_active
