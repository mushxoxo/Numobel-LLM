---
plan: 02-01
status: complete
wave: 1
---

# Summary: Plan 02-01 — Intent Config + Exemplars

## What was done
- Added 5 intent classifier constants to `app/config.py` (new "Intent classifier" section after the WhatsApp templates block)
- Created `app/intent_exemplars.json` with 8 intent keys and 15–21 exemplar phrases each

## Verification passed
- All 5 constants importable with correct values (`config OK`)
- `exemplars.json` has 8 keys, each >= 10 phrases (`exemplars OK`)
- Full test suite: 113 passed, 1 pre-existing failure in `test_generate_qna.py::test_call_llm_routes_to_ollama` (Anthropic auth not set in test env — confirmed failing before this plan)

## Outputs for Wave 2
- `app/config.py` exposes:
  - `INTENT_CONFIDENCE_THRESHOLD = 0.75`
  - `INTENT_TENTATIVE_THRESHOLD = 0.60`
  - `QNA_OVERRIDE_THRESHOLD = 0.15`
  - `INTENT_SHORT_MSG_TOKENS = 15`
  - `EXEMPLARS_PATH = BASE_DIR / "app" / "intent_exemplars.json"`
- `app/intent_exemplars.json` has all 8 intent keys using lowercase underscore strings: `greeting`, `brand_discovery`, `brand_deep_dive`, `product_line_query`, `specific_product`, `general_qna`, `out_of_scope`, `chitchat`

## Commit
- `9ff8cef` — feat(02-01): intent classifier constants + exemplars JSON
