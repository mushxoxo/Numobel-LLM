"""
Intent classifier for Numobel WhatsApp chatbot.

Public API:
  - IntentEnum: 8-value enum of message intents
  - classify_intent(text, embedding=None, history=None) -> dict
  - compute_centroids() -> None  (called by startup orchestrator)

No LLM calls, no ChromaDB queries — pure in-memory classification.
Layer 1: compiled regex rules (greeting, OOS, chitchat, brand names) — no embedding needed.
Layer 2: cosine similarity against per-intent centroid embeddings.
"""

import json
import re
import numpy as np
from enum import Enum

from app.log import get_logger
from app.config import (
    INTENT_CONFIDENCE_THRESHOLD,
    INTENT_TENTATIVE_THRESHOLD,
    INTENT_SHORT_MSG_TOKENS,
    EXEMPLARS_PATH,
)

log = get_logger()

# Module-level classifier state
_centroids: dict[str, np.ndarray] = {}
_intent_ready: bool = False


class IntentEnum(str, Enum):
    GREETING           = "greeting"
    BRAND_DISCOVERY    = "brand_discovery"
    BRAND_DEEP_DIVE    = "brand_deep_dive"
    PRODUCT_LINE_QUERY = "product_line_query"
    SPECIFIC_PRODUCT   = "specific_product"
    GENERAL_QNA        = "general_qna"
    OUT_OF_SCOPE       = "out_of_scope"
    CHITCHAT           = "chitchat"


# Intents that carry no product context — never inherit these from history.
# A greeting or out-of-scope prior should never override a tentative product query.
_NON_INHERITABLE: frozenset[IntentEnum] = frozenset({
    IntentEnum.GREETING,
    IntentEnum.CHITCHAT,
    IntentEnum.OUT_OF_SCOPE,
})

# Vague product intents — when a tentative embedding best-match is one of these but
# the prior is a more-specific product intent (e.g., brand_deep_dive), inherit the
# more-specific prior instead of accepting the vague match.  (INTENT-11 / Bug 1 fix)
_VAGUE_PRODUCT_INTENTS: frozenset[IntentEnum] = frozenset({
    IntentEnum.BRAND_DISCOVERY,
    IntentEnum.GENERAL_QNA,
})


# ─── Layer 1 compiled patterns ────────────────────────────────────────────────

_GREETING_RE = re.compile(
    r"^(hi+|hey+|hello+|helo|hlo|hlw|hai+|howdy|sup|wassup|yo|namaste|greetings"
    r"|good\s*(morning|afternoon|evening|day|nite|night))[\s!.]*$",
    re.IGNORECASE,
)

_OOS_PATTERNS = re.compile(
    r"(where.{0,10}(my|is).{0,10}order|track.{0,10}(order|shipment)|"
    r"(want to|wanna|wish to)\s+return|refund|cancel\s+(my\s+)?order|"
    r"delivery\s+(date|status|time|when)|when.{0,10}(arrive|deliver)|"
    r"shipping\s+cost|exchange\s+product|warranty\s+claim|"
    r"nearest\s+store|where\s+to\s+buy|damaged\s+product)",
    re.IGNORECASE,
)

_CHITCHAT_EXACT = frozenset({
    "how are you", "what is your name", "who are you", "are you a bot",
    "are you human", "kaise ho", "aap kaun ho", "bot hai kya",
    "thanks", "thank you", "ok", "okay", "got it", "nice", "cool",
    "great", "awesome", "bye", "goodbye", "no thanks", "not interested",
})

# Standalone brand-name queries — match when the entire message is (just) a brand name.
# Returns brand_deep_dive immediately, bypassing centroid scoring.  (INTENT-12 / Bug 2 fix)
_BRAND_NAME_RE = re.compile(
    r"^(rubio\s+monocoat|nuacoustics|nutoy|nupanel|nuwork)[\s!?.]*$",
    re.IGNORECASE,
)


# ─── Private helpers ──────────────────────────────────────────────────────────

def _prior_intent(history: list[dict]) -> "IntentEnum | None":
    """Return the most recent user-turn intent from history, or None."""
    for entry in reversed(history):
        if entry.get("role") == "user" and "intent" in entry:
            try:
                return IntentEnum(entry["intent"])
            except ValueError:
                continue
    return None


def _cosine_similarity(a: np.ndarray, b: np.ndarray) -> float:
    """Compute cosine similarity between two vectors. Returns 0.0 if either is zero."""
    denom = np.linalg.norm(a) * np.linalg.norm(b)
    if denom == 0.0:
        return 0.0
    return float(np.dot(a, b) / denom)


def _layer1_classify(text: str) -> "dict | None":
    """Layer 1: rule-based classification. Returns result dict or None if no rule matches."""
    t = text.strip()
    t_lower = t.lower()
    if _GREETING_RE.fullmatch(t):
        return {"intent": IntentEnum.GREETING, "confidence": 1.0, "layer": "rule"}
    if _OOS_PATTERNS.search(t):
        return {"intent": IntentEnum.OUT_OF_SCOPE, "confidence": 1.0, "layer": "rule"}
    if t_lower in _CHITCHAT_EXACT:
        return {"intent": IntentEnum.CHITCHAT, "confidence": 1.0, "layer": "rule"}
    if _BRAND_NAME_RE.fullmatch(t):
        return {"intent": IntentEnum.BRAND_DEEP_DIVE, "confidence": 1.0, "layer": "rule"}
    return None


def _layer2_classify(
    embedding: list[float],
    history: "list[dict] | None",
    text: str,
) -> dict:
    """Layer 2: cosine centroid classification with optional multi-turn inherit.

    Inherit rule (INTENT-06 / INTENT-09 / INTENT-10 / INTENT-11):
      Inherit is applied only when ALL of the following hold:
        1. best_sim is tentative (TENTATIVE_THRESHOLD ≤ best_sim < CONFIDENCE_THRESHOLD)
        2. word_count < INTENT_SHORT_MSG_TOKENS  (short, ambiguous message)
        3. prior is inheritable (not in _NON_INHERITABLE)
        4. One of:
           a. best_intent == prior  (embedding already agrees with prior direction), OR
           b. best_intent is non-product (chitchat/greeting/out_of_scope) — product prior wins, OR
           c. best_intent is a vague product intent (brand_discovery / general_qna) AND
              prior is NOT a vague product intent — more-specific prior wins.

      Condition 4b/4c is the INTENT-10/11 fix: a more-specific prior (brand_deep_dive) must
      not lose to a vague tentative match (brand_discovery, general_qna).
    """
    vec = np.array(embedding, dtype=float)
    best_intent = IntentEnum.GENERAL_QNA
    best_sim = -1.0

    # Collect all scores for debug logging
    all_scores: dict[str, float] = {}
    for intent_key, centroid in _centroids.items():
        sim = _cosine_similarity(vec, centroid)
        all_scores[intent_key] = sim
        if sim > best_sim:
            best_sim = sim
            try:
                best_intent = IntentEnum(intent_key)
            except ValueError:
                best_intent = IntentEnum.GENERAL_QNA

    prior = _prior_intent(history or [])
    word_count = len(text.split())

    # Only inherit product-relevant priors. Greeting / chitchat / out-of-scope carry
    # no product context and must never override a tentative product embedding match.
    inheritable_prior = prior if (prior and prior not in _NON_INHERITABLE) else None

    if best_sim >= INTENT_CONFIDENCE_THRESHOLD:
        result = {"intent": best_intent, "confidence": best_sim, "layer": "embedding"}
        fallback_reason = None
    elif best_sim >= INTENT_TENTATIVE_THRESHOLD:
        # Inherit rules (INTENT-06 / INTENT-09 / INTENT-10 / INTENT-11):
        #   ALWAYS inherit when: prior is product-relevant AND best is non-product
        #     (chitchat/greeting/out_of_scope) — a product prior beats a non-product match.
        #   ALWAYS inherit when: prior is a specific product intent AND best is a vague
        #     product intent (brand_discovery / general_qna) — specificity wins.
        #   ONLY inherit when prior==best when: both are specific product intents — prevents
        #     a generic product prior from overriding a specific product match at the same level.
        _best_is_non_product  = best_intent in _NON_INHERITABLE
        _best_is_vague        = best_intent in _VAGUE_PRODUCT_INTENTS
        _prior_is_specific    = (
            inheritable_prior is not None
            and inheritable_prior not in _VAGUE_PRODUCT_INTENTS
        )
        _should_inherit = (
            word_count < INTENT_SHORT_MSG_TOKENS
            and inheritable_prior is not None
            and (
                _best_is_non_product
                or (_best_is_vague and _prior_is_specific)
                or inheritable_prior == best_intent
            )
        )
        if _should_inherit:
            result = {"intent": inheritable_prior, "confidence": best_sim, "layer": "embedding_inherit"}
            fallback_reason = f"tentative+short({word_count}w)+inherit_from_{inheritable_prior.value}"
        else:
            result = {"intent": best_intent, "confidence": best_sim, "layer": "embedding"}
            fallback_reason = f"tentative+no_inherit(words={word_count},prior={prior})"
    else:
        result = {"intent": IntentEnum.GENERAL_QNA, "confidence": best_sim, "layer": "embedding"}
        fallback_reason = f"below_tentative_threshold({INTENT_TENTATIVE_THRESHOLD})"

    log.info(
        "INTENT_DEBUG | query=%r best=%s confidence=%.3f layer=%s "
        "fallback_reason=%s prior=%s word_count=%d scores=%s",
        text[:80],
        result["intent"].value,
        result["confidence"],
        result["layer"],
        fallback_reason,
        prior.value if prior else None,
        word_count,
        {k: round(v, 3) for k, v in sorted(all_scores.items(), key=lambda x: -x[1])},
    )

    return result


# ─── Public API ───────────────────────────────────────────────────────────────

def classify_intent(
    text: str,
    embedding: "list[float] | None" = None,
    history: "list[dict] | None" = None,
) -> dict:
    """Classify intent of a WhatsApp message.

    Returns: {intent: IntentEnum, confidence: float, layer: str}
    - embedding: pre-computed (pass from webhook to avoid a second Ollama call)
    - history: for INTENT-06 multi-turn inherit check
    No LLM calls, no ChromaDB queries — pure in-memory classification.
    """
    layer1 = _layer1_classify(text)
    if layer1 is not None:
        log.info(
            "INTENT_DEBUG | query=%r layer=rule intent=%s confidence=1.0",
            text[:80], layer1["intent"].value,
        )
        return layer1

    if not _intent_ready:
        log.warning(
            "INTENT_DEBUG | query=%r centroids_not_ready — degraded to GENERAL_QNA",
            text[:80],
        )
        return {"intent": IntentEnum.GENERAL_QNA, "confidence": 0.0, "layer": "fallback"}

    # Skip Layer 2 entirely when no embedding is provided — an all-zero vector would
    # produce meaningless scores and misleading debug log lines.
    if not embedding:
        return {"intent": IntentEnum.GENERAL_QNA, "confidence": 0.0, "layer": "fallback_no_embedding"}

    return _layer2_classify(embedding, history, text)


def compute_centroids() -> None:
    """Load exemplar phrases and compute per-intent centroid embeddings.

    Called by the startup orchestrator (step 0). Non-fatal on failure —
    sets _intent_ready=False so classify_intent() degrades to GENERAL_QNA.
    """
    global _centroids, _intent_ready
    try:
        import app.rag as _rag
        exemplars = json.loads(EXEMPLARS_PATH.read_text(encoding="utf-8"))
        new_centroids: dict[str, np.ndarray] = {}
        for intent_key, phrases in exemplars.items():
            vecs = [np.array(_rag.get_embedding(p), dtype=float) for p in phrases]
            new_centroids[intent_key] = np.mean(vecs, axis=0)
        _centroids = new_centroids
        _intent_ready = True
        log.info("INTENT | centroids computed for %d intents", len(_centroids))
    except Exception:
        log.warning("INTENT | centroid computation failed — degraded to GENERAL_QNA", exc_info=True)
        _intent_ready = False
