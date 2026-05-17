"""
Intent classifier for Numobel WhatsApp chatbot.

Public API:
  - IntentEnum: 8-value enum of message intents
  - classify_intent(text, embedding=None, history=None) -> dict
  - compute_centroids() -> None  (called by startup orchestrator)

No LLM calls, no ChromaDB queries — pure in-memory classification.
Layer 1: compiled regex rules (greeting, OOS, chitchat) — no embedding needed.
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
    return None


def _layer2_classify(
    embedding: list[float],
    history: "list[dict] | None",
    text: str,
) -> dict:
    """Layer 2: cosine centroid classification with optional multi-turn inherit."""
    vec = np.array(embedding, dtype=float)
    best_intent = IntentEnum.GENERAL_QNA
    best_sim = -1.0

    for intent_key, centroid in _centroids.items():
        sim = _cosine_similarity(vec, centroid)
        if sim > best_sim:
            best_sim = sim
            try:
                best_intent = IntentEnum(intent_key)
            except ValueError:
                best_intent = IntentEnum.GENERAL_QNA

    if best_sim >= INTENT_CONFIDENCE_THRESHOLD:
        return {"intent": best_intent, "confidence": best_sim, "layer": "embedding"}

    if best_sim >= INTENT_TENTATIVE_THRESHOLD:
        word_count = len(text.split())
        if word_count < INTENT_SHORT_MSG_TOKENS:
            prior = _prior_intent(history or [])
            if prior:
                return {"intent": prior, "confidence": best_sim, "layer": "embedding_inherit"}
        return {"intent": best_intent, "confidence": best_sim, "layer": "embedding"}

    return {"intent": IntentEnum.GENERAL_QNA, "confidence": best_sim, "layer": "embedding"}


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
        return layer1

    if not _intent_ready:
        return {"intent": IntentEnum.GENERAL_QNA, "confidence": 0.0, "layer": "fallback"}

    return _layer2_classify(embedding or [], history, text)


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
