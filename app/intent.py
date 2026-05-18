"""
Intent classifier for Numobel WhatsApp chatbot.

Public API:
  - IntentEnum: 8-value enum of message intents
  - classify_intent(text, embedding=None, history=None) -> dict
  - compute_centroids() -> None  (called by startup orchestrator)
  - load_entity_index(brands, product_lines, product_names) -> None  (called by startup)

No LLM calls, no ChromaDB queries — pure in-memory classification.
Layer 1:   compiled regex rules (greeting, OOS, chitchat, brand names) — no embedding needed.
Layer 1.5: deterministic catalog entity pre-pass — checks query tokens against known brands,
           product line names, and product name tokens loaded from SQLite at startup.
Layer 2:   cosine similarity against per-intent centroid embeddings.
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

# ─── Layer 1.5 entity index ───────────────────────────────────────────────────
# Populated at startup by load_entity_index(). Each set contains lowercased values.
_entity_brand_names:        frozenset[str] = frozenset()
_entity_brand_tokens:       frozenset[str] = frozenset()  # individual tokens from brand names
_entity_product_line_names: frozenset[str] = frozenset()
_entity_product_tokens:     frozenset[str] = frozenset()
_entity_index_ready: bool = False

# Minimum character length for a token to be used in product-token matching.
# Prevents short noise words ("a", "is", "in") from triggering false positives.
_PRODUCT_TOKEN_MIN_LEN = 4


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

# Specific product intents — when the embedding returns a vague intent (brand_discovery)
# even at high confidence, a more-specific prior should override on short follow-up queries.
# This handles context-anchored follow-ups like "what colors do you have" after a product
# discussion, which embed close to brand_discovery but are actually about the prior product.
_SPECIFIC_PRODUCT_INTENTS: frozenset[IntentEnum] = frozenset({
    IntentEnum.BRAND_DEEP_DIVE,
    IntentEnum.PRODUCT_LINE_QUERY,
    IntentEnum.SPECIFIC_PRODUCT,
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

# Visual request phrases — "show me", "can you show me", "show me a photo", etc.
# These bypass brand_discovery_anchor (INTENT-14) and resolve directly to specific_product
# so the planner emits "media" and the user gets an image, not a text fallback.  (Bug 1 fix)
_VISUAL_REQUEST_RE = re.compile(
    r"(^(can (you )?)?show me|^(let me |can i |could i )?see (a |an )?(photo|image|picture|pic))",
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
    if _VISUAL_REQUEST_RE.search(t):
        return {"intent": IntentEnum.SPECIFIC_PRODUCT, "confidence": 1.0, "layer": "rule"}
    return None


def _layer1_5_classify(text: str) -> "dict | None":
    """Layer 1.5: deterministic catalog entity pre-pass.

    Checks query tokens against the entity index loaded from SQLite at startup.
    Returns result dict or None if no catalog entity is matched.

    Resolution order (most-specific first):
      1. Full phrase match against product line names → product_line_query
      2. Full phrase match against brand names (multi-word, e.g. 'rubio monocoat') → brand_deep_dive
         Note: single-word brand names are already caught by Layer 1 _BRAND_NAME_RE.
      3. Any query token (len >= _PRODUCT_TOKEN_MIN_LEN) found in product token index:
         3a. If ALL matched tokens are brand name tokens → brand_deep_dive
             (e.g. "what is rubio monocoat" matches "rubio"+"monocoat" — all brand tokens)
         3b. Otherwise → specific_product

    Skipped silently when entity index has not been populated (e.g., during tests that
    do not call load_entity_index — entity_index_ready stays False).
    """
    if not _entity_index_ready:
        return None

    t_lower = text.strip().lower()
    # Strip trailing punctuation for cleaner phrase matching
    t_clean = re.sub(r"[!?.]+$", "", t_lower).strip()

    # 1. Product line name full-phrase match (e.g. "on wheels", "building block")
    if t_clean in _entity_product_line_names:
        return {"intent": IntentEnum.PRODUCT_LINE_QUERY, "confidence": 1.0, "layer": "catalog_entity"}

    # 2. Brand name full-phrase match for multi-word brands not caught by Layer 1 regex
    if t_clean in _entity_brand_names:
        return {"intent": IntentEnum.BRAND_DEEP_DIVE, "confidence": 1.0, "layer": "catalog_entity"}

    # 3. Product token overlap — any meaningful token in query found in product index
    query_tokens = {
        tok for tok in re.split(r"[\s\-_/]+", t_clean)
        if len(tok) >= _PRODUCT_TOKEN_MIN_LEN
    }
    matched_tokens = query_tokens & _entity_product_tokens
    if matched_tokens:
        log.debug(
            "INTENT | layer1_5 matched product tokens=%s in query=%r",
            matched_tokens, text[:80],
        )
        # 3a. If ALL matched tokens are brand name tokens, this is a brand query, not a
        # specific product query.  Example: "what is rubio monocoat" matches "rubio" and
        # "monocoat" — both are brand tokens, not product-line or model tokens.
        if matched_tokens <= _entity_brand_tokens:
            log.debug(
                "INTENT | layer1_5 all matched tokens are brand tokens=%s — brand_deep_dive",
                matched_tokens,
            )
            return {"intent": IntentEnum.BRAND_DEEP_DIVE, "confidence": 1.0, "layer": "catalog_entity"}
        return {"intent": IntentEnum.SPECIFIC_PRODUCT, "confidence": 1.0, "layer": "catalog_entity"}

    return None


def _layer2_classify(
    embedding: list[float],
    history: "list[dict] | None",
    text: str,
) -> dict:
    """Layer 2: cosine centroid classification with optional multi-turn inherit.

    Inherit rule (INTENT-06 / INTENT-09 / INTENT-10 / INTENT-11 / INTENT-14):
      Standard inherit (tentative band only) — applied when ALL hold:
        1. best_sim is tentative (TENTATIVE_THRESHOLD ≤ best_sim < CONFIDENCE_THRESHOLD)
        2. word_count < INTENT_SHORT_MSG_TOKENS  (short, ambiguous message)
        3. prior is inheritable (not in _NON_INHERITABLE)
        4. One of:
           a. best_intent == prior  (embedding already agrees with prior direction), OR
           b. best_intent is non-product (chitchat/greeting/out_of_scope) — product prior wins, OR
           c. best_intent is a vague product intent (brand_discovery / general_qna) AND
              prior is NOT a vague product intent — more-specific prior wins.

      Strong context-anchor inherit (INTENT-14) — applied even at high confidence when:
        1. best_intent is brand_discovery  (a vague high-confidence match)
        2. prior is a specific product intent (brand_deep_dive / product_line_query /
           specific_product / general_qna)
        3. word_count < INTENT_SHORT_MSG_TOKENS  (short contextual follow-up)
        This handles queries like "what colors do you have" / "what sizes do you have"
        after a product discussion — they embed as brand_discovery but are clearly
        context-anchored follow-ups that should stay as general_qna for the prior product.
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

    # INTENT-14: Strong context-anchor — brand_discovery at ANY confidence level should
    # yield to a more-specific product prior on short queries.  A user asking "what colors
    # do you have" after discussing a specific product is clearly asking about that product,
    # not requesting a brand list.
    _is_brand_discovery_best  = best_intent == IntentEnum.BRAND_DISCOVERY
    _prior_is_product_specific = (
        inheritable_prior is not None
        and inheritable_prior in _SPECIFIC_PRODUCT_INTENTS
    )
    if (
        _is_brand_discovery_best
        and _prior_is_product_specific
        and word_count < INTENT_SHORT_MSG_TOKENS
    ):
        # Anchor to general_qna — keep the user in a product conversation rather than
        # sending them back to the top-level brand list.
        anchored = IntentEnum.GENERAL_QNA
        result = {"intent": anchored, "confidence": best_sim, "layer": "embedding_inherit"}
        fallback_reason = (
            f"brand_discovery_anchor(words={word_count},prior={inheritable_prior.value})"
        )
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

    Classification order:
      Layer 1   — compiled regex rules (greeting, OOS, chitchat, standalone brand names)
      Layer 1.5 — deterministic catalog entity pre-pass (product lines, brand phrases, product tokens)
      Layer 2   — cosine similarity against per-intent centroid embeddings
    """
    layer1 = _layer1_classify(text)
    if layer1 is not None:
        log.info(
            "INTENT_DEBUG | query=%r layer=rule intent=%s confidence=1.0",
            text[:80], layer1["intent"].value,
        )
        return layer1

    layer1_5 = _layer1_5_classify(text)
    if layer1_5 is not None:
        log.info(
            "INTENT_DEBUG | query=%r layer=catalog_entity intent=%s confidence=1.0",
            text[:80], layer1_5["intent"].value,
        )
        return layer1_5

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


def load_entity_index(
    brands: "list[str]",
    product_lines: "list[str]",
    product_names: "list[str]",
) -> None:
    """Populate the catalog entity index for Layer 1.5 classification.

    Called by the startup orchestrator (step 1.5) after SQLite sync completes.
    Non-fatal on failure — _entity_index_ready stays False and Layer 1.5 is skipped.

    Args:
        brands:        Brand name strings (e.g. ['Rubio Monocoat', 'Nutoy', ...])
        product_lines: Product line name strings (e.g. ['On Wheels', 'Stacker', ...])
        product_names: Full product name strings (e.g. ['Nutoy-On Wheels-Duck', ...])
    """
    global _entity_brand_names, _entity_brand_tokens, _entity_product_line_names
    global _entity_product_tokens, _entity_index_ready
    try:
        brand_set = frozenset(b.lower().strip() for b in brands if b)
        line_set  = frozenset(pl.lower().strip() for pl in product_lines if pl)

        # Extract individual tokens from brand names for brand-token matching in step 3a.
        # This lets "what is rubio monocoat" match brand tokens "rubio"+"monocoat" → brand_deep_dive.
        brand_token_set: set[str] = set()
        for brand in brands:
            if not brand:
                continue
            for tok in re.split(r"[\s\-_/]+", brand.lower()):
                if len(tok) >= _PRODUCT_TOKEN_MIN_LEN:
                    brand_token_set.add(tok)

        # Extract individual tokens from product names for token-level matching.
        # Split on word boundaries, hyphens, and slashes; filter by minimum length
        # to avoid noise (e.g. "a", "in", "of").
        token_set: set[str] = set()
        for name in product_names:
            if not name:
                continue
            for tok in re.split(r"[\s\-_/]+", name.lower()):
                if len(tok) >= _PRODUCT_TOKEN_MIN_LEN:
                    token_set.add(tok)

        _entity_brand_names        = brand_set
        _entity_brand_tokens       = frozenset(brand_token_set)
        _entity_product_line_names = line_set
        _entity_product_tokens     = frozenset(token_set)
        _entity_index_ready        = True

        log.info(
            "INTENT | entity index loaded: %d brands, %d brand tokens, "
            "%d product lines, %d product tokens",
            len(brand_set), len(brand_token_set), len(line_set), len(token_set),
        )
    except Exception:
        log.warning("INTENT | entity index load failed — Layer 1.5 disabled", exc_info=True)
        _entity_index_ready = False


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
