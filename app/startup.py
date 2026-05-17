"""
Startup orchestrator for Numobel RAG chatbot.

Validates required env vars eagerly, then runs SQLite sync, ChromaDB product
ingest, and QnA migration in a background daemon thread. Exposes `_ready` for
the webhook readiness gate and health endpoint.
"""

import os
import threading

import app.config as config
from app.config import PRODUCTS_COLLECTION, QNA_COLLECTION
from app.log import get_logger

log = get_logger()

_ready: bool = False
_REQUIRED_ENV_VARS = ["WA2MATION_API_KEY", "WA2MATION_VENDOR_UID"]


def validate_env() -> None:
    """Raise RuntimeError if any required env var is missing."""
    missing = [var for var in _REQUIRED_ENV_VARS if not os.getenv(var)]
    if missing:
        raise RuntimeError(
            f"Missing required environment variables: {', '.join(missing)}. "
            "Check your .env file."
        )


def _migrate_qna_if_needed() -> None:
    """Migrate approved.jsonl into the approved QnA collection once."""
    from app.db import get_meta, set_meta

    if get_meta("qna_migration_done") == "1":
        log.info("STARTUP | QnA migration already done; skipping")
        return

    if not config.APPROVED_PATH.exists():
        log.warning(
            "STARTUP | %s not found; marking migration done with 0 pairs",
            config.APPROVED_PATH,
        )
        set_meta("qna_migration_done", "1")
        return

    from app.refinement.storage import load_jsonl
    from app.rag import get_collection, ingest_qna_pair

    pairs = load_jsonl(config.APPROVED_PATH)
    qna_collection = get_collection(QNA_COLLECTION)
    count = 0
    for pair in pairs:
        try:
            ingest_qna_pair(pair, qna_collection)
            count += 1
        except Exception as exc:
            log.warning("STARTUP | QnA pair migration failed: %s", exc)
    set_meta("qna_migration_done", "1")
    log.info("STARTUP | migrated %d QnA pairs into %s", count, QNA_COLLECTION)


def _startup_orchestrator() -> None:
    """Run startup work in dependency order and set _ready only on success."""
    global _ready
    try:
        # Step 0 has its own try/except — failure is non-fatal
        try:
            log.info("STARTUP | step 0/4 - computing intent centroids")
            from app.intent import compute_centroids
            compute_centroids()
        except Exception:
            log.warning(
                "STARTUP | step 0 failed — intent classifier degraded to GENERAL_QNA",
                exc_info=True,
            )

        log.info("STARTUP | step 1/4 - SQLite sync")
        from app.db import sync_products

        sync_result = sync_products()
        log.info(
            "STARTUP | step 1 result: %s",
            {k: v for k, v in sync_result.items() if k in ("added", "updated", "deleted")},
        )

        # Step 1.5: Build system prompt and initialize hallucination validator
        log.info("STARTUP | step 1.5/4 - building authorized system prompt + validator init")
        try:
            from app.db import get_db
            import app.rag as _rag
            from app.validators.hallucination import initialize_validator
            rows   = get_db().execute("SELECT name FROM brands").fetchall()
            brands = [row["name"] for row in rows]
            _rag.SYSTEM_PROMPT = _rag.build_system_prompt(brands)
            initialize_validator(brands)
            log.info(
                "STARTUP | system prompt built with %d authorized brands: %s",
                len(brands), brands,
            )
        except Exception:
            log.warning(
                "STARTUP | step 1.5 failed — system prompt and validator may use defaults",
                exc_info=True,
            )

        log.info("STARTUP | step 2/4 - ChromaDB product ingest into %s", PRODUCTS_COLLECTION)
        from app.rag import get_collection, ingest_data

        products_collection = get_collection(PRODUCTS_COLLECTION)
        deleted_ids = sync_result.get("deleted_chunk_ids") or []
        if deleted_ids:
            try:
                products_collection.delete(ids=deleted_ids)
                log.info("STARTUP | purged %d chunks of deleted products", len(deleted_ids))
            except Exception as exc:
                log.warning("STARTUP | failed to purge deleted chunks: %s", exc)

        if products_collection.count() == 0 or sync_result.get("changed_product_names"):
            ingest_data(products_collection)

        log.info("STARTUP | step 3/4 - QnA migration check")
        _migrate_qna_if_needed()

        _ready = True
        log.info("STARTUP | orchestrator complete; _ready=True")
    except Exception:
        log.critical(
            "STARTUP | orchestrator failed - bot degraded, _ready stays False",
            exc_info=True,
        )


def run_startup() -> None:
    """Validate env eagerly, then launch the background orchestrator thread."""
    validate_env()
    thread = threading.Thread(
        target=_startup_orchestrator,
        daemon=True,
        name="numobel-startup",
    )
    thread.start()
    log.info("STARTUP | background orchestrator thread started")
