"""
SQLite product catalog for Numobel.

Lazy singleton connection (check_same_thread=False, WAL mode).
Provides schema creation, product sync, content hash, and meta key-value helpers.
All writes committed within sync_products(); Flask threads only read.
"""

import hashlib
import json
import sqlite3

import app.config as config
from app.config import SQLITE_PATH
from app.log import get_logger

log = get_logger()

_conn: sqlite3.Connection | None = None

_SCHEMA = """
CREATE TABLE IF NOT EXISTS app_meta (
    key   TEXT PRIMARY KEY,
    value TEXT NOT NULL
);

CREATE TABLE IF NOT EXISTS brands (
    id          INTEGER PRIMARY KEY AUTOINCREMENT,
    name        TEXT NOT NULL UNIQUE,
    description TEXT
);

CREATE TABLE IF NOT EXISTS product_lines (
    id       INTEGER PRIMARY KEY AUTOINCREMENT,
    brand_id INTEGER NOT NULL REFERENCES brands(id),
    name     TEXT NOT NULL,
    UNIQUE(brand_id, name)
);

CREATE TABLE IF NOT EXISTS products (
    id               INTEGER PRIMARY KEY AUTOINCREMENT,
    name             TEXT NOT NULL,
    brand_id         INTEGER REFERENCES brands(id),
    product_line_id  INTEGER REFERENCES product_lines(id),
    description      TEXT,
    price_original   REAL,
    price_discounted REAL,
    sku              TEXT,
    product_link     TEXT,
    attributes       TEXT,
    content_hash     TEXT NOT NULL,
    chroma_chunk_ids TEXT NOT NULL DEFAULT '[]'
);

CREATE TABLE IF NOT EXISTS product_images (
    id         INTEGER PRIMARY KEY AUTOINCREMENT,
    product_id INTEGER NOT NULL REFERENCES products(id) ON DELETE CASCADE,
    url        TEXT NOT NULL,
    position   INTEGER NOT NULL DEFAULT 0
);
"""


def _create_schema(conn: sqlite3.Connection) -> None:
    conn.executescript(_SCHEMA)
    conn.commit()


def get_db() -> sqlite3.Connection:
    """Return lazy singleton SQLite connection (WAL, FK enforcement, Row factory)."""
    global _conn
    if _conn is None:
        from app.config import SQLITE_PATH as _path

        _path.parent.mkdir(parents=True, exist_ok=True)
        _conn = sqlite3.connect(str(_path), check_same_thread=False)
        _conn.execute("PRAGMA journal_mode=WAL")
        _conn.execute("PRAGMA foreign_keys=ON")
        _conn.row_factory = sqlite3.Row
        _create_schema(_conn)
        log.info("DB | connection initialised at %s", _path)
    return _conn


def _content_hash(product: dict) -> str:
    """MD5 over fields that affect RAG text or retrieval metadata."""
    relevant = {
        "name": product.get("name") or "",
        "brand": product.get("brand") or "",
        "product_line": product.get("product_line") or "",
        "description": product.get("description") or "",
        "attributes": product.get("attributes") or {},
        "price": product.get("price") or {},
        "media": (product.get("media") or {}).get("images") or [],
        "product_link": (product.get("metadata") or {}).get("product_link") or "",
    }
    return hashlib.md5(
        json.dumps(relevant, sort_keys=True).encode("utf-8")
    ).hexdigest()


def get_meta(key: str) -> str | None:
    row = get_db().execute(
        "SELECT value FROM app_meta WHERE key = ?", (key,)
    ).fetchone()
    return row["value"] if row else None


def set_meta(key: str, value: str) -> None:
    db = get_db()
    db.execute(
        "INSERT OR REPLACE INTO app_meta(key, value) VALUES (?, ?)",
        (key, value),
    )
    db.commit()


def _upsert_brand(db: sqlite3.Connection, brand_name: str) -> int:
    db.execute("INSERT OR IGNORE INTO brands(name) VALUES (?)", (brand_name,))
    return db.execute(
        "SELECT id FROM brands WHERE name = ?", (brand_name,)
    ).fetchone()["id"]


def _upsert_product_line(
    db: sqlite3.Connection, brand_id: int, line_name: str | None
) -> int | None:
    if not line_name:
        return None
    db.execute(
        "INSERT OR IGNORE INTO product_lines(brand_id, name) VALUES (?, ?)",
        (brand_id, line_name),
    )
    return db.execute(
        "SELECT id FROM product_lines WHERE brand_id = ? AND name = ?",
        (brand_id, line_name),
    ).fetchone()["id"]


def _replace_images(db: sqlite3.Connection, product_id: int, image_urls: list[str]) -> None:
    db.execute("DELETE FROM product_images WHERE product_id = ?", (product_id,))
    for pos, url in enumerate(image_urls or []):
        db.execute(
            "INSERT INTO product_images(product_id, url, position) VALUES (?, ?, ?)",
            (product_id, url, pos),
        )


def sync_products() -> dict:
    """Sync clean_products.json into SQLite and return ChromaDB cleanup hints."""
    log.info("DB | sync_products starting from %s", config.DATA_FILE)
    with open(config.DATA_FILE, encoding="utf-8") as f:
        products = json.load(f)

    db = get_db()
    seen_names: set[str] = set()
    added = updated = deleted = 0
    changed_product_names: list[str] = []
    deleted_chunk_ids: list[str] = []

    for product in products:
        name = (product.get("name") or "").strip()
        if not name:
            continue
        seen_names.add(name)

        content_hash = _content_hash(product)
        brand_id = _upsert_brand(db, product.get("brand") or "Unknown")
        line_id = _upsert_product_line(db, brand_id, product.get("product_line"))

        price = product.get("price") or {}
        price_orig = price.get("original")
        price_disc = price.get("discounted")
        metadata = product.get("metadata") or {}
        sku = metadata.get("sku")
        product_link = metadata.get("product_link")
        description = product.get("description") or ""
        attrs_json = json.dumps(product.get("attributes") or {}, sort_keys=True)
        images = (product.get("media") or {}).get("images") or []

        existing = db.execute(
            "SELECT id, content_hash FROM products WHERE name = ?", (name,)
        ).fetchone()

        if existing is None:
            cur = db.execute(
                """INSERT INTO products(name, brand_id, product_line_id, description,
                                        price_original, price_discounted, sku, product_link,
                                        attributes, content_hash, chroma_chunk_ids)
                   VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, '[]')""",
                (
                    name,
                    brand_id,
                    line_id,
                    description,
                    price_orig,
                    price_disc,
                    sku,
                    product_link,
                    attrs_json,
                    content_hash,
                ),
            )
            product_id = cur.lastrowid
            _replace_images(db, product_id, images)
            added += 1
            changed_product_names.append(name)
        elif existing["content_hash"] != content_hash:
            db.execute(
                """UPDATE products SET brand_id=?, product_line_id=?, description=?,
                                       price_original=?, price_discounted=?, sku=?,
                                       product_link=?, attributes=?, content_hash=?
                   WHERE id=?""",
                (
                    brand_id,
                    line_id,
                    description,
                    price_orig,
                    price_disc,
                    sku,
                    product_link,
                    attrs_json,
                    content_hash,
                    existing["id"],
                ),
            )
            _replace_images(db, existing["id"], images)
            updated += 1
            changed_product_names.append(name)
        else:
            db.execute(
                """UPDATE products SET brand_id=?, product_line_id=?,
                                       price_original=?, price_discounted=?, sku=?,
                                       product_link=?
                   WHERE id=?""",
                (
                    brand_id,
                    line_id,
                    price_orig,
                    price_disc,
                    sku,
                    product_link,
                    existing["id"],
                ),
            )
            _replace_images(db, existing["id"], images)

    existing_rows = db.execute(
        "SELECT id, name, chroma_chunk_ids FROM products"
    ).fetchall()
    for row in existing_rows:
        if row["name"] not in seen_names:
            try:
                deleted_chunk_ids.extend(json.loads(row["chroma_chunk_ids"] or "[]"))
            except json.JSONDecodeError:
                log.warning("DB | invalid chroma_chunk_ids JSON for product id=%s", row["id"])
            db.execute("DELETE FROM products WHERE id = ?", (row["id"],))
            deleted += 1

    db.commit()
    log.info("DB | sync complete: added=%d updated=%d deleted=%d", added, updated, deleted)
    return {
        "added": added,
        "updated": updated,
        "deleted": deleted,
        "deleted_chunk_ids": deleted_chunk_ids,
        "changed_product_names": changed_product_names,
    }


def set_chroma_chunk_ids(product_name: str, chunk_ids: list[str]) -> None:
    """Persist the ChromaDB chunk IDs for a product after embedding."""
    db = get_db()
    cur = db.execute(
        "UPDATE products SET chroma_chunk_ids = ? WHERE name = ?",
        (json.dumps(chunk_ids), product_name),
    )
    if cur.rowcount != 1:
        raise RuntimeError(f"Product not found while saving Chroma chunk IDs: {product_name}")
    db.commit()
