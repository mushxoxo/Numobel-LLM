"""Tests for app/db.py - SQLite product catalog."""

import json

import pytest


SAMPLE_PRODUCT = {
    "name": "Test Stacker",
    "brand": "Nutoy",
    "product_line": "Stacker",
    "description": "A test stacker product.",
    "attributes": {"colors": ["red", "blue"]},
    "price": {"original": 999, "discounted": 799},
    "media": {"images": ["https://example.com/img1.jpg", "https://example.com/img2.jpg"]},
    "metadata": {"sku": "TST-001", "product_link": "https://example.com/p/test"},
}

SAMPLE_PRODUCT_NO_LINE = {
    "name": "Rubio Oil 100",
    "brand": "Rubio Monocoat",
    "product_line": None,
    "description": "Hardwax oil.",
    "attributes": {},
    "price": {"original": 1500},
    "media": {"images": []},
    "metadata": {},
}


@pytest.fixture
def db_env(tmp_path, monkeypatch):
    """Isolate SQLite singleton and data file per test."""
    import app.config as cfg
    import app.db as db_module

    data_file = tmp_path / "clean_products.json"
    data_file.write_text(json.dumps([SAMPLE_PRODUCT, SAMPLE_PRODUCT_NO_LINE]))
    monkeypatch.setattr(cfg, "DATA_FILE", data_file)
    monkeypatch.setattr(cfg, "SQLITE_PATH", tmp_path / "test.db")
    db_module._conn = None
    yield {"data_file": data_file, "module": db_module}
    db_module._conn = None


def test_schema_creates_all_tables(db_env):
    conn = db_env["module"].get_db()
    tables = {r[0] for r in conn.execute("SELECT name FROM sqlite_master WHERE type='table'")}
    assert {"app_meta", "brands", "product_lines", "products", "product_images"} <= tables


def test_sync_products_inserts_all_fields(db_env):
    result = db_env["module"].sync_products()
    assert result["added"] == 2
    conn = db_env["module"].get_db()
    row = conn.execute("SELECT * FROM products WHERE name = ?", ("Test Stacker",)).fetchone()
    assert row is not None
    assert row["description"] == "A test stacker product."
    assert row["price_original"] == 999
    assert row["price_discounted"] == 799
    assert row["sku"] == "TST-001"
    assert row["product_link"] == "https://example.com/p/test"
    assert json.loads(row["attributes"]) == {"colors": ["red", "blue"]}

    brand = conn.execute("SELECT name FROM brands WHERE id = ?", (row["brand_id"],)).fetchone()
    assert brand["name"] == "Nutoy"
    line = conn.execute("SELECT name FROM product_lines WHERE id = ?", (row["product_line_id"],)).fetchone()
    assert line["name"] == "Stacker"

    images = conn.execute(
        "SELECT url, position FROM product_images WHERE product_id = ? ORDER BY position",
        (row["id"],),
    ).fetchall()
    assert len(images) == 2
    assert images[0]["url"] == "https://example.com/img1.jpg"
    assert images[0]["position"] == 0

    rubio = conn.execute(
        "SELECT product_line_id FROM products WHERE name = ?", ("Rubio Oil 100",)
    ).fetchone()
    assert rubio["product_line_id"] is None


def test_incremental_sync_idempotent(db_env):
    module = db_env["module"]
    first = module.sync_products()
    second = module.sync_products()
    assert first["added"] == 2
    assert second == {
        "added": 0,
        "updated": 0,
        "deleted": 0,
        "deleted_chunk_ids": [],
        "changed_product_names": [],
    }


def test_sync_hard_deletes_removed_products(db_env):
    module = db_env["module"]
    module.sync_products()
    module.set_chroma_chunk_ids("Test Stacker", ["chunk_aaa_0", "chunk_aaa_1"])
    db_env["data_file"].write_text(json.dumps([SAMPLE_PRODUCT_NO_LINE]))

    result = module.sync_products()

    assert result["deleted"] == 1
    assert set(result["deleted_chunk_ids"]) == {"chunk_aaa_0", "chunk_aaa_1"}
    conn = module.get_db()
    assert conn.execute(
        "SELECT COUNT(*) FROM products WHERE name = ?", ("Test Stacker",)
    ).fetchone()[0] == 0
    assert conn.execute("SELECT COUNT(*) FROM product_images").fetchone()[0] == 0


def test_set_chroma_chunk_ids_raises_for_missing_product(db_env):
    module = db_env["module"]
    module.get_db()
    with pytest.raises(RuntimeError, match="Product not found"):
        module.set_chroma_chunk_ids("Missing Product", ["chunk_1"])


def test_content_hash_tracks_price_change(db_env):
    module = db_env["module"]
    module.sync_products()
    new_products = json.loads(db_env["data_file"].read_text())
    new_products[0]["price"]["original"] = 1234
    new_products[0]["price"]["discounted"] = 999
    db_env["data_file"].write_text(json.dumps(new_products))

    result = module.sync_products()

    assert result["updated"] == 1
    assert "Test Stacker" in result["changed_product_names"]
    row = module.get_db().execute(
        "SELECT price_original FROM products WHERE name = ?", ("Test Stacker",)
    ).fetchone()
    assert row["price_original"] == 1234


def test_content_hash_tracks_image_and_link_change(db_env):
    module = db_env["module"]
    module.sync_products()
    new_products = json.loads(db_env["data_file"].read_text())
    new_products[0]["media"]["images"] = ["https://example.com/new.jpg"]
    new_products[0]["metadata"]["product_link"] = "https://example.com/p/new"
    db_env["data_file"].write_text(json.dumps(new_products))

    result = module.sync_products()

    assert result["updated"] == 1
    assert "Test Stacker" in result["changed_product_names"]


def test_content_hash_differs_on_description_change(db_env):
    module = db_env["module"]
    module.sync_products()
    new_products = json.loads(db_env["data_file"].read_text())
    new_products[0]["description"] = "completely rewritten product description"
    db_env["data_file"].write_text(json.dumps(new_products))

    result = module.sync_products()

    assert result["updated"] == 1
    assert "Test Stacker" in result["changed_product_names"]


def test_meta_helpers(db_env):
    module = db_env["module"]
    module.get_db()
    assert module.get_meta("nonexistent") is None
    module.set_meta("qna_migration_done", "1")
    assert module.get_meta("qna_migration_done") == "1"
    module.set_meta("qna_migration_done", "0")
    assert module.get_meta("qna_migration_done") == "0"
