import json
import sys
import os
from pathlib import Path
from unittest.mock import MagicMock, patch, call

import pytest

# Make training/ importable without requiring __init__.py
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'training'))
import approve_qna_cli as cli


# ── helpers ─────────────────────────────────────────────────────────────────

def _make_pair(question="What is X?", answer="X is great.", approved=False, message_type="text"):
    return {
        "question": question, "answer": answer, "message_type": message_type,
        "buttons": None, "image_url": None, "product": "TestProduct",
        "approved": approved,
    }

def _write_jsonl(path: Path, pairs: list[dict]) -> None:
    path.write_text('\n'.join(json.dumps(p) for p in pairs) + '\n')


# ── test 1: load_unapproved returns only unapproved pairs ───────────────────

def test_load_unapproved_filters_approved(tmp_path, monkeypatch):
    pending = tmp_path / 'pending.jsonl'
    _write_jsonl(pending, [
        _make_pair(question="Q1?", approved=False),
        _make_pair(question="Q2?", approved=True),
        _make_pair(question="Q3?", approved=False),
    ])
    monkeypatch.setattr(cli, '_PENDING_PATH', pending)
    result = cli.load_unapproved()
    assert len(result) == 2
    assert all(not p['approved'] for p in result)


# ── test 2: load_unapproved skips malformed lines ───────────────────────────

def test_load_unapproved_skips_malformed(tmp_path, monkeypatch):
    pending = tmp_path / 'pending.jsonl'
    pending.write_text(
        '{"question":"Q1?","answer":"A.","message_type":"text","approved":false,"product":"P"}\n'
        'CORRUPT LINE\n'
        '{"question":"Q2?","answer":"A.","message_type":"text","approved":false,"product":"P"}\n'
    )
    monkeypatch.setattr(cli, '_PENDING_PATH', pending)
    result = cli.load_unapproved()
    assert len(result) == 2


# ── test 3: [A] approve triggers all three writes + ingest ──────────────────

def test_approve_action_calls_all_three_operations(tmp_path, monkeypatch):
    pending = tmp_path / 'pending.jsonl'
    approved_file = tmp_path / 'approved.jsonl'
    pair = _make_pair()
    _write_jsonl(pending, [pair])

    monkeypatch.setattr(cli, '_PENDING_PATH', pending)
    monkeypatch.setattr(cli, '_APPROVED_PATH', approved_file)

    mock_collection = MagicMock()
    inputs = iter(['a'])
    monkeypatch.setattr('builtins.input', lambda *_: next(inputs))

    with patch.object(cli.rag, 'ingest_qna_pair') as mock_ingest:
        approved_c, skipped_c = cli.run_approval_loop([pair], mock_collection)

    assert approved_c == 1
    assert skipped_c == 0
    mock_ingest.assert_called_once()
    # approved.jsonl should have the pair with approved=True
    saved = cli.load_jsonl(approved_file)
    assert len(saved) == 1
    assert saved[0]['approved'] is True
    # pending.jsonl should have approved=True
    pending_data = cli.load_jsonl(pending)
    assert pending_data[0]['approved'] is True


# ── test 4: [K] skip increments skipped_count, does not call ingest ─────────

def test_skip_action_increments_skipped_only(tmp_path, monkeypatch):
    pending = tmp_path / 'pending.jsonl'
    pair = _make_pair()
    _write_jsonl(pending, [pair])
    monkeypatch.setattr(cli, '_PENDING_PATH', pending)

    mock_collection = MagicMock()
    inputs = iter(['k'])
    monkeypatch.setattr('builtins.input', lambda *_: next(inputs))

    with patch.object(cli.rag, 'ingest_qna_pair') as mock_ingest:
        approved_c, skipped_c = cli.run_approval_loop([pair], mock_collection)

    assert approved_c == 0
    assert skipped_c == 1
    mock_ingest.assert_not_called()


# ── test 5: [Q] quit exits immediately ──────────────────────────────────────

def test_quit_action_exits_loop_immediately(tmp_path, monkeypatch):
    pending = tmp_path / 'pending.jsonl'
    pairs = [_make_pair(question=f"Q{i}?") for i in range(3)]
    _write_jsonl(pending, pairs)
    monkeypatch.setattr(cli, '_PENDING_PATH', pending)

    mock_collection = MagicMock()
    inputs = iter(['q'])
    monkeypatch.setattr('builtins.input', lambda *_: next(inputs))

    with patch.object(cli.rag, 'ingest_qna_pair') as mock_ingest:
        approved_c, skipped_c = cli.run_approval_loop(pairs, mock_collection)

    assert approved_c == 0
    assert skipped_c == 0
    mock_ingest.assert_not_called()


# ── test 6: [S] suggest refines pair and writes back to pending ─────────────

def test_suggest_refines_and_writes_back_then_approve(tmp_path, monkeypatch):
    pending = tmp_path / 'pending.jsonl'
    approved_file = tmp_path / 'approved.jsonl'
    pair = _make_pair()
    _write_jsonl(pending, [pair])
    monkeypatch.setattr(cli, '_PENDING_PATH', pending)
    monkeypatch.setattr(cli, '_APPROVED_PATH', approved_file)

    mock_collection = MagicMock()
    call_count = 0
    def fake_input(prompt=""):
        nonlocal call_count
        call_count += 1
        if call_count == 1:
            return 's'
        elif call_count == 2:
            return 'make it shorter'
        else:
            return 'a'
    monkeypatch.setattr('builtins.input', fake_input)

    refined_pair = {**pair, 'answer': 'Refined answer.'}
    with patch.object(cli, 'refine_with_llm', return_value=refined_pair) as mock_refine, \
         patch.object(cli.rag, 'ingest_qna_pair'):
        cli.run_approval_loop([pair], mock_collection)

    mock_refine.assert_called_once()
    # refined answer should be written to pending
    pending_data = cli.load_jsonl(pending)
    assert pending_data[0]['answer'] == 'Refined answer.'


# ── test 7: [S] suggest with LLM failure falls back gracefully ──────────────

def test_suggest_llm_failure_falls_back(tmp_path, monkeypatch):
    pending = tmp_path / 'pending.jsonl'
    pair = _make_pair()
    _write_jsonl(pending, [pair])
    monkeypatch.setattr(cli, '_PENDING_PATH', pending)

    mock_collection = MagicMock()
    call_count = 0
    def fake_input(prompt=""):
        nonlocal call_count
        call_count += 1
        if call_count == 1:
            return 's'
        elif call_count == 2:
            return 'a suggestion'
        else:
            return 'k'
    monkeypatch.setattr('builtins.input', fake_input)

    with patch.object(cli, 'refine_with_llm', side_effect=RuntimeError("Ollama down")), \
         patch.object(cli.rag, 'ingest_qna_pair') as mock_ingest:
        approved_c, skipped_c = cli.run_approval_loop([pair], mock_collection)

    mock_ingest.assert_not_called()
    assert skipped_c == 1


# ── test 8: main() calls ingest_data when collection is empty ───────────────

def test_main_ingests_data_when_collection_empty(tmp_path, monkeypatch, capsys):
    pending = tmp_path / 'pending.jsonl'
    monkeypatch.setattr(cli, '_PENDING_PATH', pending)

    mock_collection = MagicMock()
    mock_collection.count.return_value = 0

    with patch.object(cli.rag, 'get_collection', return_value=mock_collection), \
         patch.object(cli.rag, 'ingest_data') as mock_ingest:
        cli.main()

    mock_ingest.assert_called_once_with(mock_collection)
    out = capsys.readouterr().out
    assert 'generate_qna' in out or 'No unapproved' in out


# ── test 9: main() prints actionable error when no pending pairs ─────────────

def test_main_prints_actionable_error_when_no_pairs(tmp_path, monkeypatch, capsys):
    pending = tmp_path / 'pending.jsonl'
    monkeypatch.setattr(cli, '_PENDING_PATH', pending)

    mock_collection = MagicMock()
    mock_collection.count.return_value = 5

    with patch.object(cli.rag, 'get_collection', return_value=mock_collection):
        cli.main()

    out = capsys.readouterr().out
    assert 'generate_qna' in out
