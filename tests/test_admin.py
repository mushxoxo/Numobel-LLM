import json
import pytest
from datetime import datetime, timedelta
from unittest.mock import patch, MagicMock, mock_open

import app.admin as admin_module

ADMIN_PHONE = "919999999999"
NON_ADMIN   = "911234567890"
CONFIG      = {"admin_phones": [ADMIN_PHONE], "admin_timeout_minutes": 15}

SAMPLE_PAIR = {
    "question": "What is a stacker?",
    "answer":   "A wooden stacking toy.",
    "message_type": "carousel",
    "buttons":  ["Stackers"],
    "product":  "Nutoy-Stacker-Mountain",
}


@pytest.fixture(autouse=True)
def reset_sessions():
    """Wipe in-memory admin sessions between every test."""
    admin_module._sessions.clear()
    yield
    admin_module._sessions.clear()


@pytest.fixture
def mock_config():
    with patch("app.admin._load_config", return_value=CONFIG):
        yield


def _active_session(state, pair=None):
    return {"state": state, "current_pair": pair, "last_active": datetime.utcnow()}


# ─── is_admin ─────────────────────────────────────────────────────────────────

def test_is_admin_recognises_admin_phone(mock_config):
    assert admin_module.is_admin(ADMIN_PHONE) is True

def test_is_admin_rejects_non_admin(mock_config):
    assert admin_module.is_admin(NON_ADMIN) is False

def test_is_admin_strips_leading_plus(mock_config):
    assert admin_module.is_admin("+" + ADMIN_PHONE) is True


# ─── needs_admin_handling ─────────────────────────────────────────────────────

def test_needs_handling_admin_on(mock_config):
    assert admin_module.needs_admin_handling(ADMIN_PHONE, ":admin on") is True

def test_needs_handling_admin_off(mock_config):
    assert admin_module.needs_admin_handling(ADMIN_PHONE, ":admin off") is True

def test_needs_handling_train_command(mock_config):
    assert admin_module.needs_admin_handling(ADMIN_PHONE, ":train") is True

def test_needs_handling_non_admin_always_false(mock_config):
    assert admin_module.needs_admin_handling(NON_ADMIN, ":admin on") is False

def test_needs_handling_regular_message_without_session(mock_config):
    assert admin_module.needs_admin_handling(ADMIN_PHONE, "hello") is False

def test_needs_handling_regular_message_with_active_session(mock_config):
    admin_module._sessions[ADMIN_PHONE] = _active_session("menu")
    assert admin_module.needs_admin_handling(ADMIN_PHONE, "hello") is True


# ─── :admin on ───────────────────────────────────────────────────────────────

def test_admin_on_sets_menu_state_and_sends_menu(mock_config):
    with patch("app.admin.send_interactive") as mock_inter, \
         patch("app.admin.send_text"):
        admin_module.handle_admin(ADMIN_PHONE, ":admin on", MagicMock())
    assert admin_module._sessions[ADMIN_PHONE]["state"] == "menu"
    mock_inter.assert_called_once()
    call_kwargs = mock_inter.call_args[1]
    assert "QnA Review" in call_kwargs["buttons"]


# ─── :admin off ──────────────────────────────────────────────────────────────

def test_admin_off_clears_session(mock_config):
    admin_module._sessions[ADMIN_PHONE] = _active_session("menu")
    with patch("app.admin.send_text") as mock_text:
        admin_module.handle_admin(ADMIN_PHONE, ":admin off", MagicMock())
    assert ADMIN_PHONE not in admin_module._sessions
    mock_text.assert_called_once()


# ─── QnA Review / :train ─────────────────────────────────────────────────────

def test_qna_review_loads_pending_pair(mock_config):
    admin_module._sessions[ADMIN_PHONE] = _active_session("menu")
    with patch("app.admin._load_next_pending", return_value=SAMPLE_PAIR), \
         patch("app.admin.send_text"), \
         patch("app.admin.send_interactive"):
        admin_module.handle_admin(ADMIN_PHONE, "QnA Review", MagicMock())
    session = admin_module._sessions[ADMIN_PHONE]
    assert session["state"] == "training_review"
    assert session["current_pair"] == SAMPLE_PAIR

def test_train_shortcut_same_as_qna_review(mock_config):
    admin_module._sessions[ADMIN_PHONE] = _active_session("menu")
    with patch("app.admin._load_next_pending", return_value=SAMPLE_PAIR), \
         patch("app.admin.send_text"), \
         patch("app.admin.send_interactive"):
        admin_module.handle_admin(ADMIN_PHONE, ":train", MagicMock())
    assert admin_module._sessions[ADMIN_PHONE]["state"] == "training_review"

def test_qna_review_no_pending_sends_message(mock_config):
    admin_module._sessions[ADMIN_PHONE] = _active_session("menu")
    with patch("app.admin._load_next_pending", return_value=None), \
         patch("app.admin.send_text") as mock_text, \
         patch("app.admin.send_interactive"):
        admin_module.handle_admin(ADMIN_PHONE, ":train", MagicMock())
    text_sent = mock_text.call_args[0][1]
    assert "No pending" in text_sent


# ─── Approve ─────────────────────────────────────────────────────────────────

def test_approve_ingests_pair_and_loads_next(mock_config):
    mock_collection = MagicMock()
    next_pair = {**SAMPLE_PAIR, "question": "What is the price?"}
    admin_module._sessions[ADMIN_PHONE] = _active_session("training_review", SAMPLE_PAIR)
    with patch("app.admin._load_next_pending", return_value=next_pair), \
         patch("app.admin._mark_approved_in_pending"), \
         patch("app.admin._append_approved"), \
         patch("app.admin.send_text"), \
         patch("app.admin.send_interactive"), \
         patch("app.admin.rag") as mock_rag:
        admin_module.handle_admin(ADMIN_PHONE, "Approve", mock_collection)
    mock_rag.ingest_qna_pair.assert_called_once_with(SAMPLE_PAIR, mock_collection)
    assert admin_module._sessions[ADMIN_PHONE]["state"] == "training_review"
    assert admin_module._sessions[ADMIN_PHONE]["current_pair"] == next_pair

def test_approve_all_done_returns_to_menu(mock_config):
    admin_module._sessions[ADMIN_PHONE] = _active_session("training_review", SAMPLE_PAIR)
    with patch("app.admin._load_next_pending", return_value=None), \
         patch("app.admin._mark_approved_in_pending"), \
         patch("app.admin._append_approved"), \
         patch("app.admin.send_text"), \
         patch("app.admin.send_interactive"), \
         patch("app.admin.rag"):
        admin_module.handle_admin(ADMIN_PHONE, "Approve", MagicMock())
    assert admin_module._sessions[ADMIN_PHONE]["state"] == "menu"


# ─── Suggest ─────────────────────────────────────────────────────────────────

def test_suggest_transitions_to_suggest_state(mock_config):
    admin_module._sessions[ADMIN_PHONE] = _active_session("training_review", SAMPLE_PAIR)
    with patch("app.admin.send_text"), \
         patch("app.admin.send_interactive"):
        admin_module.handle_admin(ADMIN_PHONE, "Suggest", MagicMock())
    assert admin_module._sessions[ADMIN_PHONE]["state"] == "training_suggest"
    assert admin_module._sessions[ADMIN_PHONE]["current_pair"] == SAMPLE_PAIR

def test_suggestion_text_refines_and_shows_pair(mock_config):
    refined = {**SAMPLE_PAIR, "answer": "An improved answer."}
    admin_module._sessions[ADMIN_PHONE] = _active_session("training_suggest", SAMPLE_PAIR)
    with patch("app.admin._refine_with_llm", return_value=refined) as mock_refine, \
         patch("app.admin.send_text"), \
         patch("app.admin.send_interactive"):
        admin_module.handle_admin(ADMIN_PHONE, "add price info", MagicMock())
    mock_refine.assert_called_once_with(SAMPLE_PAIR, "add price info")
    assert admin_module._sessions[ADMIN_PHONE]["state"] == "training_review"
    assert admin_module._sessions[ADMIN_PHONE]["current_pair"]["answer"] == "An improved answer."

def test_suggestion_llm_failure_falls_back_to_original(mock_config):
    admin_module._sessions[ADMIN_PHONE] = _active_session("training_suggest", SAMPLE_PAIR)
    with patch("app.admin._refine_with_llm", side_effect=Exception("ollama down")), \
         patch("app.admin.send_text") as mock_text, \
         patch("app.admin.send_interactive"):
        admin_module.handle_admin(ADMIN_PHONE, "add price info", MagicMock())
    assert admin_module._sessions[ADMIN_PHONE]["current_pair"] == SAMPLE_PAIR
    assert any("failed" in str(c).lower() for c in mock_text.call_args_list)

def test_cancel_in_training_review_returns_to_menu(mock_config):
    admin_module._sessions[ADMIN_PHONE] = _active_session("training_review", SAMPLE_PAIR)
    with patch("app.admin.send_text") as mock_text, \
         patch("app.admin.send_interactive"):
        admin_module.handle_admin(ADMIN_PHONE, "Cancel", MagicMock())
    assert admin_module._sessions[ADMIN_PHONE]["state"] == "menu"
    assert any("cancel" in str(c).lower() for c in mock_text.call_args_list)

def test_cancel_in_training_suggest_returns_to_review(mock_config):
    admin_module._sessions[ADMIN_PHONE] = _active_session("training_suggest", SAMPLE_PAIR)
    with patch("app.admin.send_text"), \
         patch("app.admin.send_interactive"), \
         patch("app.admin._refine_with_llm") as mock_refine:
        admin_module.handle_admin(ADMIN_PHONE, "cancel", MagicMock())
    assert admin_module._sessions[ADMIN_PHONE]["state"] == "training_review"
    assert admin_module._sessions[ADMIN_PHONE]["current_pair"] == SAMPLE_PAIR
    mock_refine.assert_not_called()


# ─── _refine_with_llm unit tests ─────────────────────────────────────────────

def test_refine_with_llm_parses_full_json_response():
    """Suggestion that changes message_type should update the full pair."""
    pair = {"question": "Q?", "answer": "A.", "message_type": "text", "buttons": None, "image_url": None}
    llm_json = json.dumps({
        "answer": "Updated answer.",
        "message_type": "media",
        "buttons": None,
        "image_url": "https://example.com/img.jpg",
    })
    mock_resp = {"message": {"content": llm_json}}
    with patch("app.admin.ollama.chat", return_value=mock_resp):
        result = admin_module._refine_with_llm(pair, "send media with product image")
    assert result["answer"] == "Updated answer."
    assert result["message_type"] == "media"
    assert result["image_url"] == "https://example.com/img.jpg"


def test_refine_with_llm_falls_back_to_text_on_bad_json():
    """If LLM returns plain text instead of JSON, answer is updated, rest unchanged."""
    pair = {"question": "Q?", "answer": "A.", "message_type": "text", "buttons": None, "image_url": None}
    mock_resp = {"message": {"content": "Plain improved answer text."}}
    with patch("app.admin.ollama.chat", return_value=mock_resp):
        result = admin_module._refine_with_llm(pair, "make it better")
    assert result["answer"] == "Plain improved answer text."
    assert result["message_type"] == "text"  # unchanged


# ─── Auto-generate stub ───────────────────────────────────────────────────────

def test_auto_generate_returns_coming_soon(mock_config):
    admin_module._sessions[ADMIN_PHONE] = _active_session("menu")
    with patch("app.admin.send_text") as mock_text, \
         patch("app.admin.send_interactive"):
        admin_module.handle_admin(ADMIN_PHONE, "Auto-generate", MagicMock())
    assert "coming soon" in mock_text.call_args[0][1].lower()


# ─── Timeout ──────────────────────────────────────────────────────────────────

def test_timed_out_session_is_cleared(mock_config):
    admin_module._sessions[ADMIN_PHONE] = {
        "state": "menu",
        "current_pair": None,
        "last_active": datetime.utcnow() - timedelta(minutes=20),
    }
    with patch("app.admin.send_text") as mock_text:
        admin_module.handle_admin(ADMIN_PHONE, "Approve", MagicMock())
    assert ADMIN_PHONE not in admin_module._sessions
    assert "timed out" in mock_text.call_args[0][1].lower()
