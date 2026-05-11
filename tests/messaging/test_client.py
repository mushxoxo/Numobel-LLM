import pytest
from unittest.mock import patch, MagicMock


@pytest.fixture(autouse=True)
def reset_client(monkeypatch):
    import app.messaging.client as client_mod
    monkeypatch.setattr(client_mod, "_session", None)


@pytest.fixture
def mock_env(monkeypatch):
    monkeypatch.setenv("WA2MATION_API_KEY", "test-key")
    monkeypatch.setenv("WA2MATION_VENDOR_UID", "test-uid")


def test_session_auth_header_set_from_env(mock_env):
    import app.messaging.client as client_mod
    session = client_mod._get_session()
    assert session.headers["Authorization"] == "Bearer test-key"
    assert session.headers["Content-Type"] == "application/json"


def test_session_is_reused(mock_env):
    import app.messaging.client as client_mod
    s1 = client_mod._get_session()
    s2 = client_mod._get_session()
    assert s1 is s2


def test_raises_when_credentials_missing(monkeypatch):
    monkeypatch.delenv("WA2MATION_API_KEY", raising=False)
    monkeypatch.delenv("WA2MATION_VENDOR_UID", raising=False)
    import app.messaging.client as client_mod
    with pytest.raises(RuntimeError, match="WA2MATION_API_KEY"):
        client_mod._get_session()


def test_wa2mation_post_builds_correct_url(mock_env):
    import app.messaging.client as client_mod
    mock_session = MagicMock()
    mock_session.post.return_value = MagicMock(status_code=200)
    with patch.object(client_mod, "_get_session", return_value=mock_session):
        client_mod.wa2mation_post("send-message", {"phone_number": "123"})
    called_url = mock_session.post.call_args[0][0]
    assert "test-uid" in called_url
    assert "send-message" in called_url
