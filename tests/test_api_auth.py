"""API session 認證與公開端點的回歸測試。"""
from __future__ import annotations

import pytest

from app import create_app
from app.config import Config


@pytest.fixture
def app(tmp_path):
    cfg = Config(
        base_dir=tmp_path,
        data_dir=tmp_path,
        upload_dir=tmp_path / "up",
        mask_dir=tmp_path / "mask",
        db_file=tmp_path / "store.json",
    )
    cfg.use_real_sam = False
    cfg.use_real_embedding = False
    application = create_app(cfg)
    application.config["TESTING"] = True
    return application


@pytest.mark.parametrize(
    ("method", "path"),
    [
        ("get", "/api"),
        ("get", "/api/images"),
        ("post", "/api/images/example/segment"),
        ("delete", "/api/images/example"),
    ],
)
def test_anonymous_api_requests_return_json_401(app, method, path):
    response = getattr(app.test_client(), method)(path)

    assert response.status_code == 401
    assert response.is_json
    assert response.get_json() == {
        "error": "authentication_required",
        "message": "請先登入後再使用 API",
    }


def test_valid_login_session_can_access_api(app):
    client = app.test_client()
    user = app.repo.get_user_by_username(app.smart_config.default_admin_user)
    assert user is not None

    with client.session_transaction() as sess:
        sess["user_id"] = user.id
        sess["username"] = user.username

    response = client.get("/api/images")

    assert response.status_code == 200
    assert response.get_json() == []


def test_stale_user_session_removes_only_login_fields(app):
    client = app.test_client()
    with client.session_transaction() as sess:
        sess["user_id"] = "missing-user"
        sess["username"] = "deleted-user"
        sess["line_oauth_state"] = "state-token"
        sess["line_oauth_nonce"] = "nonce-token"

    response = client.get("/api/images")

    assert response.status_code == 401
    with client.session_transaction() as sess:
        assert "user_id" not in sess
        assert "username" not in sess
        assert sess["line_oauth_state"] == "state-token"
        assert sess["line_oauth_nonce"] == "nonce-token"


def test_anonymous_api_request_preserves_line_oauth_session(app):
    client = app.test_client()
    with client.session_transaction() as sess:
        sess["line_oauth_state"] = "state-token"
        sess["line_oauth_nonce"] = "nonce-token"

    response = client.get("/api/images")

    assert response.status_code == 401
    with client.session_transaction() as sess:
        assert sess["line_oauth_state"] == "state-token"
        assert sess["line_oauth_nonce"] == "nonce-token"


def test_stale_user_session_cannot_access_home_page(app):
    client = app.test_client()
    with client.session_transaction() as sess:
        sess["user_id"] = "missing-user"
        sess["username"] = "deleted-user"

    response = client.get("/")

    assert response.status_code == 302
    assert response.headers["Location"].endswith("/login?next=/")
    with client.session_transaction() as sess:
        assert "user_id" not in sess
        assert "username" not in sess


def test_healthz_remains_public(app):
    response = app.test_client().get("/healthz")

    assert response.status_code == 200
    assert response.get_json() == {"status": "ok"}


def test_line_callback_remains_public(app):
    # 沒有 LINE 簽章仍會被 webhook 自己拒絕，但不應被 API 認證攔成 401。
    response = app.test_client().post("/callback")

    assert response.status_code == 400
