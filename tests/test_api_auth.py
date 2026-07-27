"""API session 認證與公開端點的回歸測試。"""
from __future__ import annotations

import pytest

from app import create_app
from app.config import Config
from app.models import AnnotationTask
from app.services import line_login


@pytest.fixture
def app(tmp_path):
    cfg = Config(
        base_dir=tmp_path,
        data_dir=tmp_path,
        upload_dir=tmp_path / "up",
        mask_dir=tmp_path / "mask",
        db_file=tmp_path / "store.json",
    )
    cfg.db_backend = "json"
    cfg.use_real_sam = False
    cfg.use_real_embedding = False
    cfg.use_gcs = False
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


def test_line_binding_claims_unowned_liff_tasks(
    app,
    monkeypatch,
):
    client = app.test_client()
    repo = app.repo

    user = repo.get_user_by_username(
        app.smart_config.default_admin_user
    )

    assert user is not None

    task = AnnotationTask(
        line_user_id="U-line-binding-test",
        prompt="請標註貓咪",
        image_ids=["image-1"],
    )

    repo.add_task(task)

    monkeypatch.setattr(
        line_login,
        "exchange_token",
        lambda *args: {
            "id_token": "fake-id-token",
        },
    )

    monkeypatch.setattr(
        line_login,
        "verify_id_token",
        lambda *args: {
            "sub": "U-line-binding-test",
            "name": "LINE 測試使用者",
            "picture": "https://example.com/avatar.png",
        },
    )

    with client.session_transaction() as session:
        session["user_id"] = user.id
        session["username"] = user.username
        session["line_oauth_state"] = "test-state"
        session["line_oauth_nonce"] = "test-nonce"
        session["line_oauth_bind"] = True

    response = client.get(
        "/login/line/callback"
        "?state=test-state"
        "&code=test-code"
    )

    assert response.status_code == 302

    updated_user = repo.get_user(user.id)
    updated_task = repo.get_task(task.id)

    assert updated_user is not None
    assert updated_task is not None

    assert (
        updated_user.line_user_id
        == "U-line-binding-test"
    )

    assert updated_task.user_id == user.id

def test_line_login_claims_unowned_liff_tasks(
    app,
    monkeypatch,
):
    client = app.test_client()
    repo = app.repo

    task = AnnotationTask(
        line_user_id="U-direct-line-login",
        prompt="請標註狗狗",
        image_ids=["image-2"],
    )

    repo.add_task(task)

    monkeypatch.setattr(
        line_login,
        "exchange_token",
        lambda *args: {
            "id_token": "fake-id-token",
        },
    )

    monkeypatch.setattr(
        line_login,
        "verify_id_token",
        lambda *args: {
            "sub": "U-direct-line-login",
            "name": "LINE 直接登入使用者",
            "picture": "",
        },
    )

    with client.session_transaction() as session:
        session["line_oauth_state"] = "test-state"
        session["line_oauth_nonce"] = "test-nonce"
        session["line_oauth_bind"] = False

    response = client.get(
        "/login/line/callback"
        "?state=test-state"
        "&code=test-code"
    )

    assert response.status_code == 302

    user = repo.get_user_by_line_id(
        "U-direct-line-login"
    )

    loaded_task = repo.get_task(task.id)

    assert user is not None
    assert loaded_task is not None
    assert loaded_task.user_id == user.id

    with client.session_transaction() as session:
        assert session["user_id"] == user.id
