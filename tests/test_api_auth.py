"""API session 認證與公開端點的回歸測試。"""
from __future__ import annotations

import pytest

from app import create_app
from app.config import Config
from app.models import (
    AnnotationTask,
    ImageRecord,
    LabelExample,
    Segment,
    User,
)
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


def test_images_are_isolated_between_users(app, tmp_path):
    repo = app.repo
    owner = repo.add_user(User(username="owner"))
    other = repo.add_user(User(username="other"))

    owner_file = tmp_path / "owner.png"
    owner_file.write_bytes(b"owner-image")
    other_file = tmp_path / "other.png"
    other_file.write_bytes(b"other-image")

    owner_image = repo.add_image(ImageRecord(
        id="owner-image",
        owner_id=owner.id,
        filename="owner.png",
        path=str(owner_file),
    ))
    other_image = repo.add_image(ImageRecord(
        id="other-image",
        owner_id=other.id,
        filename="other.png",
        path=str(other_file),
    ))
    other_mask = tmp_path / "other-mask.png"
    other_mask.write_bytes(b"mask")
    other_segment = repo.add_segment(Segment(
        id="other-segment",
        image_id=other_image.id,
        mask_path=str(other_mask),
        needs_review=True,
    ))

    client = app.test_client()
    with client.session_transaction() as sess:
        sess["user_id"] = owner.id
        sess["username"] = owner.username

    listed = client.get("/api/images")
    assert listed.status_code == 200
    assert [image["id"] for image in listed.get_json()] == [owner_image.id]

    assert client.get(f"/api/images/{other_image.id}/file").status_code == 404
    assert client.delete(f"/api/images/{other_image.id}").status_code == 404
    assert client.get(
        f"/api/images/{other_image.id}/segments"
    ).status_code == 404
    assert client.get(
        f"/api/segments/{other_segment.id}/mask"
    ).status_code == 404
    assert client.get("/api/review/queue").get_json() == []


def test_legacy_unowned_images_are_admin_only(app, tmp_path):
    repo = app.repo
    normal_user = repo.add_user(User(username="normal"))
    legacy_file = tmp_path / "legacy.png"
    legacy_file.write_bytes(b"legacy")
    legacy = repo.add_image(ImageRecord(
        id="legacy-image",
        filename="legacy.png",
        path=str(legacy_file),
    ))

    normal_client = app.test_client()
    with normal_client.session_transaction() as sess:
        sess["user_id"] = normal_user.id
    assert normal_client.get("/api/images").get_json() == []
    assert normal_client.get(
        f"/api/images/{legacy.id}/file"
    ).status_code == 404

    admin = repo.get_user_by_username(app.smart_config.default_admin_user)
    assert admin is not None
    admin_client = app.test_client()
    with admin_client.session_transaction() as sess:
        sess["user_id"] = admin.id
    assert [
        image["id"] for image in admin_client.get("/api/images").get_json()
    ] == [legacy.id]


def test_labels_examples_and_delete_are_isolated_by_user(app):
    repo = app.repo
    owner = repo.add_user(User(username="label-owner"))
    other = repo.add_user(User(username="label-other"))
    owner_image = repo.add_image(ImageRecord(
        id="label-owner-image",
        owner_id=owner.id,
    ))
    other_image = repo.add_image(ImageRecord(
        id="label-other-image",
        owner_id=other.id,
    ))
    owner_segment = repo.add_segment(Segment(
        id="label-owner-segment",
        image_id=owner_image.id,
        human_label="cat",
        reviewed=True,
    ))
    other_segment = repo.add_segment(Segment(
        id="label-other-segment",
        image_id=other_image.id,
        human_label="dog",
        reviewed=True,
    ))
    repo.add_example(LabelExample(
        id="label-owner-example",
        owner_id=owner.id,
        label="cat",
        feature=[1.0, 0.0],
        source_segment_id=owner_segment.id,
    ))
    repo.add_example(LabelExample(
        id="label-other-example",
        owner_id=other.id,
        label="dog",
        feature=[0.0, 1.0],
        source_segment_id=other_segment.id,
    ))
    app.pipeline.refit()
    assert app.pipeline.classifiers[owner.id].labels == ["cat"]
    assert app.pipeline.classifiers[other.id].labels == ["dog"]

    client = app.test_client()
    with client.session_transaction() as sess:
        sess["user_id"] = owner.id
        sess["username"] = owner.username

    assert client.get("/api/labels").get_json() == ["cat"]
    examples = client.get("/api/examples").get_json()
    assert [example["id"] for example in examples] == [
        "label-owner-example"
    ]
    stats = client.get("/api/stats").get_json()
    assert stats["num_examples"] == 1
    assert stats["num_labels"] == 1
    assert stats["label_counts"] == {"cat": 1}

    deleted = client.delete("/api/labels/cat")
    assert deleted.status_code == 200
    assert repo.labels(owner.id) == []
    assert repo.labels(other.id) == ["dog"]
    assert repo.get_segment(owner_segment.id).human_label is None
    assert repo.get_segment(other_segment.id).human_label == "dog"


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
    task_image = repo.add_image(ImageRecord(id="image-1"))

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
    assert repo.get_image(task_image.id).owner_id == user.id

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
    task_image = repo.add_image(ImageRecord(id="image-2"))

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
    assert repo.get_image(task_image.id).owner_id == user.id

    with client.session_transaction() as session:
        assert session["user_id"] == user.id
