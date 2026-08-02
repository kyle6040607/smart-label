import io
from pathlib import Path

import pytest
from PIL import Image

from app import create_app
from app.config import Config
from app.models import AnnotationTask, User
from app.services import line_login


@pytest.fixture
def app(tmp_path):
    cfg = Config(
        base_dir=tmp_path,
        data_dir=tmp_path,
        upload_dir=tmp_path / "uploads",
        mask_dir=tmp_path / "masks",
        db_file=tmp_path / "store.json",
    )

    cfg.db_backend = "json"
    cfg.line_login_channel_id = "test-channel-id"
    cfg.liff_id = "test-liff-id"
    cfg.use_real_sam = False
    cfg.use_real_embedding = False
    cfg.use_gcs = False
    cfg.ensure_dirs()

    application = create_app(cfg)
    application.config["TESTING"] = True

    return application


def make_png_file() -> io.BytesIO:
    image_file = io.BytesIO()

    Image.new(
        "RGB",
        (20, 10),
        color=(255, 0, 0),
    ).save(
        image_file,
        format="PNG",
    )

    image_file.seek(0)

    return image_file


def test_liff_upload_creates_annotation_task(
    app,
    monkeypatch,
):
    monkeypatch.setattr(
        line_login,
        "verify_id_token",
        lambda *args: {
            "sub": "U-liff-upload-test",
            "name": "LIFF 測試使用者",
            "picture": "",
        },
    )

    client = app.test_client()

    response = client.post(
        "/liff/upload",
        data={
            "prompt": "請標註圖片中的貓咪",
            "id_token": "fake-id-token",
            "images": (
                make_png_file(),
                "cat.png",
            ),
        },
        content_type="multipart/form-data",
    )

    assert response.status_code == 201

    result = response.get_json()

    assert result is not None
    assert result["ok"] is True
    assert result["task_status"] == "pending"
    assert result["image_count"] == 1
    assert result["user_id"] == ""

    task = app.repo.get_task(
        result["task_id"]
    )

    assert task is not None
    assert task.line_user_id == "U-liff-upload-test"
    assert task.user_id == ""
    assert task.prompt == "請標註圖片中的貓咪"
    assert len(task.image_ids) == 1

    image_record = app.repo.get_image(
        task.image_ids[0]
    )

    assert image_record is not None
    assert image_record.owner_id == ""
    assert image_record.width == 20
    assert image_record.height == 10
    assert Path(image_record.path).exists()


def test_liff_upload_assigns_linked_user_default_project(
    app,
    monkeypatch,
):
    linked_user = User(
        username="linked-user",
        line_user_id="U-linked-user",
        display_name="已綁定使用者",
    )
    app.repo.add_user(linked_user)

    monkeypatch.setattr(
        line_login,
        "verify_id_token",
        lambda *args: {
            "sub": "U-linked-user",
            "name": "LINE 顯示名稱",
            "picture": "",
        },
    )

    response = app.test_client().post(
        "/liff/upload",
        data={
            "prompt": "請標註圖片中的貓咪",
            "id_token": "fake-id-token",
            "images": (make_png_file(), "cat.png"),
        },
        content_type="multipart/form-data",
    )

    assert response.status_code == 201
    result = response.get_json()
    task = app.repo.get_task(result["task_id"])
    image_record = app.repo.get_image(task.image_ids[0])
    project = app.repo.get_project(image_record.project_id)

    assert task.user_id == linked_user.id
    assert image_record.owner_id == linked_user.id
    assert project is not None
    assert project.owner_id == linked_user.id
    assert task.settings_snapshot["project_id"] == project.id
    assert image_record in app.repo.list_images(project.id)

def test_liff_task_download_checks_token_and_status(
    app,
    tmp_path,
    monkeypatch,
):
    client = app.test_client()

    task = AnnotationTask(
        status="pending",
    )
    zip_path = (
        tmp_path
        / "tasks"
        / task.id
        / "dataset.zip"
    )
    task.dataset_zip_path = str(zip_path)
    app.repo.add_task(task)

    invalid_token_response = client.get(
        f"/liff/tasks/{task.id}/download"
        "?token=wrong-token"
    )
    assert invalid_token_response.status_code == 403

    pending_response = client.get(
        f"/liff/tasks/{task.id}/download"
        f"?token={task.download_token}"
    )
    assert pending_response.status_code == 409

    task.status = "completed"
    app.repo.update_task(task)

    missing_file_response = client.get(
        f"/liff/tasks/{task.id}/download"
        f"?token={task.download_token}"
    )
    assert missing_file_response.status_code == 404

    zip_path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )
    zip_path.write_bytes(
        b"test-zip-content"
    )
    monkeypatch.setattr(
        app.storage,
        "read_bytes",
        lambda *_: pytest.fail(
            "ZIP 下載不應把整個檔案讀入記憶體"
        ),
    )
    monkeypatch.setattr(
        app.storage,
        "generate_download_url",
        lambda *_: (_ for _ in ()).throw(
            AttributeError("test signing unavailable")
        ),
    )

    success_response = client.get(
        f"/liff/tasks/{task.id}/download"
        f"?token={task.download_token}"
    )

    assert success_response.status_code == 200
    assert success_response.data == b"test-zip-content"
    assert "attachment" in success_response.headers[
        "Content-Disposition"
    ]

def test_liff_task_status_returns_download_url(
    app,
):
    client = app.test_client()

    task = app.repo.add_task(
        AnnotationTask(
            status="pending",
        )
    )

    invalid_response = client.get(
        f"/liff/tasks/{task.id}/status"
        "?token=wrong-token"
    )
    assert invalid_response.status_code == 403

    pending_response = client.get(
        f"/liff/tasks/{task.id}/status"
        f"?token={task.download_token}"
    )
    pending_result = pending_response.get_json()

    assert pending_response.status_code == 200
    assert pending_result is not None
    assert pending_result["task_status"] == "pending"
    assert "download_url" not in pending_result

    task.status = "completed"
    task.dataset_zip_path = "dataset.zip"
    app.repo.update_task(task)

    completed_response = client.get(
        f"/liff/tasks/{task.id}/status"
        f"?token={task.download_token}"
    )
    completed_result = completed_response.get_json()

    assert completed_response.status_code == 200
    assert completed_result is not None
    assert completed_result["task_status"] == "completed"
    assert completed_result["download_url"] == (
        f"/liff/tasks/{task.id}/download"
        f"?token={task.download_token}"
    )

    task.status = "failed"
    task.error_message = "測試失敗原因"
    app.repo.update_task(task)

    failed_response = client.get(
        f"/liff/tasks/{task.id}/status"
        f"?token={task.download_token}"
    )
    failed_result = failed_response.get_json()

    assert failed_response.status_code == 200
    assert failed_result is not None
    assert failed_result["task_status"] == "failed"
    assert failed_result["error_message"] == "測試失敗原因"


def test_liff_gcs_download_redirects_to_signed_url(
    app,
    monkeypatch,
):
    task = app.repo.add_task(
        AnnotationTask(
            status="completed",
            dataset_zip_path=(
                "gs://test-bucket/datasets/task/dataset.zip"
            ),
        )
    )
    monkeypatch.setattr(
        app.storage,
        "exists",
        lambda reference: reference == task.dataset_zip_path,
    )
    monkeypatch.setattr(
        app.storage,
        "generate_download_url",
        lambda reference, download_name: (
            "https://storage.example/dataset.zip?signed=1"
        ),
    )

    response = app.test_client().get(
        f"/liff/tasks/{task.id}/download"
        f"?token={task.download_token}"
    )

    assert response.status_code == 302
    assert response.headers["Location"] == (
        "https://storage.example/dataset.zip?signed=1"
    )


def test_liff_task_list_only_returns_verified_user_tasks(
    app,
    monkeypatch,
):
    monkeypatch.setattr(
        line_login,
        "verify_id_token",
        lambda *args: {
            "sub": "U-task-owner",
            "name": "任務擁有者",
        },
    )

    own_task = app.repo.add_task(
        AnnotationTask(
            line_user_id="U-task-owner",
            prompt="cat",
            image_ids=["own-image"],
            processed_image_ids=["own-image"],
            status="completed",
            dataset_zip_path="dataset.zip",
            dataset_version=1,
        )
    )

    app.repo.add_task(
        AnnotationTask(
            line_user_id="U-other-user",
            prompt="dog",
            image_ids=["other-image"],
        )
    )

    client = app.test_client()

    response = client.post(
        "/liff/tasks",
        json={
            "id_token": "fake-id-token",
        },
    )

    assert response.status_code == 200

    result = response.get_json()

    assert result is not None
    assert result["ok"] is True
    assert result["task_count"] == 1

    task_result = result["tasks"][0]

    assert task_result["task_id"] == own_task.id
    assert task_result["prompt"] == "cat"
    assert task_result["image_count"] == 1
    assert task_result["processed_image_count"] == 1
    assert task_result["dataset_version"] == 1
    assert task_result["can_add_images"] is True
    assert "download_url" in task_result
    assert "line_user_id" not in task_result
    assert "download_token" not in task_result


def test_liff_excluded_results_require_owner_and_return_thumbnail(
    app,
    tmp_path,
    monkeypatch,
):
    preview = tmp_path / "preview.jpg"
    preview.write_bytes(b"jpeg-preview")
    task = app.repo.add_task(
        AnnotationTask(
            line_user_id="U-owner",
            status="completed",
            excluded_count=1,
            excluded_results=[
                {
                    "segment_id": "seg-1",
                    "image_id": "img-1",
                    "detection_confidence": 0.3,
                    "preview_path": str(preview),
                }
            ],
        )
    )
    monkeypatch.setattr(
        line_login,
        "verify_id_token",
        lambda token, *args: {
            "sub": "U-owner" if token == "owner-token" else "U-other"
        },
    )
    client = app.test_client()
    denied = client.post(
        f"/liff/tasks/{task.id}/excluded",
        json={"id_token": "other-token"},
    )
    assert denied.status_code == 404

    response = client.post(
        f"/liff/tasks/{task.id}/excluded",
        json={"id_token": "owner-token"},
    )
    assert response.status_code == 200
    item = response.get_json()["items"][0]
    assert "preview_path" not in item
    thumbnail = client.get(item["thumbnail_url"])
    assert thumbnail.status_code == 200
    assert thumbnail.data == b"jpeg-preview"
