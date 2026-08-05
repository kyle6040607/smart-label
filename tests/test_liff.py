import io
from pathlib import Path

import pytest
from PIL import Image

from app import create_app
from app.config import Config
from app.models import AnnotationTask, ImageRecord, User
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


def test_liff_chunked_upload_finalizes_one_task_and_triggers_once(
    app,
    monkeypatch,
):
    operations = []
    app.smart_config.cloud_run_task_job_name = "smart-label-task-worker"
    monkeypatch.setattr(
        "app.routes.liff.trigger_task_worker",
        lambda config: operations.append(config.cloud_run_task_job_name)
        or "operations/chunked-job-1",
    )
    monkeypatch.setattr(
        line_login,
        "verify_id_token",
        lambda *args: {
            "sub": "U-chunked-upload",
            "name": "分批上傳測試",
            "picture": "",
        },
    )
    client = app.test_client()

    init_response = client.post(
        "/liff/uploads/init",
        json={
            "id_token": "fake-id-token",
            "prompt": "cat",
            "expected_image_count": 2,
        },
    )
    assert init_response.status_code == 201
    init_result = init_response.get_json()
    session_id = init_result["session_id"]
    assert app.repo.get_task(session_id).status == "uploading"
    assert operations == []

    batch_response = client.post(
        f"/liff/uploads/{session_id}/batch",
        data={
            "id_token": "fake-id-token",
            "batch_id": "batch-1",
            "images": [
                (make_png_file(), "cat-1.png"),
                (make_png_file(), "cat-2.png"),
            ],
        },
        content_type="multipart/form-data",
    )
    assert batch_response.status_code == 201
    assert batch_response.get_json()["uploaded_count"] == 2
    assert app.repo.get_task(session_id).status == "uploading"
    assert operations == []

    finalize_response = client.post(
        f"/liff/uploads/{session_id}/finalize",
        json={"id_token": "fake-id-token"},
    )
    assert finalize_response.status_code == 202
    finalize_result = finalize_response.get_json()
    assert finalize_result["task_id"] == session_id
    assert finalize_result["task_status"] == "pending"
    assert finalize_result["image_count"] == 2
    assert finalize_result["already_finalized"] is False
    assert operations == ["smart-label-task-worker"]

    repeated_response = client.post(
        f"/liff/uploads/{session_id}/finalize",
        json={"id_token": "fake-id-token"},
    )
    assert repeated_response.status_code == 200
    assert repeated_response.get_json()["already_finalized"] is True
    assert operations == ["smart-label-task-worker"]


def test_liff_chunked_upload_accepts_more_than_thirty_images(
    app,
    monkeypatch,
):
    monkeypatch.setattr(
        line_login,
        "verify_id_token",
        lambda *args: {
            "sub": "U-many-images",
            "name": "大量圖片測試",
            "picture": "",
        },
    )

    response = app.test_client().post(
        "/liff/uploads/init",
        json={
            "id_token": "fake-id-token",
            "prompt": "cat",
            "expected_image_count": 31,
            "expected_total_bytes": 3100,
        },
    )

    assert response.status_code == 201
    result = response.get_json()
    assert result["expected_image_count"] == 31
    assert result["batch_max_images"] == 5
    assert result["batch_max_bytes"] == 15 * 1024 * 1024


def test_liff_chunked_upload_rejects_task_over_total_byte_quota(
    app,
    monkeypatch,
):
    app.smart_config.liff_upload_max_total_bytes = 100
    monkeypatch.setattr(
        line_login,
        "verify_id_token",
        lambda *args: {
            "sub": "U-over-quota",
            "name": "上傳配額測試",
            "picture": "",
        },
    )

    response = app.test_client().post(
        "/liff/uploads/init",
        json={
            "id_token": "fake-id-token",
            "prompt": "cat",
            "expected_image_count": 1,
            "expected_total_bytes": 101,
        },
    )

    assert response.status_code == 413
    assert response.get_json()["max_total_bytes"] == 100


def test_liff_upload_page_exposes_progress_and_server_batch_limits(app):
    response = app.test_client().get("/liff/create")

    assert response.status_code == 200
    html = response.get_data(as_text=True)
    assert 'id="upload-progress"' in html
    assert 'id="upload-success-modal"' in html
    assert 'id="upload-success-close"' in html
    assert "完成並離開" in html
    assert 'data-upload-batch-max-images="5"' in html
    assert f'data-upload-batch-max-bytes="{15 * 1024 * 1024}"' in html


def test_liff_append_images_merges_into_completed_task_and_triggers_once(
    app,
    monkeypatch,
):
    operations = []
    app.smart_config.cloud_run_task_job_name = "smart-label-task-worker"
    monkeypatch.setattr(
        "app.routes.liff.trigger_task_worker",
        lambda config: operations.append(config.cloud_run_task_job_name)
        or "operations/append-job-1",
    )
    monkeypatch.setattr(
        line_login,
        "verify_id_token",
        lambda *args: {
            "sub": "U-append-owner",
            "name": "追加照片測試",
            "picture": "",
        },
    )
    app.repo.add_image(
        ImageRecord(
            id="existing-image",
            owner_id="web-owner",
            project_id="project-1",
            filename="existing.png",
            path="existing.png",
        )
    )
    target = app.repo.add_task(
        AnnotationTask(
            user_id="web-owner",
            project_id="project-1",
            line_user_id="U-append-owner",
            prompt="cat",
            image_ids=["existing-image"],
            processed_image_ids=["existing-image"],
            dataset_version=1,
            dataset_zip_path="dataset-v1.zip",
            status="completed",
            attempt_count=2,
            settings_snapshot={
                "project_id": "project-1",
                "export_confidence_threshold": 0.5,
            },
        )
    )
    client = app.test_client()

    context_response = client.post(
        f"/liff/tasks/{target.id}/append/context",
        json={"id_token": "fake-id-token"},
    )
    assert context_response.status_code == 200
    assert context_response.get_json()["prompt"] == "cat"

    init_response = client.post(
        "/liff/uploads/init",
        json={
            "id_token": "fake-id-token",
            "prompt": "front-end-cannot-change-this",
            "target_task_id": target.id,
            "expected_image_count": 1,
        },
    )
    assert init_response.status_code == 201
    init_result = init_response.get_json()
    assert init_result["append_mode"] is True
    session_id = init_result["session_id"]
    session = app.repo.get_task(session_id)
    assert session.prompt == "cat"
    assert session.user_id == target.user_id
    assert session.project_id == target.project_id

    batch_response = client.post(
        f"/liff/uploads/{session_id}/batch",
        data={
            "id_token": "fake-id-token",
            "batch_id": "append-batch-1",
            "images": (make_png_file(), "new-cat.png"),
        },
        content_type="multipart/form-data",
    )
    assert batch_response.status_code == 201

    finalize_response = client.post(
        f"/liff/uploads/{session_id}/finalize",
        json={"id_token": "fake-id-token"},
    )
    assert finalize_response.status_code == 202
    result = finalize_response.get_json()
    assert result["task_id"] == target.id
    assert result["task_status"] == "pending"
    assert result["added_image_count"] == 1
    assert result["image_count"] == 2
    assert operations == ["smart-label-task-worker"]

    updated_target = app.repo.get_task(target.id)
    assert updated_target.status == "pending"
    assert updated_target.processed_image_ids == ["existing-image"]
    assert updated_target.dataset_version == 1
    assert updated_target.dataset_zip_path == "dataset-v1.zip"
    assert updated_target.attempt_count == 0
    assert len(updated_target.image_ids) == 2
    assert app.repo.get_task(session_id).status == "upload_merged"
    assert app.repo.get_task(session_id).image_ids == []

    repeated_response = client.post(
        f"/liff/uploads/{session_id}/finalize",
        json={"id_token": "fake-id-token"},
    )
    assert repeated_response.status_code == 200
    assert repeated_response.get_json()["already_finalized"] is True
    assert operations == ["smart-label-task-worker"]

    task_list = client.post(
        "/liff/tasks",
        json={"id_token": "fake-id-token"},
    ).get_json()
    assert task_list["task_count"] == 1
    assert task_list["tasks"][0]["task_id"] == target.id


def test_liff_append_images_requires_owner_and_completed_task(
    app,
    monkeypatch,
):
    current_line_user = {"sub": "U-other"}
    monkeypatch.setattr(
        line_login,
        "verify_id_token",
        lambda *args: {
            "sub": current_line_user["sub"],
            "name": "測試使用者",
            "picture": "",
        },
    )
    target = app.repo.add_task(
        AnnotationTask(
            line_user_id="U-owner",
            prompt="cat",
            status="completed",
        )
    )
    client = app.test_client()

    unauthorized = client.post(
        f"/liff/tasks/{target.id}/append/context",
        json={"id_token": "fake-id-token"},
    )
    assert unauthorized.status_code == 404

    current_line_user["sub"] = "U-owner"
    target.status = "processing"
    app.repo.update_task(target)
    not_completed = client.post(
        "/liff/uploads/init",
        json={
            "id_token": "fake-id-token",
            "target_task_id": target.id,
            "expected_image_count": 1,
        },
    )
    assert not_completed.status_code == 409
    assert "已完成" in not_completed.get_json()["message"]


def test_liff_chunked_upload_repeated_batch_is_idempotent(
    app,
    monkeypatch,
):
    monkeypatch.setattr(
        line_login,
        "verify_id_token",
        lambda *args: {
            "sub": "U-repeated-batch",
            "name": "批次重送測試",
            "picture": "",
        },
    )
    client = app.test_client()
    init_result = client.post(
        "/liff/uploads/init",
        json={
            "id_token": "fake-id-token",
            "prompt": "cat",
            "expected_image_count": 1,
        },
    ).get_json()
    session_id = init_result["session_id"]

    first_image = make_png_file()
    image_bytes = len(first_image.getvalue())
    first_response = client.post(
        f"/liff/uploads/{session_id}/batch",
        data={
            "id_token": "fake-id-token",
            "batch_id": "batch-1",
            "images": (first_image, "cat.png"),
        },
        content_type="multipart/form-data",
    )
    assert first_response.status_code == 201
    assert first_response.get_json()["uploaded_bytes"] == image_bytes

    repeated_response = client.post(
        f"/liff/uploads/{session_id}/batch",
        data={
            "id_token": "fake-id-token",
            "batch_id": "batch-1",
            "images": (make_png_file(), "cat.png"),
        },
        content_type="multipart/form-data",
    )
    assert repeated_response.status_code == 200
    repeated_result = repeated_response.get_json()
    assert repeated_result["duplicate_batch"] is True
    assert repeated_result["uploaded_count"] == 1
    assert repeated_result["uploaded_bytes"] == image_bytes
    assert len(app.repo.get_task(session_id).image_ids) == 1
    upload = app.repo.get_task(session_id).settings_snapshot["upload"]
    assert upload["uploaded_bytes"] == image_bytes
    assert upload["completed_batch_bytes"] == {"batch-1": image_bytes}


def test_liff_chunked_upload_rejects_bytes_beyond_declared_total(
    app,
    monkeypatch,
):
    monkeypatch.setattr(
        line_login,
        "verify_id_token",
        lambda *args: {
            "sub": "U-byte-mismatch",
            "name": "大小不符測試",
            "picture": "",
        },
    )
    client = app.test_client()
    init_result = client.post(
        "/liff/uploads/init",
        json={
            "id_token": "fake-id-token",
            "prompt": "cat",
            "expected_image_count": 1,
            "expected_total_bytes": 1,
        },
    ).get_json()

    response = client.post(
        f"/liff/uploads/{init_result['session_id']}/batch",
        data={
            "id_token": "fake-id-token",
            "batch_id": "batch-1",
            "images": (make_png_file(), "cat.png"),
        },
        content_type="multipart/form-data",
    )

    assert response.status_code == 409
    assert "預期大小" in response.get_json()["message"]
    assert app.repo.get_task(init_result["session_id"]).image_ids == []


def test_liff_chunked_upload_does_not_finalize_incomplete_task(
    app,
    monkeypatch,
):
    operations = []
    monkeypatch.setattr(
        "app.routes.liff.trigger_task_worker",
        lambda config: operations.append(config.cloud_run_task_job_name),
    )
    monkeypatch.setattr(
        line_login,
        "verify_id_token",
        lambda *args: {
            "sub": "U-incomplete-upload",
            "name": "未完成上傳測試",
            "picture": "",
        },
    )
    client = app.test_client()
    init_result = client.post(
        "/liff/uploads/init",
        json={
            "id_token": "fake-id-token",
            "prompt": "cat",
            "expected_image_count": 2,
        },
    ).get_json()
    session_id = init_result["session_id"]

    client.post(
        f"/liff/uploads/{session_id}/batch",
        data={
            "id_token": "fake-id-token",
            "batch_id": "batch-1",
            "images": (make_png_file(), "cat.png"),
        },
        content_type="multipart/form-data",
    )
    finalize_response = client.post(
        f"/liff/uploads/{session_id}/finalize",
        json={"id_token": "fake-id-token"},
    )

    assert finalize_response.status_code == 409
    assert finalize_response.get_json()["code"] == "UPLOAD_INCOMPLETE"
    assert app.repo.get_task(session_id).status == "uploading"
    assert operations == []


def test_liff_chunked_upload_rejects_different_line_user(
    app,
    monkeypatch,
):
    def verify_id_token(id_token, *_args):
        return {
            "sub": "U-owner" if id_token == "owner-token" else "U-other",
            "name": "LIFF 使用者",
            "picture": "",
        }

    monkeypatch.setattr(line_login, "verify_id_token", verify_id_token)
    client = app.test_client()
    init_result = client.post(
        "/liff/uploads/init",
        json={
            "id_token": "owner-token",
            "prompt": "cat",
            "expected_image_count": 1,
        },
    ).get_json()

    response = client.post(
        f"/liff/uploads/{init_result['session_id']}/batch",
        data={
            "id_token": "other-token",
            "batch_id": "batch-1",
            "images": (make_png_file(), "cat.png"),
        },
        content_type="multipart/form-data",
    )

    assert response.status_code == 404
    assert app.repo.get_task(init_result["session_id"]).image_ids == []


def test_liff_chunked_upload_cleans_expired_session(
    app,
    monkeypatch,
):
    monkeypatch.setattr(
        line_login,
        "verify_id_token",
        lambda *args: {
            "sub": "U-expired-upload",
            "name": "過期上傳測試",
            "picture": "",
        },
    )
    client = app.test_client()
    init_result = client.post(
        "/liff/uploads/init",
        json={
            "id_token": "fake-id-token",
            "prompt": "cat",
            "expected_image_count": 1,
        },
    ).get_json()
    session_id = init_result["session_id"]
    client.post(
        f"/liff/uploads/{session_id}/batch",
        data={
            "id_token": "fake-id-token",
            "batch_id": "batch-1",
            "images": (make_png_file(), "cat.png"),
        },
        content_type="multipart/form-data",
    )

    task = app.repo.get_task(session_id)
    image_id = task.image_ids[0]
    image_path = Path(app.repo.get_image(image_id).path)
    task.settings_snapshot["upload"]["expires_at"] = 1
    app.repo.update_task(task)
    assert image_path.exists()

    response = client.post(
        "/liff/uploads/init",
        json={
            "id_token": "fake-id-token",
            "prompt": "dog",
            "expected_image_count": 1,
        },
    )

    assert response.status_code == 201
    expired_task = app.repo.get_task(session_id)
    assert expired_task.status == "upload_expired"
    assert expired_task.image_ids == []
    assert app.repo.get_image(image_id) is None
    assert not image_path.exists()


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
    assert result["job_triggered"] is False
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


def test_liff_upload_triggers_configured_cloud_run_job(app, monkeypatch):
    app.smart_config.cloud_run_task_job_name = "smart-label-task-worker"
    operations = []
    monkeypatch.setattr(
        "app.routes.liff.trigger_task_worker",
        lambda config: operations.append(config.cloud_run_task_job_name)
        or "operations/job-run-1",
    )
    monkeypatch.setattr(
        line_login,
        "verify_id_token",
        lambda *args: {
            "sub": "U-job-trigger-test",
            "name": "Job 觸發測試",
            "picture": "",
        },
    )

    response = app.test_client().post(
        "/liff/upload",
        data={
            "prompt": "cat",
            "id_token": "fake-id-token",
            "images": (make_png_file(), "cat.png"),
        },
        content_type="multipart/form-data",
    )

    assert response.status_code == 201
    result = response.get_json()
    assert result["job_triggered"] is True
    assert operations == ["smart-label-task-worker"]
    assert app.repo.get_task(result["task_id"]).status == "pending"


def test_liff_upload_keeps_pending_task_when_job_trigger_fails(app, monkeypatch):
    app.smart_config.cloud_run_task_job_name = "smart-label-task-worker"
    monkeypatch.setattr(
        "app.routes.liff.trigger_task_worker",
        lambda config: (_ for _ in ()).throw(RuntimeError(config.cloud_run_task_job_name)),
    )
    monkeypatch.setattr(
        line_login,
        "verify_id_token",
        lambda *args: {
            "sub": "U-job-trigger-failure",
            "name": "Job 失敗測試",
            "picture": "",
        },
    )

    response = app.test_client().post(
        "/liff/upload",
        data={
            "prompt": "cat",
            "id_token": "fake-id-token",
            "images": (make_png_file(), "cat.png"),
        },
        content_type="multipart/form-data",
    )

    assert response.status_code == 201
    result = response.get_json()
    assert result["job_triggered"] is False
    assert app.repo.get_task(result["task_id"]).status == "pending"


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
