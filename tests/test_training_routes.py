import pytest

from app import create_app
from app.config import Config
from app.models import AnnotationTask, Project


@pytest.fixture
def training_context(tmp_path):
    cfg = Config(
        base_dir=tmp_path,
        data_dir=tmp_path,
        upload_dir=tmp_path / "uploads",
        mask_dir=tmp_path / "masks",
        db_file=tmp_path / "store.json",
    )
    cfg.db_backend = "json"
    cfg.use_real_sam = False
    cfg.use_real_embedding = False
    cfg.use_gcs = False
    cfg.ensure_dirs()

    app = create_app(cfg)
    app.config["TESTING"] = True
    user = app.repo.get_user_by_username(cfg.default_admin_user)
    assert user is not None
    project = app.repo.add_project(
        Project(owner_id=user.id, name="訓練測試專案")
    )
    client = app.test_client()
    with client.session_transaction() as session:
        session["user_id"] = user.id
        session["username"] = user.username

    return app, client, user, project


def test_liff_annotation_task_is_not_reported_or_stopped_as_training(
    training_context,
):
    app, client, user, project = training_context
    liff_task = AnnotationTask(
        user_id=user.id,
        project_id=project.id,
        line_user_id="U-training-isolation",
        prompt="cat",
        status="processing",
    )
    app.repo.add_task(liff_task)

    status_response = client.get(
        f"/api/projects/{project.id}/train/status"
    )
    assert status_response.status_code == 200
    assert status_response.get_json()["has_training"] is False

    stop_response = client.post(
        f"/api/projects/{project.id}/train/stop"
    )
    assert stop_response.status_code == 400
    assert app.repo.get_task(liff_task.id).status == "processing"


def test_typed_liff_task_wins_over_legacy_prompt_fallback(training_context):
    app, client, user, project = training_context
    liff_task = AnnotationTask(
        user_id=user.id,
        project_id=project.id,
        prompt=f"[project:{project.id}] YOLOv26x-seg 訓練",
        status="completed",
        settings_snapshot={"task_type": "liff_annotation"},
        best_model_path="missing.pt",
    )
    app.repo.add_task(liff_task)

    status_response = client.get(
        f"/api/projects/{project.id}/train/status"
    )
    assert status_response.get_json()["has_training"] is False

    download_response = client.get(
        f"/api/projects/{project.id}/train/download"
    )
    assert download_response.status_code == 404


def test_legacy_training_task_remains_visible_and_stoppable(training_context):
    app, client, user, project = training_context
    legacy_task = AnnotationTask(
        user_id=user.id,
        prompt=(
            f"[project:{project.id}] "
            f"YOLOv26x-seg 訓練 ({project.name})"
        ),
        status="processing",
    )
    app.repo.add_task(legacy_task)

    status_response = client.get(
        f"/api/projects/{project.id}/train/status"
    )
    status_result = status_response.get_json()
    assert status_result["has_training"] is True
    assert status_result["task_id"] == legacy_task.id

    stop_response = client.post(
        f"/api/projects/{project.id}/train/stop"
    )
    assert stop_response.status_code == 200
    assert app.repo.get_task(legacy_task.id).status == "canceled"


def test_new_training_task_records_explicit_task_type(
    training_context,
    monkeypatch,
):
    app, client, user, project = training_context
    monkeypatch.setattr(
        app.repo,
        "list_labeled_segments_by_project",
        lambda owner_id, project_id: [object()],
    )

    class DeferredThread:
        def __init__(self, target, daemon):
            self.target = target
            self.daemon = daemon

        def start(self):
            return None

    monkeypatch.setattr(
        "app.routes.training.threading.Thread",
        DeferredThread,
    )

    response = client.post(
        f"/api/projects/{project.id}/train",
        json={"epochs": 5, "imgsz": 640},
    )
    assert response.status_code == 202
    result = response.get_json()
    task = app.repo.get_task(result["task_id"])
    assert task is not None
    assert task.user_id == user.id
    assert task.project_id == project.id
    assert task.settings_snapshot["task_type"] == "yolo_training"

    status_response = client.get(
        f"/api/projects/{project.id}/train/status"
    )
    assert status_response.get_json()["task_id"] == task.id
