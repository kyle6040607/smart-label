from app.models import AnnotationTask
from app.repository import Repository


def test_annotation_task_persists_after_reload(
    tmp_path,
):
    store_file = tmp_path / "store.json"

    repo = Repository(store_file)

    task = AnnotationTask(
        user_id="web-user-1",
        line_user_id="U123456789",
        prompt="請標註圖片中的貓咪",
        image_ids=["image-1", "image-2"],
    )

    repo.add_task(task)

    assert store_file.exists()

    reloaded_repo = Repository(store_file)
    loaded_task = reloaded_repo.get_task(task.id)

    assert loaded_task is not None
    assert loaded_task.id == task.id
    assert loaded_task.user_id == "web-user-1"
    assert loaded_task.line_user_id == "U123456789"
    assert loaded_task.prompt == "請標註圖片中的貓咪"
    assert loaded_task.image_ids == [
        "image-1",
        "image-2",
    ]
    assert loaded_task.status == "pending"
    assert loaded_task.download_token == task.download_token

def test_annotation_task_update_persists(
    tmp_path,
):
    store_file = tmp_path / "store.json"

    repo = Repository(store_file)

    task = AnnotationTask(
        line_user_id="U123456789",
        prompt="請標註狗狗",
        image_ids=["image-1"],
    )

    repo.add_task(task)

    task.status = "completed"
    task.dataset_zip_path = str(
        tmp_path / "dataset.zip"
    )
    task.best_model_path = str(
        tmp_path / "best.pt"
    )

    repo.update_task(task)

    reloaded_repo = Repository(store_file)
    loaded_task = reloaded_repo.get_task(task.id)

    assert loaded_task is not None
    assert loaded_task.status == "completed"
    assert loaded_task.dataset_zip_path == str(
        tmp_path / "dataset.zip"
    )
    assert loaded_task.best_model_path == str(
        tmp_path / "best.pt"
    )

def test_assign_tasks_to_user_only_claims_unowned_tasks(
    tmp_path,
):
    store_file = tmp_path / "store.json"
    repo = Repository(store_file)

    unowned_task = AnnotationTask(
        line_user_id="U-line-1",
        prompt="尚未綁定的任務",
    )

    already_owned_task = AnnotationTask(
        user_id="other-web-user",
        line_user_id="U-line-1",
        prompt="已經屬於其他帳號",
    )

    different_line_task = AnnotationTask(
        line_user_id="U-line-2",
        prompt="另一位 LINE 使用者",
    )

    repo.add_task(unowned_task)
    repo.add_task(already_owned_task)
    repo.add_task(different_line_task)

    assigned_count = repo.assign_tasks_to_user(
        line_user_id="U-line-1",
        user_id="web-user-1",
    )

    assert assigned_count == 1

    reloaded_repo = Repository(store_file)

    loaded_unowned = reloaded_repo.get_task(
        unowned_task.id
    )
    loaded_owned = reloaded_repo.get_task(
        already_owned_task.id
    )
    loaded_different = reloaded_repo.get_task(
        different_line_task.id
    )

    assert loaded_unowned is not None
    assert loaded_owned is not None
    assert loaded_different is not None

    assert loaded_unowned.user_id == "web-user-1"
    assert loaded_owned.user_id == "other-web-user"
    assert loaded_different.user_id == ""

def test_claim_next_pending_task_claims_oldest_task(
    tmp_path,
):
    store_file = tmp_path / "store.json"
    repo = Repository(store_file)

    newer_task = AnnotationTask(
        prompt="較新的任務",
        created_at=200.0,
        updated_at=200.0,
    )

    completed_task = AnnotationTask(
        prompt="已完成的任務",
        status="completed",
        created_at=50.0,
        updated_at=50.0,
    )

    older_task = AnnotationTask(
        prompt="較舊的任務",
        created_at=100.0,
        updated_at=100.0,
    )

    repo.add_task(newer_task)
    repo.add_task(completed_task)
    repo.add_task(older_task)

    claimed_task = repo.claim_next_pending_task()

    assert claimed_task is not None
    assert claimed_task.id == older_task.id
    assert claimed_task.status == "processing"
    assert claimed_task.error_message == ""

    reloaded_repo = Repository(store_file)

    loaded_oldest = reloaded_repo.get_task(
        older_task.id
    )
    loaded_newer = reloaded_repo.get_task(
        newer_task.id
    )
    loaded_completed = reloaded_repo.get_task(
        completed_task.id
    )

    assert loaded_oldest is not None
    assert loaded_newer is not None
    assert loaded_completed is not None

    assert loaded_oldest.status == "processing"
    assert loaded_newer.status == "pending"
    assert loaded_completed.status == "completed"