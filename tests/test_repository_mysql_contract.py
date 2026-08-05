"""不連外部 MySQL 的 SQL 契約測試。"""

from contextlib import contextmanager
from types import SimpleNamespace

from app.models import AnnotationTask
from app.repository_mysql import MySQLRepository


class RecordingCursor:
    def __init__(self, rows=None):
        self.calls = []
        self.rows = list(rows or [])

    def execute(self, sql, params=None):
        self.calls.append((sql, params or ()))
        return 1

    def fetchone(self):
        return self.rows.pop(0) if self.rows else None

    def fetchall(self):
        rows = self.rows
        self.rows = []
        return rows

    def __enter__(self):
        return self

    def __exit__(self, *args):
        return None


class RecordingConnection:
    def __init__(self, cursor):
        self.cursor_instance = cursor

    def cursor(self):
        return self.cursor_instance


def _repo_with_cursor(cursor):
    repo = object.__new__(MySQLRepository)

    @contextmanager
    def transaction():
        yield cursor

    repo._tx = transaction
    return repo


def test_add_task_supplies_every_sql_placeholder():
    cursor = RecordingCursor()
    _repo_with_cursor(cursor).add_task(AnnotationTask(prompt="cat"))
    sql, params = cursor.calls[0]
    assert sql.count("%s") == len(params)
    assert "no_detection_image_ids" in sql
    assert "claim_token" in sql


def test_claim_uses_skip_locked_and_only_updates_state_fields():
    task = AnnotationTask(prompt="cat").to_dict()
    import json

    for field in (
        "image_ids",
        "processed_image_ids",
        "settings_snapshot",
        "no_detection_image_ids",
        "excluded_results",
    ):
        task[field] = json.dumps(task[field])
    cursor = RecordingCursor([task])
    claimed = _repo_with_cursor(cursor).claim_next_pending_task(
        worker_id="worker-a",
        lease_seconds=900,
        max_attempts=3,
    )
    assert claimed is not None
    select_sql, _ = cursor.calls[0]
    update_sql, _ = cursor.calls[1]
    assert "FOR UPDATE SKIP LOCKED" in select_sql
    assert "claim_token = ''" in select_sql
    assert "status = 'processing'" in update_sql
    assert "dataset_zip_path" not in update_sql


def test_stale_recovery_uses_skip_locked():
    cursor = RecordingCursor([])
    repo = _repo_with_cursor(cursor)
    assert repo.recover_stale_tasks(now=100, max_attempts=3, limit=10) == []
    assert "FOR UPDATE SKIP LOCKED" in cursor.calls[0][0]


def test_stale_recovery_returns_token_for_storage_cleanup():
    import json

    task = AnnotationTask(
        status="processing",
        claim_token="stale-token",
        attempt_count=1,
        lease_expires_at=10,
    ).to_dict()
    for field in (
        "image_ids",
        "processed_image_ids",
        "settings_snapshot",
        "no_detection_image_ids",
        "excluded_results",
    ):
        task[field] = json.dumps(task[field])
    cursor = RecordingCursor([task])

    recovered = _repo_with_cursor(cursor).recover_stale_tasks(
        now=20,
        max_attempts=3,
        limit=10,
    )

    assert recovered[0].attempt_token == "stale-token"
    assert recovered[0].task.claim_token == "stale-token"
    assert recovered[0].task.status == "retry_wait"


def test_schema_lock_uses_mysql_advisory_lock_and_releases_it():
    cursor = RecordingCursor([{"acquired": 1}])
    repo = object.__new__(MySQLRepository)
    repo._cfg = SimpleNamespace(mysql_database="smart_label_db")
    repo._conn = lambda: RecordingConnection(cursor)

    with repo._schema_lock():
        pass

    assert "GET_LOCK" in cursor.calls[0][0]
    assert "RELEASE_LOCK" in cursor.calls[-1][0]
    assert cursor.calls[0][1][0] == cursor.calls[-1][1][0]


def test_finish_recovered_cleanup_clears_matching_token():
    cursor = RecordingCursor()
    repo = _repo_with_cursor(cursor)

    assert repo.finish_recovered_task_cleanup("task-1", "stale-token")

    sql, params = cursor.calls[0]
    assert "claim_token=''" in sql
    assert "claim_token=%s" in sql
    assert params[-1] == "stale-token"


def test_record_liff_upload_batch_persists_uploaded_bytes_atomically():
    import json

    task = AnnotationTask(
        id="upload-session",
        line_user_id="U-owner",
        status="uploading",
        settings_snapshot={
            "upload": {
                "expected_image_count": 1,
                "expected_total_bytes": 321,
                "uploaded_bytes": 0,
                "completed_batches": {},
                "completed_batch_bytes": {},
            }
        },
    ).to_dict()
    for field in (
        "image_ids",
        "processed_image_ids",
        "settings_snapshot",
        "no_detection_image_ids",
        "excluded_results",
    ):
        task[field] = json.dumps(task[field])

    cursor = RecordingCursor([task])
    updated, recorded = _repo_with_cursor(cursor).record_liff_upload_batch(
        "upload-session",
        "U-owner",
        "batch-1",
        ["image-1"],
        321,
    )

    assert recorded is True
    assert updated is not None
    assert updated.settings_snapshot["upload"]["uploaded_bytes"] == 321
    assert updated.settings_snapshot["upload"]["completed_batch_bytes"] == {
        "batch-1": 321,
    }
    assert "FOR UPDATE" in cursor.calls[0][0]
    saved_snapshot = json.loads(cursor.calls[1][1][1])
    assert saved_snapshot["upload"]["uploaded_bytes"] == 321


def test_finalize_liff_append_locks_session_and_target_in_one_transaction():
    import json

    target = AnnotationTask(
        id="target-task",
        line_user_id="U-owner",
        image_ids=["old-image"],
        processed_image_ids=["old-image"],
        status="completed",
        attempt_count=2,
    ).to_dict()
    session = AnnotationTask(
        id="append-session",
        line_user_id="U-owner",
        image_ids=["new-image"],
        status="uploading",
        settings_snapshot={
            "upload": {
                "target_task_id": "target-task",
                "expected_image_count": 1,
            }
        },
    ).to_dict()
    for row in (session, target):
        for field in (
            "image_ids",
            "processed_image_ids",
            "settings_snapshot",
            "no_detection_image_ids",
            "excluded_results",
        ):
            row[field] = json.dumps(row[field])

    cursor = RecordingCursor([session, target])
    merged, transitioned = _repo_with_cursor(
        cursor
    ).finalize_liff_append_upload("append-session", "U-owner")

    assert transitioned is True
    assert merged is not None
    assert merged.id == "target-task"
    assert merged.status == "pending"
    assert merged.image_ids == ["old-image", "new-image"]
    assert merged.processed_image_ids == ["old-image"]
    assert merged.attempt_count == 0
    assert len(cursor.calls) == 4
    assert all("FOR UPDATE" in call[0] for call in cursor.calls[:2])
    assert "status='pending'" in cursor.calls[2][0]
    assert "status='upload_merged'" in cursor.calls[3][0]
