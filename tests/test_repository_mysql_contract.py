"""不連外部 MySQL 的 SQL 契約測試。"""

from contextlib import contextmanager

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
    assert "status = 'processing'" in update_sql
    assert "dataset_zip_path" not in update_sql


def test_stale_recovery_uses_skip_locked():
    cursor = RecordingCursor([])
    repo = _repo_with_cursor(cursor)
    assert repo.recover_stale_tasks(now=100, max_attempts=3, limit=10) == []
    assert "FOR UPDATE SKIP LOCKED" in cursor.calls[0][0]
