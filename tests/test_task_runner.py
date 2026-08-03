import threading
from types import SimpleNamespace

from app.models import AnnotationTask, RecoveredTaskAttempt
from app.services import task_runner
from app.services.task_runner import TaskRunResult


class RecordingLogger:
    def __init__(self):
        self.messages = []

    def info(self, message, *args):
        self.messages.append(("info", message, args))

    def warning(self, message, *args):
        self.messages.append(("warning", message, args))

    def exception(self, message, *args):
        self.messages.append(("exception", message, args))


class SingleTaskRepo:
    def __init__(self, task=None):
        self.task = task
        self.updated_tasks = []
        self.claim_transaction_active = False
        self.claimed_task = None

    def claim_next_pending_task(
        self,
        worker_id="worker",
        lease_seconds=900,
        max_attempts=3,
    ):
        del worker_id, lease_seconds, max_attempts
        if self.task is None:
            return None

        # 模擬 MySQL claim 的短交易：processing 狀態完成後才回傳。
        self.claim_transaction_active = True
        task = self.task
        self.task = None
        task.status = "processing"
        task.attempt_count += 1
        task.claim_token = "claim-token"
        self.claimed_task = task
        self.claim_transaction_active = False
        return task

    def heartbeat_task(self, task_id, claim_token, lease_seconds):
        del task_id, claim_token, lease_seconds
        return True

    def list_segments(self):
        return []

    def fail_or_retry_task(
        self,
        task_id,
        claim_token,
        error_message,
        *,
        max_attempts,
        retry_at,
    ):
        del task_id, claim_token
        task = self.claimed_task
        if task is None:
            return None
        task.error_message = error_message
        task.last_error = error_message
        task.status = (
            "failed" if task.attempt_count >= max_attempts else "retry_wait"
        )
        task.next_attempt_at = retry_at if task.status == "retry_wait" else 0
        return task

    def update_task(self, task):
        self.updated_tasks.append(task)
        return task


def make_app(tmp_path, task=None):
    return SimpleNamespace(
        repo=SingleTaskRepo(task),
        pipeline=object(),
        smart_config=SimpleNamespace(
            data_dir=tmp_path,
            task_lease_seconds=900,
            task_heartbeat_seconds=60,
            task_max_attempts=3,
            task_retry_base_seconds=60,
        ),
        storage=object(),
        logger=RecordingLogger(),
    )


def test_run_next_task_returns_idle_without_pending_task(tmp_path):
    app = make_app(tmp_path)

    assert task_runner.run_next_task(app) == TaskRunResult.IDLE


def test_run_next_task_completes_claimed_task_outside_claim_transaction(
    tmp_path,
    monkeypatch,
):
    task = AnnotationTask(prompt="cat")
    app = make_app(tmp_path, task)

    def fake_process_task(
        repo, pipeline, claimed_task, output_dir, storage=None, ensure_lease=None
    ):
        del pipeline, output_dir, storage, ensure_lease
        assert repo.claim_transaction_active is False
        assert claimed_task.status == "processing"
        claimed_task.status = "completed"
        return claimed_task

    monkeypatch.setattr(task_runner, "process_task", fake_process_task)
    monkeypatch.setattr(
        task_runner,
        "notify_task_completed",
        lambda *args: False,
    )

    assert task_runner.run_next_task(app) == TaskRunResult.COMPLETED
    assert task.status == "completed"


def test_run_next_task_schedules_retry_after_processing_failure(tmp_path, monkeypatch):
    task = AnnotationTask(prompt="dog")
    app = make_app(tmp_path, task)

    def fail_processing(*args, **kwargs):
        raise RuntimeError("model failed")

    monkeypatch.setattr(task_runner, "process_task", fail_processing)

    # SingleTaskRepo 的 claim 測試替身需模擬 attempt 計數。
    assert task_runner.run_next_task(app) == TaskRunResult.RETRY_SCHEDULED
    assert task.status == "retry_wait"
    assert task.error_message == "model failed"
    assert app.repo.claimed_task is task


def test_run_next_task_sends_completion_notification(tmp_path, monkeypatch):
    task = AnnotationTask(prompt="bird")
    app = make_app(tmp_path, task)
    notifications = []

    def complete_task(
        repo, pipeline, claimed_task, output_dir, storage=None, ensure_lease=None
    ):
        del repo, pipeline, output_dir, storage, ensure_lease
        claimed_task.status = "completed"
        return claimed_task

    def record_notification(repo, config, completed_task):
        notifications.append((repo, config, completed_task))
        return True

    monkeypatch.setattr(task_runner, "process_task", complete_task)
    monkeypatch.setattr(
        task_runner,
        "notify_task_completed",
        record_notification,
    )

    assert task_runner.run_next_task(app) == TaskRunResult.COMPLETED
    assert notifications == [(app.repo, app.smart_config, task)]


def test_run_next_task_marks_third_failure_and_notifies(tmp_path, monkeypatch):
    task = AnnotationTask(prompt="cat", attempt_count=2)
    app = make_app(tmp_path, task)
    notifications = []
    monkeypatch.setattr(
        task_runner,
        "process_task",
        lambda *args, **kwargs: (_ for _ in ()).throw(
            RuntimeError("permanent failure")
        ),
    )
    monkeypatch.setattr(
        task_runner,
        "notify_task_failed",
        lambda repo, config, failed: notifications.append(failed) or True,
    )
    assert task_runner.run_next_task(app) == TaskRunResult.FAILED
    assert task.status == "failed"
    assert task.attempt_count == 3
    assert notifications == [task]


def test_task_heartbeat_closes_thread_connection():
    heartbeat_sent = threading.Event()

    class Repo:
        def __init__(self):
            self.calls = 0
            self.closed = False

        def heartbeat_task(self, *args):
            self.calls += 1
            heartbeat_sent.set()
            return True

        def close_thread_connection(self):
            self.closed = True

    repo = Repo()
    task = AnnotationTask(status="processing", claim_token="token")
    with task_runner.TaskHeartbeat(repo, task, 0.001, 60):
        assert heartbeat_sent.wait(1)
    assert repo.calls >= 1
    assert repo.closed


def test_recovery_cleans_stale_attempt_before_requeue(tmp_path, monkeypatch):
    task = AnnotationTask(id="task-1", status="retry_wait")
    finished = []
    repo = SimpleNamespace(
        recover_stale_tasks=lambda **kwargs: [
            RecoveredTaskAttempt(task, "stale-token")
        ],
        finish_recovered_task_cleanup=lambda task_id, token: (
            finished.append((task_id, token)) or True
        ),
    )
    app = SimpleNamespace(
        repo=repo,
        storage=object(),
        smart_config=SimpleNamespace(
            task_max_attempts=3,
            task_recovery_batch_size=10,
        ),
        logger=RecordingLogger(),
    )
    cleaned = []
    monkeypatch.setattr(
        task_runner,
        "cleanup_task_attempt",
        lambda repo, storage, task_id, token: cleaned.append(
            (repo, storage, task_id, token)
        ),
    )

    assert task_runner.recover_stale_tasks(app) == (1, 0)
    assert cleaned == [(repo, app.storage, "task-1", "stale-token")]
    assert finished == [("task-1", "stale-token")]


def test_recovery_keeps_cleanup_pending_when_storage_delete_fails(
    tmp_path,
    monkeypatch,
):
    task = AnnotationTask(id="task-1", status="retry_wait")
    finished = []
    repo = SimpleNamespace(
        recover_stale_tasks=lambda **kwargs: [
            RecoveredTaskAttempt(task, "stale-token")
        ],
        finish_recovered_task_cleanup=lambda *args: finished.append(args),
    )
    app = SimpleNamespace(
        repo=repo,
        storage=object(),
        smart_config=SimpleNamespace(
            task_max_attempts=3,
            task_recovery_batch_size=10,
        ),
        logger=RecordingLogger(),
    )
    monkeypatch.setattr(
        task_runner,
        "cleanup_task_attempt",
        lambda *args: (_ for _ in ()).throw(RuntimeError("GCS unavailable")),
    )

    assert task_runner.recover_stale_tasks(app) == (0, 0)
    assert finished == []
