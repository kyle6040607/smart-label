from types import SimpleNamespace

from app.models import AnnotationTask
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

    def claim_next_pending_task(self):
        if self.task is None:
            return None

        # 模擬 MySQL claim 的短交易：processing 狀態完成後才回傳。
        self.claim_transaction_active = True
        task = self.task
        self.task = None
        task.status = "processing"
        self.claim_transaction_active = False
        return task

    def update_task(self, task):
        self.updated_tasks.append(task)
        return task


def make_app(tmp_path, task=None):
    return SimpleNamespace(
        repo=SingleTaskRepo(task),
        pipeline=object(),
        smart_config=SimpleNamespace(data_dir=tmp_path),
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

    def fake_process_task(repo, pipeline, claimed_task, output_dir, storage=None):
        del pipeline, output_dir, storage
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


def test_run_next_task_marks_processing_failure(tmp_path, monkeypatch):
    task = AnnotationTask(prompt="dog")
    app = make_app(tmp_path, task)

    def fail_processing(*args, **kwargs):
        raise RuntimeError("model failed")

    monkeypatch.setattr(task_runner, "process_task", fail_processing)

    assert task_runner.run_next_task(app) == TaskRunResult.FAILED
    assert task.status == "failed"
    assert task.error_message == "model failed"
    assert app.repo.updated_tasks == [task]


def test_run_next_task_sends_completion_notification(tmp_path, monkeypatch):
    task = AnnotationTask(prompt="bird")
    app = make_app(tmp_path, task)
    notifications = []

    def complete_task(repo, pipeline, claimed_task, output_dir, storage=None):
        del repo, pipeline, output_dir, storage
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
