import threading

from app.services.task_runner import TaskRunResult
from scripts.task_worker import run_worker


def test_loop_mode_continues_after_single_task_failure():
    stop_event = threading.Event()
    results = iter([
        TaskRunResult.FAILED,
        TaskRunResult.COMPLETED,
    ])
    calls = []

    def run_task(app):
        calls.append(app)
        result = next(results)
        if result == TaskRunResult.COMPLETED:
            stop_event.set()
        return result

    app = object()
    stats = run_worker(
        app,
        mode="loop",
        poll_seconds=0.01,
        stop_event=stop_event,
        run_task=run_task,
    )

    assert calls == [app, app]
    assert stats.completed == 1
    assert stats.failed == 1


def test_drain_mode_exits_after_queue_is_empty():
    results = iter([
        TaskRunResult.COMPLETED,
        TaskRunResult.COMPLETED,
        TaskRunResult.IDLE,
    ])
    calls = []

    def run_task(app):
        calls.append(app)
        return next(results)

    app = object()
    stats = run_worker(
        app,
        mode="drain",
        poll_seconds=0.01,
        run_task=run_task,
    )

    assert calls == [app, app, app]
    assert stats.completed == 2
    assert stats.failed == 0
