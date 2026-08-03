"""領取並執行單一 LIFF 背景任務。"""

from __future__ import annotations

import threading
import time
from enum import Enum

from app.services.task_notifier import (
    notify_task_completed,
    notify_task_failed,
)
from app.services.task_processor import (
    TaskLeaseLostError,
    cleanup_task_attempt,
    process_task,
)


class TaskRunResult(str, Enum):
    IDLE = "idle"
    COMPLETED = "completed"
    RETRY_SCHEDULED = "retry_scheduled"
    FAILED = "failed"
    LEASE_LOST = "lease_lost"


class TaskHeartbeat:
    """每個任務只建立一條 heartbeat thread，結束時確實回收連線。"""

    def __init__(self, repo, task, interval_seconds: float, lease_seconds: float):
        self.repo = repo
        self.task = task
        self.interval_seconds = interval_seconds
        self.lease_seconds = lease_seconds
        self._stop = threading.Event()
        self._lost = threading.Event()
        self._thread: threading.Thread | None = None

    def __enter__(self):
        if self.task.claim_token and hasattr(self.repo, "heartbeat_task"):
            self._thread = threading.Thread(
                target=self._run,
                name=f"task-heartbeat-{self.task.id}",
                daemon=True,
            )
            self._thread.start()
        return self

    def _run(self) -> None:
        try:
            while not self._stop.wait(self.interval_seconds):
                try:
                    renewed = self.repo.heartbeat_task(
                        self.task.id,
                        self.task.claim_token,
                        self.lease_seconds,
                    )
                except Exception:
                    # 單次 Cloud SQL 短暫錯誤不立刻宣告 lease 遺失；15 分鐘
                    # lease 會提供多次重試空間。
                    continue
                if not renewed:
                    self._lost.set()
                    return
        finally:
            close_connection = getattr(
                self.repo,
                "close_thread_connection",
                None,
            )
            if close_connection is not None:
                close_connection()

    def ensure_active(self) -> None:
        if self._lost.is_set():
            raise TaskLeaseLostError("任務已被其他 Worker 重新領取")

    def __exit__(self, exc_type, exc, traceback) -> None:
        del exc_type, exc, traceback
        self._stop.set()
        if self._thread is not None:
            self._thread.join()


def _notify_final_failure(app, task) -> None:
    try:
        notify_task_failed(app.repo, app.smart_config, task)
    except Exception:
        app.logger.exception("LIFF 最終失敗通知發生例外：%s", task.id)


def recover_stale_tasks(app) -> tuple[int, int]:
    """執行一次排他回收；回傳 (重新排隊數, 最終失敗數)。"""
    repo = getattr(app, "repo", None)
    if repo is None or not hasattr(repo, "recover_stale_tasks"):
        return 0, 0
    config = app.smart_config
    recovered = repo.recover_stale_tasks(
        max_attempts=getattr(config, "task_max_attempts", 3),
        limit=getattr(config, "task_recovery_batch_size", 10),
    )
    retry_count = 0
    failed_count = 0
    for recovered_attempt in recovered:
        task = recovered_attempt.task
        if not _safe_cleanup(
            app,
            task.id,
            recovered_attempt.attempt_token,
        ):
            continue
        finish_cleanup = getattr(
            repo,
            "finish_recovered_task_cleanup",
            None,
        )
        if finish_cleanup is not None and not finish_cleanup(
            task.id,
            recovered_attempt.attempt_token,
        ):
            continue
        task.claim_token = ""
        if task.status == "failed":
            failed_count += 1
            _notify_final_failure(app, task)
        else:
            retry_count += 1
    if recovered:
        app.logger.info(
            "回收逾時 LIFF 任務：retry=%s failed=%s",
            retry_count,
            failed_count,
        )
    return retry_count, failed_count


def run_next_task(app, worker_id: str | None = None) -> TaskRunResult:
    """以極短交易領取任務，在交易外推論、產縮圖與 ZIP。"""
    config = app.smart_config
    worker_id = worker_id or getattr(app, "task_worker_id", "worker")
    task = app.repo.claim_next_pending_task(
        worker_id=worker_id,
        lease_seconds=getattr(config, "task_lease_seconds", 900.0),
        max_attempts=getattr(config, "task_max_attempts", 3),
    )
    if task is None:
        return TaskRunResult.IDLE

    app.logger.info(
        "開始處理 LIFF 任務：%s attempt=%s worker=%s",
        task.id,
        task.attempt_count,
        worker_id,
    )
    claim_token = task.claim_token

    try:
        with TaskHeartbeat(
            app.repo,
            task,
            getattr(config, "task_heartbeat_seconds", 60.0),
            getattr(config, "task_lease_seconds", 900.0),
        ) as heartbeat:
            completed_task = process_task(
                app.repo,
                app.pipeline,
                task,
                config.data_dir / "tasks",
                storage=app.storage,
                ensure_lease=heartbeat.ensure_active,
            )
    except TaskLeaseLostError:
        _safe_cleanup(app, task.id, claim_token)
        app.logger.warning("LIFF 任務 lease 已遺失，放棄 attempt：%s", task.id)
        return TaskRunResult.LEASE_LOST
    except Exception as exc:  # noqa: BLE001 Worker 必須記錄後繼續下一筆
        _safe_cleanup(app, task.id, claim_token)
        retry_delay = getattr(config, "task_retry_base_seconds", 60.0) * (
            5 ** max(task.attempt_count - 1, 0)
        )
        failed_task = app.repo.fail_or_retry_task(
            task.id,
            claim_token,
            str(exc),
            max_attempts=getattr(config, "task_max_attempts", 3),
            retry_at=time.time() + retry_delay,
        )
        if failed_task is None:
            app.logger.warning("LIFF 任務失敗時 lease 已遺失：%s", task.id)
            return TaskRunResult.LEASE_LOST
        app.logger.exception(
            "LIFF 任務處理失敗：%s attempt=%s status=%s",
            task.id,
            task.attempt_count,
            failed_task.status,
        )
        if failed_task.status == "failed":
            _notify_final_failure(app, failed_task)
            return TaskRunResult.FAILED
        return TaskRunResult.RETRY_SCHEDULED

    try:
        notified = notify_task_completed(
            app.repo,
            config,
            completed_task,
        )
    except Exception:
        app.logger.exception(
            "LIFF 任務已完成，但 LINE 通知發生例外：%s",
            completed_task.id,
        )
    else:
        if not notified:
            app.logger.warning(
                "LIFF 任務已完成，但 LINE 通知未送出：%s",
                completed_task.id,
            )

    app.logger.info("LIFF 任務處理完成：%s", completed_task.id)
    return TaskRunResult.COMPLETED


def _safe_cleanup(app, task_id: str, claim_token: str) -> bool:
    try:
        cleanup_task_attempt(app.repo, app.storage, task_id, claim_token)
    except Exception:
        app.logger.exception("清理 LIFF attempt 暫存結果失敗：%s", task_id)
        return False
    return True
