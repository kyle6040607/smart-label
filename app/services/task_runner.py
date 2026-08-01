"""領取並執行單一 LIFF 背景任務。"""

from __future__ import annotations

from enum import Enum

from app.services.task_notifier import notify_task_completed
from app.services.task_processor import process_task


class TaskRunResult(str, Enum):
    """單次背景任務執行結果。"""

    IDLE = "idle"
    COMPLETED = "completed"
    FAILED = "failed"


def run_next_task(app) -> TaskRunResult:
    """以極短交易領取一個 pending 任務，並在交易外完成耗時處理。"""
    # MySQLRepository 只在 claim_next_pending_task() 內持有
    # FOR UPDATE SKIP LOCKED；函式回傳前已將 processing 狀態 commit。
    task = app.repo.claim_next_pending_task()

    if task is None:
        return TaskRunResult.IDLE

    app.logger.info("開始處理 LIFF 任務：%s", task.id)

    try:
        completed_task = process_task(
            app.repo,
            app.pipeline,
            task,
            app.smart_config.data_dir / "tasks",
            storage=app.storage,
        )
    except Exception as exc:  # noqa: BLE001 背景任務必須記錄失敗後繼續服務
        task.status = "failed"
        task.error_message = str(exc)
        app.repo.update_task(task)
        app.logger.exception(
            "LIFF 任務處理失敗：%s",
            task.id,
        )
        return TaskRunResult.FAILED

    try:
        notified = notify_task_completed(
            app.repo,
            app.smart_config,
            completed_task,
        )
    except Exception:  # noqa: BLE001 通知異常不得推翻已完成的任務
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
