"""LIFF 任務完成後的 LINE 通知。"""

import time
from urllib.parse import quote

from app.config import Config
from app.models import AnnotationTask
from app.repository import Repository
from app.routes.line_bot import (
    push_task_download,
    push_task_failed,
    push_task_no_export,
)


def build_task_download_url(config: Config, task: AnnotationTask) -> str:
    """建立可以放進 LINE 按鈕的絕對下載網址。"""
    base_url = config.public_base_url.rstrip("/")
    task_id = quote(task.id, safe="")
    token = quote(task.download_token, safe="")

    return f"{base_url}/liff/tasks/{task_id}/download?token={token}&openExternalBrowser=1"


def notify_task_completed(
    repo: Repository,
    config: Config,
    task: AnnotationTask,
) -> bool:
    """推播尚未通知過的最新資料集版本。"""
    if task.status != "completed" or not task.line_user_id:
        return False

    if not task.dataset_zip_path:
        if task.notified_dataset_version < 0:
            return False
        sent = push_task_no_export(
            line_user_id=task.line_user_id,
            task_id=task.id,
            excluded_count=task.excluded_count,
        )
        if not sent:
            return False
        task.notified_dataset_version = -max(1, task.dataset_version + 1)
        repo.update_task(task)
        return True

    if (
        task.dataset_version <= 0
        or task.dataset_version <= task.notified_dataset_version
    ):
        return False

    sent = push_task_download(
        line_user_id=task.line_user_id,
        task_id=task.id,
        dataset_version=task.dataset_version,
        download_url=build_task_download_url(config, task),
    )

    if not sent:
        return False

    task.notified_dataset_version = task.dataset_version
    repo.update_task(task)
    return True


def notify_task_failed(
    repo: Repository,
    config: Config,
    task: AnnotationTask,
) -> bool:
    """最終 failed 僅推播一次；一般 retry_wait 不通知。"""
    del config
    if (
        task.status != "failed"
        or not task.line_user_id
        or task.failure_notified_at > 0
    ):
        return False
    sent = push_task_failed(
        line_user_id=task.line_user_id,
        task_id=task.id,
        error_message=task.last_error or task.error_message,
    )
    if not sent:
        return False
    task.failure_notified_at = time.time()
    repo.update_task(task)
    return True
