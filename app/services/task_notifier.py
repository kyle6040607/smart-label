"""LIFF 任務完成後的 LINE 通知。"""

from urllib.parse import quote

from app.config import Config
from app.models import AnnotationTask
from app.repository import Repository
from app.routes.line_bot import push_task_download


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
    if (
        task.status != "completed"
        or not task.line_user_id
        or not task.dataset_zip_path
        or task.dataset_version <= 0
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
