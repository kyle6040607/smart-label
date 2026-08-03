"""以 Web 服務身分觸發 LIFF Cloud Run Job。"""

from __future__ import annotations

from urllib.parse import quote

import google.auth
from google.auth.transport.requests import AuthorizedSession

from app.config import Config


_CLOUD_PLATFORM_SCOPE = "https://www.googleapis.com/auth/cloud-platform"


def trigger_task_worker(config: Config) -> str:
    """觸發背景 Worker，回傳 Google long-running operation 名稱。

    Job 名稱留空代表目前環境未啟用自動觸發。呼叫端應在任務已持久化後
    執行本函式，並在觸發失敗時保留 pending 任務供人工或排程補處理。
    """
    job_name = config.cloud_run_task_job_name
    if not job_name:
        return ""

    project_id = config.cloud_run_task_job_project_id
    region = config.cloud_run_task_job_region
    if not project_id:
        raise ValueError("已設定 CLOUD_RUN_TASK_JOB_NAME，但缺少 Job project ID")
    if not region:
        raise ValueError("已設定 CLOUD_RUN_TASK_JOB_NAME，但缺少 Job region")

    encoded_project = quote(project_id, safe="")
    encoded_region = quote(region, safe="")
    encoded_job = quote(job_name, safe="")
    url = (
        "https://run.googleapis.com/v2/"
        f"projects/{encoded_project}/locations/{encoded_region}/"
        f"jobs/{encoded_job}:run"
    )

    credentials, _ = google.auth.default(scopes=[_CLOUD_PLATFORM_SCOPE])
    with AuthorizedSession(credentials) as session:
        response = session.post(
            url,
            json={},
            timeout=config.cloud_run_task_job_trigger_timeout_seconds,
        )
        response.raise_for_status()
        payload = response.json()

    return str(payload.get("name", ""))
