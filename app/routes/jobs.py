"""批量分割 API：送出一批圖 → 拿 job_id → 輪詢進度。

所有輸入驗證（400/404）都在建立 job 之前同步做完；
背景執行期的單張失敗記在 job.failed，不影響 HTTP 語意。
"""
from __future__ import annotations

from flask import Blueprint, abort, jsonify, request

from app.models import SegmentJob
from app.routes import get_job_runner, get_repo
from app.routes.auth import api_login_required

bp = Blueprint("jobs", __name__, url_prefix="/api")


@bp.post("/segment_jobs")
@api_login_required
def create_segment_job():
    """建立批量分割工作。

    body:
      {"image_ids": [...]}            指定圖片
      或 {"scope": "all"}             所有圖片
      或 {"scope": "unprocessed"}     尚未有任何片段的圖片
      加上可省略的 "prompt": "cat"    有 prompt 逐張文字分割，否則自動分割整張

    立刻回 202 與 job 內容，實際處理在背景排隊進行。
    """
    repo = get_repo()
    data = request.get_json(silent=True) or {}

    # 只有一個模型 worker，同時多個 job 只會排隊、前端也難以呈現，直接擋下
    if any(j.status in ("queued", "running") for j in repo.list_jobs()):
        abort(409, "已有批量分割工作進行中，請等它完成再送出")

    scope = data.get("scope")
    if scope is not None:
        if scope not in ("all", "unprocessed"):
            abort(400, "scope 只能是 all 或 unprocessed")
        image_ids = [i.id for i in repo.list_images()]
        if scope == "unprocessed":
            image_ids = [i for i in image_ids if not repo.list_segments(i)]
        if not image_ids:
            abort(400, "沒有符合條件的圖片")
    else:
        raw_ids = data.get("image_ids")
        if not isinstance(raw_ids, list) or not raw_ids:
            abort(400, "image_ids 不可為空")
        image_ids = list(dict.fromkeys(str(i) for i in raw_ids))  # 去重、保序

        missing = [i for i in image_ids if repo.get_image(i) is None]
        if missing:
            abort(404, f"找不到圖片：{', '.join(missing)}")

    prompt = str(data.get("prompt") or "").strip() or None
    if prompt and len(prompt) > 200:
        abort(400, "prompt 不可超過 200 個字元")

    job = SegmentJob(image_ids=image_ids, prompt=prompt)
    repo.add_job(job)
    payload = job.to_dict()  # submit 前先快照，背景 worker 可能立刻改動狀態
    get_job_runner().submit(job)
    return jsonify(payload), 202


@bp.get("/segment_jobs/<job_id>")
@api_login_required
def get_segment_job(job_id: str):
    job = get_repo().get_job(job_id)
    if job is None:
        abort(404)
    return jsonify(job.to_dict())


@bp.get("/segment_jobs")
@api_login_required
def list_segment_jobs():
    """最近的工作在前；前端重整頁面後靠這個找回進行中的批量工作。"""
    return jsonify([j.to_dict() for j in get_repo().list_jobs()])
