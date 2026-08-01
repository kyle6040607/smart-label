"""YOLOv26x-seg 模型訓練 API 路由。"""
from __future__ import annotations

import threading
from flask import Blueprint, abort, jsonify, request

from app.models import AnnotationTask
from app.routes import (
    get_config,
    get_current_user_id,
    get_repo,
    get_storage,
)
from app.ml.yolov26x_seg import train_yolov26x_seg

bp = Blueprint("training", __name__, url_prefix="/api/projects")


@bp.post("/<project_id>/train")
def trigger_training(project_id: str):
    """觸發特定專案的 YOLOv26x-seg 訓練任務。"""
    user_id = get_current_user_id()
    if not user_id:
        abort(401, "請先登入")

    repo = get_repo()
    proj = repo.get_project(project_id)
    if proj is None or proj.owner_id != user_id:
        abort(404, "查無此專案")

    # 檢查是否有可供訓練的已標註數據
    labeled_segs = repo.list_labeled_segments_by_project(user_id, project_id)
    if not labeled_segs:
        return jsonify({
            "error": "目前專案沒有可供訓練的已標註片段 (final_label 需有值)",
            "labeled_count": 0,
        }), 400

    data = request.get_json(silent=True) or {}
    epochs = int(data.get("epochs", 5))
    imgsz = int(data.get("imgsz", 640))
    device = str(data.get("device", "auto"))

    task = AnnotationTask(
        user_id=user_id,
        project_id=project_id,
        prompt=f"YOLOv26x-seg 訓練 ({proj.name})",
        status="pending",
    )
    repo.add_task(task)
    storage = get_storage()

    # 在同步模式或非同步背景執行訓練
    def run_training_in_background():
        try:
            train_yolov26x_seg(
                repo=repo,
                storage=storage,
                task=task,
                epochs=epochs,
                imgsz=imgsz,
                device=device,
            )
        except Exception:
            pass

    # 非同步背景啟動訓練
    thread = threading.Thread(target=run_training_in_background, daemon=True)
    thread.start()

    return jsonify({
        "message": "訓練任務已成功觸發並開始執行",
        "task_id": task.id,
        "project_id": project_id,
        "labeled_count": len(labeled_segs),
        "status": task.status,
    }), 202


@bp.get("/<project_id>/train/status")
def get_training_status(project_id: str):
    """查詢專案最新的 YOLOv26x-seg 訓練狀態。"""
    user_id = get_current_user_id()
    if not user_id:
        abort(401, "請先登入")

    repo = get_repo()
    proj = repo.get_project(project_id)
    if proj is None or proj.owner_id != user_id:
        abort(404, "查無此專案")

    # 尋找該專案最新一筆任務
    tasks = [
        t for t in repo.tasks.values()
        if t.user_id == user_id and getattr(t, "project_id", "") == project_id
    ] if hasattr(repo, "tasks") else []

    if not tasks:
        # 相容 MySQLRepository 查詢
        all_user_tasks = [t for t in repo.list_tasks_by_user(user_id)] if hasattr(repo, "list_tasks_by_user") else []
        tasks = [t for t in all_user_tasks if getattr(t, "project_id", "") == project_id]

    if not tasks:
        return jsonify({
            "project_id": project_id,
            "has_training": False,
            "message": "該專案目前沒有進行過訓練任務",
        })

    latest_task = sorted(tasks, key=lambda t: t.created_at, reverse=True)[0]
    return jsonify({
        "project_id": project_id,
        "has_training": True,
        "task_id": latest_task.id,
        "status": latest_task.status,
        "best_model_path": latest_task.best_model_path,
        "error_message": latest_task.error_message,
        "updated_at": latest_task.updated_at,
    })


@bp.get("/<project_id>/train/download")
def download_training_model(project_id: str):
    """下載專案最新的 YOLOv26x-seg 訓練模型權重檔 (.pt)。"""
    import io
    from pathlib import Path
    from flask import send_file

    user_id = get_current_user_id()
    if not user_id:
        abort(401, "請先登入")

    repo = get_repo()
    proj = repo.get_project(project_id)
    if proj is None or proj.owner_id != user_id:
        abort(404, "查無此專案")

    tasks = [
        t for t in repo.tasks.values()
        if t.user_id == user_id and getattr(t, "project_id", "") == project_id
    ] if hasattr(repo, "tasks") else []

    if not tasks and hasattr(repo, "list_tasks_by_user"):
        all_user_tasks = repo.list_tasks_by_user(user_id)
        tasks = [t for t in all_user_tasks if getattr(t, "project_id", "") == project_id]

    completed_tasks = [t for t in tasks if t.status == "completed" and t.best_model_path]
    if not completed_tasks:
        abort(404, "該專案尚無訓練完成的模型可供下載")

    latest_task = sorted(completed_tasks, key=lambda t: t.created_at, reverse=True)[0]
    storage = get_storage()

    filename = Path(latest_task.best_model_path).name or "yolo26x_seg_model.pt"
    try:
        model_bytes = storage.read_bytes(latest_task.best_model_path)
    except Exception:
        abort(404, f"無法從儲存庫讀取模型檔案：{latest_task.best_model_path}")

    return send_file(
        io.BytesIO(model_bytes),
        mimetype="application/octet-stream",
        as_attachment=True,
        download_name=filename,
    )
