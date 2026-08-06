"""專案 / 會話管理 API 路由。"""
from __future__ import annotations

from flask import Blueprint, abort, jsonify, request, session

from app.models import Project
from app.routes import (
    get_current_project_id,
    get_current_user_id,
    get_repo,
    get_storage,
)

bp = Blueprint("projects", __name__, url_prefix="/api/projects")


@bp.get("")
def list_projects():
    """列出當前使用者的所有專案，並標明目前的 active_project_id。"""
    user_id = get_current_user_id()
    if not user_id:
        abort(401, "請先登入")
    repo = get_repo()

    # 確保使用者至少擁有一個專案（無專案時自動建立預設專案）
    active_id = get_current_project_id()
    projects = repo.list_projects_by_owner(user_id)

    return jsonify({
        "active_project_id": active_id,
        "projects": [p.to_dict() for p in projects],
    })


@bp.post("")
def create_project():
    """建立新專案，並自動切換為當前活躍專案。"""
    user_id = get_current_user_id()
    if not user_id:
        abort(401, "請先登入")

    data = request.get_json(silent=True) or {}
    name = (data.get("name") or "").strip() or "新專案"
    mode = data.get("mode") if data.get("mode") in ("novice", "engineer") else "novice"

    repo = get_repo()
    existing_projects = repo.list_projects_by_owner(user_id)
    if any(p.name.strip().lower() == name.lower() for p in existing_projects):
        return jsonify({"error": "專案名稱已存在，請使用其他名稱"}), 400

    proj = Project(
        owner_id=user_id,
        name=name,
        mode=mode,
    )
    repo.add_project(proj)

    session["active_project_id"] = proj.id
    return jsonify(proj.to_dict()), 201


@bp.get("/<project_id>")
def get_project(project_id: str):
    """取得單一專案詳細資訊。"""
    user_id = get_current_user_id()
    repo = get_repo()
    proj = repo.get_project(project_id)
    if proj is None or proj.owner_id != user_id:
        abort(404, "查無此專案")
    return jsonify(proj.to_dict())


@bp.put("/<project_id>")
def update_project(project_id: str):
    """修改專案名稱或模式。"""
    user_id = get_current_user_id()
    repo = get_repo()
    proj = repo.get_project(project_id)
    if proj is None or proj.owner_id != user_id:
        abort(404, "查無此專案")

    data = request.get_json(silent=True) or {}
    if "name" in data:
        name = (data["name"] or "").strip()
        if name:
            existing_projects = repo.list_projects_by_owner(user_id)
            if any(p.name.strip().lower() == name.lower() and p.id != project_id for p in existing_projects):
                return jsonify({"error": "專案名稱已存在，請使用其他名稱"}), 400
            proj.name = name
    if "mode" in data and data["mode"] in ("novice", "engineer"):
        proj.mode = data["mode"]

    repo.update_project(proj)
    return jsonify(proj.to_dict())


@bp.delete("/<project_id>")
def delete_project(project_id: str):
    """刪除專案及其所有圖檔、遮罩與訓練範例。"""
    user_id = get_current_user_id()
    repo, storage = get_repo(), get_storage()
    proj = repo.get_project(project_id)
    if proj is None or proj.owner_id != user_id:
        abort(404, "查無此專案")

    liff_task_ids = [
        task.id
        for task in repo.list_tasks_by_user(user_id)
        if task.project_id == project_id and task.line_user_id
    ]

    # 刪除 DB 紀錄並取得所有需清理的磁碟檔案
    paths = repo.delete_project(project_id)
    files_removed = sum(storage.delete(p) for p in paths if p)
    for task_id in liff_task_ids:
        for prefix in (
            f"liff-uploads/{task_id}",
            f"previews/tasks/{task_id}",
            f"datasets/{task_id}",
        ):
            files_removed += storage.delete_prefix(prefix)

    # 若被刪除的是當前活躍專案，自動切換至其他專案或預設專案
    if session.get("active_project_id") == project_id:
        session.pop("active_project_id", None)
        new_active_id = get_current_project_id()
    else:
        new_active_id = session.get("active_project_id", "")

    return jsonify({
        "deleted_id": project_id,
        "files_removed": files_removed,
        "active_project_id": new_active_id,
    })


@bp.post("/<project_id>/select")
def select_project(project_id: str):
    """切換當前活躍專案。"""
    user_id = get_current_user_id()
    repo = get_repo()
    proj = repo.get_project(project_id)
    if proj is None or proj.owner_id != user_id:
        abort(404, "查無此專案")

    session["active_project_id"] = proj.id
    return jsonify({
        "active_project_id": proj.id,
        "project": proj.to_dict(),
    })
