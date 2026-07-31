"""Routes blueprints 共用小工具。"""
from __future__ import annotations

from flask import abort, current_app, session

from app.config import Config
from app.services.pipeline import Pipeline
from app.repository import Repository
from app.storage import StorageService


def get_repo() -> Repository:
    return current_app.repo  # type: ignore[attr-defined]


def get_pipeline() -> Pipeline:
    return current_app.pipeline  # type: ignore[attr-defined]


def get_config() -> Config:
    return current_app.smart_config  # type: ignore[attr-defined]


def get_storage() -> StorageService:
    return current_app.storage  # type: ignore[attr-defined]


def get_current_user_id() -> str:
    """回傳已通過全域 API 認證的使用者 ID。"""
    return str(session.get("user_id") or "")


def get_current_project_id() -> str:
    """回傳目前活躍的專案 ID。若 session 無記錄，自動取得或建立預設專案。"""
    user_id = get_current_user_id()
    if not user_id:
        return ""
    repo = get_repo()
    proj_id = session.get("active_project_id")
    if proj_id:
        proj = repo.get_project(str(proj_id))
        if proj and proj.owner_id == user_id:
            return proj.id

    default_proj = repo.get_or_create_default_project(user_id)
    session["active_project_id"] = default_proj.id
    return default_proj.id


def can_access_image(image) -> bool:
    """圖片只能由擁有者存取；舊的無主圖片僅開放管理者。"""
    user_id = get_current_user_id()
    if image.owner_id:
        return image.owner_id == user_id
    user = get_repo().get_user(user_id)
    return bool(user and user.role == "admin")


def get_owned_image(image_id: str):
    image = get_repo().get_image(image_id)
    if image is None or not can_access_image(image):
        abort(404)
    return image


def get_owned_segment(segment_id: str):
    repo = get_repo()
    segment = repo.get_segment(segment_id)
    if segment is None:
        abort(404)
    get_owned_image(segment.image_id)
    return segment
