"""Routes blueprints 共用小工具。"""
from __future__ import annotations

from flask import current_app

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
