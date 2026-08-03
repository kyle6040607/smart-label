"""標記 / 種子範例 API（提案 demo 第 2 步：點選打字標幾個範例）。"""
from __future__ import annotations

from flask import Blueprint, abort, jsonify, request

from app.routes import (
    get_current_project_id,
    get_current_user_id,
    get_owned_segment,
    get_pipeline,
    get_repo,
)

bp = Blueprint("labels", __name__, url_prefix="/api")


@bp.get("/labels")
def list_labels():
    return jsonify(get_repo().labels(get_current_user_id(), get_current_project_id()))


@bp.post("/segments/<seg_id>/label")
def label_segment(seg_id: str):
    """把某片段標成種子範例，觸發主動學習回訓。body: {"label": str}"""
    repo, pipeline = get_repo(), get_pipeline()
    seg = get_owned_segment(seg_id)
    data = request.get_json(force=True)
    label = (data.get("label") or "").strip()
    if not label:
        abort(400, "label 不可為空")
    ex = pipeline.add_example_from_segment(seg, label)
    return jsonify({"example": ex.to_dict(), "segment": seg.to_dict()}), 201


@bp.delete("/labels/<path:label>")
def delete_label(label: str):
    """刪掉建錯的類別：移除它所有種子範例並回訓。"""
    n = get_pipeline().delete_label(label, get_current_user_id(), get_current_project_id())
    if n == 0:
        abort(404, "查無此類別")
    return jsonify({"deleted": label, "examples_removed": n})


@bp.post("/labels/rename")
def rename_label():
    """重新命名或合併類別標籤。
    body: {"old_label": str, "new_label": str, "combine": bool}
    """
    data = request.get_json(force=True)
    old_label = (data.get("old_label") or "").strip()
    new_label = (data.get("new_label") or "").strip()
    combine = bool(data.get("combine", False))

    if not old_label or not new_label:
        abort(400, "old_label 與 new_label 不可為空")

    if old_label == new_label:
        return jsonify({"renamed": False, "message": "名稱相同"})

    user_id = get_current_user_id()
    project_id = get_current_project_id()
    existing_labels = get_repo().labels(user_id, project_id)

    if new_label in existing_labels and not combine:
        return jsonify({
            "exists": True,
            "message": f"類別「{new_label}」已存在，是否要進行合併？"
        }), 409

    n = get_pipeline().rename_label(old_label, new_label, user_id, project_id)
    return jsonify({
        "renamed": True,
        "old_label": old_label,
        "new_label": new_label,
        "combined": new_label in existing_labels,
    })


@bp.get("/examples")
def list_examples():
    return jsonify([
        example.to_dict()
        for example in get_repo().list_examples(get_current_user_id(), get_current_project_id())
    ])
