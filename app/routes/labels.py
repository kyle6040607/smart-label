"""標記 / 種子範例 API（提案 demo 第 2 步：點選打字標幾個範例）。"""
from __future__ import annotations

from flask import Blueprint, abort, jsonify, request

from app.routes import get_pipeline, get_repo
from app.routes.auth import api_login_required

bp = Blueprint("labels", __name__, url_prefix="/api")
bp.before_request(api_login_required)


@bp.get("/labels")
def list_labels():
    return jsonify(get_repo().labels())


@bp.post("/segments/<seg_id>/label")
def label_segment(seg_id: str):
    """把某片段標成種子範例，觸發主動學習回訓。body: {"label": str}"""
    repo, pipeline = get_repo(), get_pipeline()
    seg = repo.get_segment(seg_id)
    if not seg:
        abort(404)
    data = request.get_json(force=True)
    label = (data.get("label") or "").strip()
    if not label:
        abort(400, "label 不可為空")
    ex = pipeline.add_example_from_segment(seg, label)
    return jsonify({"example": ex.to_dict(), "segment": seg.to_dict()}), 201


@bp.delete("/labels/<path:label>")
def delete_label(label: str):
    """刪掉建錯的類別：移除它所有種子範例並回訓。"""
    n = get_pipeline().delete_label(label)
    if n == 0:
        abort(404, "查無此類別")
    return jsonify({"deleted": label, "examples_removed": n})


@bp.get("/examples")
def list_examples():
    return jsonify([e.to_dict() for e in get_repo().list_examples()])
