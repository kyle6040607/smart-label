"""審核佇列與統計 API（提案 demo 第 4–6 步：修正紅色區塊、準確率曲線、省下工時）。"""
from __future__ import annotations

from flask import Blueprint, abort, jsonify, request

from app.routes import get_pipeline, get_repo
from app.routes.auth import api_login_required, get_current_user, owns, scope_owner_id

bp = Blueprint("review", __name__, url_prefix="/api")
bp.before_request(api_login_required)


@bp.get("/review/queue")
def review_queue():
    """待人工審核的低信心片段（被標紅的）。admin 看全部，一般使用者只看自己的。"""
    owner_id = scope_owner_id(get_current_user())
    return jsonify([s.to_dict() for s in get_repo().list_review_queue(owner_id=owner_id)])


@bp.post("/segments/<seg_id>/review")
def review_segment(seg_id: str):
    """人工修正某片段的類別。body: {"label": str}

    修正後同時當作新種子範例餵回去，回訓讓模型越標越準（主動學習迴圈）。
    """
    repo, pipeline = get_repo(), get_pipeline()
    seg = repo.get_segment(seg_id)
    if not seg or not owns(get_current_user(), seg.owner_id):
        abort(404)
    data = request.get_json(force=True)
    label = (data.get("label") or "").strip()
    if not label:
        abort(400, "label 不可為空")
    ex = pipeline.add_example_from_segment(seg, label)
    return jsonify({"example": ex.to_dict(), "segment": seg.to_dict()})


@bp.get("/stats")
def stats():
    """統計：自動接受比例 ≈ 省下的工時、送審數量等。admin 看全體，一般使用者看自己的。"""
    owner_id = scope_owner_id(get_current_user())
    return jsonify(get_repo().stats(owner_id=owner_id))
