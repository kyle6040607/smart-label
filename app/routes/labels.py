"""標記 / 種子範例 API（提案 demo 第 2 步：點選打字標幾個範例）。"""
from __future__ import annotations

from flask import Blueprint, abort, jsonify, request

from app.routes import get_pipeline, get_repo
from app.routes.auth import api_login_required, get_current_user, is_admin, owns, scope_owner_id

bp = Blueprint("labels", __name__, url_prefix="/api")
bp.before_request(api_login_required)


@bp.get("/labels")
def list_labels():
    owner_id = scope_owner_id(get_current_user())
    return jsonify(get_repo().labels(owner_id=owner_id))


@bp.post("/segments/<seg_id>/label")
def label_segment(seg_id: str):
    """把某片段標成種子範例，觸發主動學習回訓。body: {"label": str}"""
    repo, pipeline = get_repo(), get_pipeline()
    seg = repo.get_segment(seg_id)
    if not seg or not owns(get_current_user(), seg.owner_id):
        abort(404)
    data = request.get_json(force=True)
    label = (data.get("label") or "").strip()
    if not label:
        abort(400, "label 不可為空")
    ex = pipeline.add_example_from_segment(seg, label)
    return jsonify({"example": ex.to_dict(), "segment": seg.to_dict()}), 201


@bp.delete("/labels/<path:label>")
def delete_label(label: str):
    """刪掉自己建錯的類別，只影響刪除者自己的分類器。

    admin 可用 ?owner_id=<user_id> 指定要刪哪個使用者的類別；不帶則刪自己的。
    """
    user = get_current_user()
    target_owner_id = user.id
    if is_admin(user) and request.args.get("owner_id"):
        target_owner_id = request.args["owner_id"]
    n = get_pipeline().delete_label(label, target_owner_id)
    if n == 0:
        abort(404, "查無此類別")
    return jsonify({"deleted": label, "examples_removed": n})


@bp.get("/examples")
def list_examples():
    """只回傳類別與時間，不外洩 feature 向量或 source_segment_id（可能指向別人看不到的片段）。"""
    owner_id = scope_owner_id(get_current_user())
    return jsonify([
        {"id": e.id, "label": e.label, "created_at": e.created_at}
        for e in get_repo().list_examples(owner_id=owner_id)
    ])
