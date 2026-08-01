"""審核佇列與統計 API（提案 demo 第 4–6 步：修正紅色區塊、準確率曲線、省下工時）。"""
from __future__ import annotations

from flask import Blueprint, abort, jsonify, request

from app.routes import (
    can_access_image,
    get_current_project_id,
    get_current_user_id,
    get_owned_segment,
    get_pipeline,
    get_repo,
)

bp = Blueprint("review", __name__, url_prefix="/api")


@bp.get("/review/queue")
def review_queue():
    """待人工審核的低信心片段（被標紅的）。"""
    repo = get_repo()
    project_id = request.args.get("project_id") or get_current_project_id()
    owned_image_ids = {
        image.id for image in repo.list_images(project_id=project_id) if can_access_image(image)
    }
    return jsonify([
        segment.to_dict()
        for segment in repo.list_review_queue()
        if segment.image_id in owned_image_ids
    ])


@bp.post("/segments/<seg_id>/review")
def review_segment(seg_id: str):
    """人工修正某片段的類別。body: {"label": str}

    修正後同時當作新種子範例餵回去，回訓讓模型越標越準（主動學習迴圈）。
    """
    repo, pipeline = get_repo(), get_pipeline()
    seg = get_owned_segment(seg_id)
    data = request.get_json(force=True)
    label = (data.get("label") or "").strip()
    if not label:
        abort(400, "label 不可為空")
    ex = pipeline.add_example_from_segment(seg, label)
    return jsonify({"example": ex.to_dict(), "segment": seg.to_dict()})


@bp.get("/stats")
def stats():
    """整體統計：自動接受比例 ≈ 省下的工時、送審數量等。"""
    repo = get_repo()
    project_id = request.args.get("project_id") or get_current_project_id()
    owned_image_ids = {
        image.id for image in repo.list_images(project_id=project_id) if can_access_image(image)
    }
    segments = [
        segment for segment in repo.list_segments()
        if segment.image_id in owned_image_ids
    ]
    total = len(segments)
    need_review = sum(1 for segment in segments if segment.needs_review)
    reviewed = sum(1 for segment in segments if segment.reviewed)
    auto_accepted = total - need_review
    labels = {
        segment.final_label for segment in segments if segment.final_label
    }
    label_counts: dict[str, int] = {}
    for segment in segments:
        if segment.final_label:
            label_counts[segment.final_label] = (
                label_counts.get(segment.final_label, 0) + 1
            )
    examples = repo.list_examples(get_current_user_id(), project_id=project_id)
    return jsonify({
        "total_segments": total,
        "auto_accepted": auto_accepted,
        "need_review": need_review,
        "reviewed": reviewed,
        "auto_ratio": round(auto_accepted / total, 3) if total else 0.0,
        "num_examples": len(examples),
        "num_labels": len(labels),
        "label_counts": label_counts,
    })
