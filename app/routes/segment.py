"""分割 API（提案 demo 第 3 步：系統自動分割，低信心標紅）。"""
from __future__ import annotations

from pathlib import Path

from flask import Blueprint, abort, jsonify, request, send_file

from app.routes import get_pipeline, get_repo

bp = Blueprint("segment", __name__, url_prefix="/api")


@bp.post("/images/<image_id>/segment")
def segment_image(image_id: str):
    """對整張圖自動切割並分類，回傳所有片段（含信心、是否送審）與完成狀態。"""
    repo, pipeline = get_repo(), get_pipeline()
    img = repo.get_image(image_id)
    if not img:
        abort(404)

    # 紀錄執行前的片段數量
    existing_count = len(repo.list_segments(image_id))

    segments = pipeline.segment_image(img)

    # 紀錄執行後的片段數量
    new_count = len(repo.list_segments(image_id))

    # 若數量沒有增加，代表產出的所有片段本來就已經存在（無缺失且已自動分割過）
    status = "already_completed" if new_count == existing_count else "added"

    return jsonify({
        "status": status,
        "segments": [s.to_dict() for s in segments]
    }), 201


@bp.post("/images/<image_id>/segment_point")
def segment_point(image_id: str):
    """互動式：在使用者點擊座標切出單一物件。body: {"x": int, "y": int}"""
    repo, pipeline = get_repo(), get_pipeline()
    img = repo.get_image(image_id)
    if not img:
        abort(404)
    data = request.get_json(force=True)
    seg = pipeline.segment_point(img, (int(data["x"]), int(data["y"])))
    return jsonify(seg.to_dict()), 201


@bp.post("/images/<image_id>/segment_text")
def segment_text(image_id: str):
    """自然語言分割。body: {"prompt": "cat"}，回傳零到多個 Segment。"""
    repo, pipeline = get_repo(), get_pipeline()
    img = repo.get_image(image_id)
    if not img:
        abort(404, "找不到圖片")

    data = request.get_json(silent=True) or {}
    prompt = str(data.get("prompt", "")).strip()
    try:
        segments = pipeline.segment_text(img, prompt)
    except ValueError as exc:
        abort(400, str(exc))
    except NotImplementedError as exc:
        abort(503, str(exc))

    return jsonify([seg.to_dict() for seg in segments]), 201


@bp.post("/images/<image_id>/segment_polygon")
def segment_polygon(image_id: str):
    """手動描邊：使用者沿邊界畫出多邊形。body: {"points": [[x,y], ...]}"""
    repo, pipeline = get_repo(), get_pipeline()
    img = repo.get_image(image_id)
    if not img:
        abort(404)
    data = request.get_json(force=True)
    points = [(int(x), int(y)) for x, y in data.get("points", [])]
    if len(points) < 3:
        abort(400, "至少需要 3 個點才能圍成區域")
    seg = pipeline.segment_polygon(img, points)
    return jsonify(seg.to_dict()), 201


@bp.get("/images/<image_id>/segments")
def list_segments(image_id: str):
    return jsonify([s.to_dict() for s in get_repo().list_segments(image_id)])


@bp.delete("/segments/<seg_id>")
def delete_segment(seg_id: str):
    """刪掉切壞/不要的片段，連同它的遮罩 PNG。"""
    repo = get_repo()
    if not repo.get_segment(seg_id):
        abort(404)
    mask = repo.delete_segment(seg_id)
    if mask:
        Path(mask).unlink(missing_ok=True)
    return jsonify({"deleted": seg_id})


@bp.get("/segments/<seg_id>/mask")
def segment_mask(seg_id: str):
    """回傳遮罩 PNG，給前端疊圖。"""
    seg = get_repo().get_segment(seg_id)
    if not seg or not seg.mask_path:
        abort(404)
    return send_file(Path(seg.mask_path), mimetype="image/png")
