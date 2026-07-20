"""分割 API（提案 demo 第 3 步：系統自動分割，低信心標紅）。"""
from __future__ import annotations

import json
import queue
import threading
from pathlib import Path

from flask import Blueprint, abort, jsonify, request, send_file, Response

from app.routes import get_pipeline, get_repo
from app.routes.auth import api_login_required, get_current_user, owns

bp = Blueprint("segment", __name__, url_prefix="/api")
bp.before_request(api_login_required)


@bp.post("/images/<image_id>/segment")
def segment_image(image_id: str):
    """對整張圖自動切割並分類，回傳所有片段（含信心、是否送審）與完成狀態。"""
    repo, pipeline = get_repo(), get_pipeline()
    img = repo.get_image(image_id)
    if not img or not owns(get_current_user(), img.owner_id):
        abort(404)

    # 紀錄執行前的片段數量
    existing_count = len(repo.list_segments(image_id))

    q = queue.Queue()

    def run_segmentation():
        try:
            def progress_callback(data):
                q.put(data)
            
            segments = pipeline.segment_image(img, progress_callback=progress_callback)
            q.put({"event": "done", "segments": segments})
        except Exception as e:
            q.put({"event": "error", "message": str(e)})

    t = threading.Thread(target=run_segmentation)
    t.start()

    def generate():
        while True:
            try:
                data = q.get(timeout=0.5)
                if data["event"] == "done":
                    # 紀錄執行後的片段數量
                    new_count = len(repo.list_segments(image_id))
                    # 若數量沒有增加，代表產出的所有片段本來就已經存在（無缺失且已自動分割過）
                    status = "already_completed" if new_count == existing_count else "added"
                    
                    yield json.dumps({
                        "event": "done",
                        "status": status,
                        "segments": [s.to_dict() for s in data["segments"]]
                    }) + "\n"
                    break
                elif data["event"] == "error":
                    yield json.dumps({"event": "error", "message": data["message"]}) + "\n"
                    break
                else:
                    yield json.dumps(data) + "\n"
            except queue.Empty:
                if not t.is_alive():
                    yield json.dumps({"event": "error", "message": "Segmentation thread terminated unexpectedly."}) + "\n"
                    break

    return Response(generate(), status=201, mimetype="application/x-ndjson")


@bp.post("/images/<image_id>/segment_point")
def segment_point(image_id: str):
    """互動式：在使用者點擊座標切出單一物件。body: {"x": int, "y": int}"""
    repo, pipeline = get_repo(), get_pipeline()
    img = repo.get_image(image_id)
    if not img or not owns(get_current_user(), img.owner_id):
        abort(404)
    data = request.get_json(force=True)
    seg = pipeline.segment_point(img, (int(data["x"]), int(data["y"])))
    return jsonify(seg.to_dict()), 201


@bp.post("/images/<image_id>/segment_text")
def segment_text(image_id: str):
    """自然語言分割。body: {"prompt": "cat"}，回傳零到多個 Segment。"""
    repo, pipeline = get_repo(), get_pipeline()
    img = repo.get_image(image_id)
    if not img or not owns(get_current_user(), img.owner_id):
        abort(404, "找不到圖片")

    data = request.get_json(silent=True) or {}
    prompt = str(data.get("prompt", "")).strip()

    q = queue.Queue()

    def run_segmentation():
        try:
            def progress_callback(data):
                q.put(data)
            
            segments = pipeline.segment_text(img, prompt, progress_callback=progress_callback)
            q.put({"event": "done", "segments": segments})
        except Exception as e:
            q.put({"event": "error", "message": str(e)})

    t = threading.Thread(target=run_segmentation)
    t.start()

    def generate():
        while True:
            try:
                data = q.get(timeout=0.5)
                if data["event"] == "done":
                    yield json.dumps({
                        "event": "done",
                        "segments": [s.to_dict() for s in data["segments"]]
                    }) + "\n"
                    break
                elif data["event"] == "error":
                    yield json.dumps({"event": "error", "message": data["message"]}) + "\n"
                    break
                else:
                    yield json.dumps(data) + "\n"
            except queue.Empty:
                if not t.is_alive():
                    yield json.dumps({"event": "error", "message": "Text segmentation thread terminated unexpectedly."}) + "\n"
                    break

    return Response(generate(), status=201, mimetype="application/x-ndjson")


@bp.post("/images/<image_id>/segment_polygon")
def segment_polygon(image_id: str):
    """手動描邊：使用者沿邊界畫出多邊形。body: {"points": [[x,y], ...]}"""
    repo, pipeline = get_repo(), get_pipeline()
    img = repo.get_image(image_id)
    if not img or not owns(get_current_user(), img.owner_id):
        abort(404)
    data = request.get_json(force=True)
    points = [(int(x), int(y)) for x, y in data.get("points", [])]
    if len(points) < 3:
        abort(400, "至少需要 3 個點才能圍成區域")
    seg = pipeline.segment_polygon(img, points)
    return jsonify(seg.to_dict()), 201


@bp.get("/images/<image_id>/segments")
def list_segments(image_id: str):
    repo = get_repo()
    img = repo.get_image(image_id)
    if not img or not owns(get_current_user(), img.owner_id):
        abort(404)
    return jsonify([s.to_dict() for s in repo.list_segments(image_id)])


@bp.delete("/segments/<seg_id>")
def delete_segment(seg_id: str):
    """刪掉切壞/不要的片段，連同它的遮罩 PNG。"""
    repo = get_repo()
    seg = repo.get_segment(seg_id)
    if not seg or not owns(get_current_user(), seg.owner_id):
        abort(404)
    mask = repo.delete_segment(seg_id)
    if mask:
        Path(mask).unlink(missing_ok=True)
    return jsonify({"deleted": seg_id})


@bp.post("/segments/delete_batch")
def delete_segments_batch():
    """批次刪除選定的遮罩片段，連同其遮罩檔案。只會刪自己擁有的（admin 不受限）。"""
    repo = get_repo()
    user = get_current_user()
    data = request.get_json(silent=True) or {}
    seg_ids = data.get("segment_ids", [])
    if not seg_ids:
        abort(400, "無效的片段 ID 清單")

    deleted_ids = []
    for seg_id in seg_ids:
        seg = repo.get_segment(seg_id)
        if seg and owns(user, seg.owner_id):
            mask = repo.delete_segment(seg_id)
            if mask:
                Path(mask).unlink(missing_ok=True)
            deleted_ids.append(seg_id)

    return jsonify({"deleted_ids": deleted_ids}), 200


@bp.get("/segments/<seg_id>/mask")
def segment_mask(seg_id: str):
    """回傳遮罩 PNG，給前端疊圖。"""
    seg = get_repo().get_segment(seg_id)
    if not seg or not seg.mask_path or not owns(get_current_user(), seg.owner_id):
        abort(404)
    return send_file(Path(seg.mask_path), mimetype="image/png")
