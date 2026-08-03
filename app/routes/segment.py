"""分割 API（提案 demo 第 3 步：系統自動分割，低信心標紅）。"""
from __future__ import annotations

import io
import json
import queue
import threading

from flask import Blueprint, abort, jsonify, request, send_file, Response

from app.routes import (
    can_access_image,
    get_owned_image,
    get_owned_segment,
    get_pipeline,
    get_repo,
    get_storage,
)
from app.services.pipeline import InferenceBusyError

bp = Blueprint("segment", __name__, url_prefix="/api")

# YOLO-World 推論解析度的可接受範圍（Ultralytics 要求為 32 的倍數）
YOLO_IMGSZ_MIN = 320
YOLO_IMGSZ_MAX = 1536


@bp.post("/images/<image_id>/segment")
def segment_image(image_id: str):
    """對整張圖自動切割並分類，回傳所有片段（含信心、是否送審）與完成狀態。"""
    repo, pipeline = get_repo(), get_pipeline()
    img = get_owned_image(image_id)

    # 紀錄執行前的片段數量
    existing_count = len(repo.list_segments(image_id))

    q = queue.Queue()

    def run_segmentation():
        try:
            def progress_callback(data):
                q.put(data)
            
            segments = pipeline.segment_image(img, progress_callback=progress_callback)
            q.put({"event": "done", "segments": segments})
        except InferenceBusyError as e:
            q.put({
                "event": "error",
                "code": "model_busy",
                "message": str(e),
            })
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
                    yield json.dumps({
                        "event": "error",
                        "code": data.get("code"),
                        "message": data["message"],
                    }) + "\n"
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
    img = get_owned_image(image_id)
    data = request.get_json(force=True)
    seg = pipeline.segment_point(img, (int(data["x"]), int(data["y"])))
    return jsonify(seg.to_dict()), 201


@bp.post("/images/<image_id>/segment_text")
def segment_text(image_id: str):
    """自然語言分割。body: {"prompt": "cat"}，回傳零到多個 Segment。"""
    repo, pipeline = get_repo(), get_pipeline()
    img = get_owned_image(image_id)

    data = request.get_json(silent=True) or {}
    prompt = str(data.get("prompt", "")).strip()

    q = queue.Queue()

    def run_segmentation():
        try:
            def progress_callback(data):
                q.put(data)
            
            segments = pipeline.segment_text(img, prompt, progress_callback=progress_callback)
            q.put({"event": "done", "segments": segments})
        except InferenceBusyError as e:
            q.put({
                "event": "error",
                "code": "model_busy",
                "message": str(e),
            })
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
                    yield json.dumps({
                        "event": "error",
                        "code": data.get("code"),
                        "message": data["message"],
                    }) + "\n"
                    break
                else:
                    yield json.dumps(data) + "\n"
            except queue.Empty:
                if not t.is_alive():
                    yield json.dumps({"event": "error", "message": "Text segmentation thread terminated unexpectedly."}) + "\n"
                    break

    return Response(generate(), status=201, mimetype="application/x-ndjson")


@bp.post("/images/batch_segment_text")
def batch_segment_text():
    """批次自然語言分割：多張圖片使用同一個 prompt。
    body: {"image_ids": ["id1", "id2"], "prompt": "cat"} 或空 image_ids 表示目前所有擁有的圖片。
    """
    repo, pipeline = get_repo(), get_pipeline()
    data = request.get_json(silent=True) or {}
    image_ids = data.get("image_ids", [])
    prompt = str(data.get("prompt", "")).strip()

    if not prompt:
        return jsonify({"error": "請提供 prompt"}), 400

    # 🔒 嚴格存取控制 (Access Control Check)：只處理當前使用者有權限存取的圖片
    if not image_ids:
        owned_images = [img for img in repo.list_images() if can_access_image(img)]
    else:
        owned_images = []
        for img_id in image_ids:
            img = repo.get_image(img_id)
            if img and can_access_image(img):
                owned_images.append(img)

    if not owned_images:
        return jsonify({"error": "目前沒有您有權限存取且可處理的圖片"}), 403

    print(f"⚡ [Batch YOLO-World] 開始對共 {len(owned_images)} 張圖片執行 Prompt '{prompt}' 批次標註...")

    # 💡 [效能極致最佳化] 在進入 101 張圖片迴圈前，Gemini LLM 只需提前解析 1 次！
    parsed_classes: list[str] = [prompt]
    words = prompt.split()
    has_chinese = any("\u4e00" <= c <= "\u9fff" for c in prompt)
    if has_chinese or len(words) > 3:
        print(f"🧠 [Batch Gemini] 提前解析 Prompt '{prompt}'...", flush=True)
        try:
            parsed_classes = pipeline.gemini_service.parse_prompt(prompt)
            print(f"✅ [Batch Gemini] 提前解析成功，提取物件類別: {parsed_classes}", flush=True)
        except Exception as e:
            print(f"⚠️ [Batch Gemini Warning] 解析失敗，降級使用原文字: {e}", flush=True)
            parsed_classes = [prompt]

    q = queue.Queue()

    def run_batch():
        total = len(owned_images)
        print(f"🚀 [Batch Start] 開始處理共 {total} 張圖片的 Prompt '{prompt}' (類別: {parsed_classes}) 標註...", flush=True)
        success_cnt = 0
        for idx, img in enumerate(owned_images, start=1):
            q.put({
                "event": "progress",
                "current": idx,
                "total": total,
                "image_id": img.id,
                "filename": img.filename,
            })
            print(f"📸 [{idx}/{total}] 正在對圖片進行分析: {img.filename} (ID: {img.id})...", flush=True)
            try:
                segments = pipeline.segment_text(img, prompt, parsed_classes=parsed_classes)
                success_cnt += 1
                print(f"✅ [{idx}/{total}] 圖片 {img.filename} 完成標註! 產生 {len(segments)} 個物件。", flush=True)
                q.put({
                    "event": "image_done",
                    "image_id": img.id,
                    "filename": img.filename,
                    "segments": [s.to_dict() for s in segments],
                })
            except Exception as e:
                print(f"❌ [{idx}/{total}] 圖片 {img.filename} 標註失敗: {e}", flush=True)
                q.put({
                    "event": "image_error",
                    "image_id": img.id,
                    "filename": img.filename,
                    "message": str(e),
                })
        print(f"🎉 [Batch Finished] 批次完成! 成功標註 {success_cnt}/{total} 張圖片。", flush=True)
        q.put({"event": "done", "total_images": total, "success_images": success_cnt})

    t = threading.Thread(target=run_batch)
    t.start()

    def generate():
        while True:
            try:
                msg = q.get(timeout=0.5)
                yield json.dumps(msg) + "\n"
                if msg["event"] in ("done", "error"):
                    break
            except queue.Empty:
                if not t.is_alive():
                    yield json.dumps({"event": "error", "message": "批次標註執行緒異常中斷"}) + "\n"
                    break

    return Response(generate(), status=200, mimetype="application/x-ndjson")


@bp.post("/images/<image_id>/segment_polygon")
def segment_polygon(image_id: str):
    """手動描邊：使用者沿邊界畫出多邊形。body: {"points": [[x,y], ...]}"""
    repo, pipeline = get_repo(), get_pipeline()
    img = get_owned_image(image_id)
    data = request.get_json(force=True)
    points = [(int(x), int(y)) for x, y in data.get("points", [])]
    if len(points) < 3:
        abort(400, "至少需要 3 個點才能圍成區域")
    seg = pipeline.segment_polygon(img, points)
    return jsonify(seg.to_dict()), 201


@bp.get("/images/<image_id>/segments")
def list_segments(image_id: str):
    get_owned_image(image_id)
    return jsonify([s.to_dict() for s in get_repo().list_segments(image_id)])


@bp.delete("/segments/<seg_id>")
def delete_segment(seg_id: str):
    """刪掉切壞/不要的片段，連同它的遮罩 PNG。"""
    repo, pipeline = get_repo(), get_pipeline()
    seg = get_owned_segment(seg_id)
    image = repo.get_image(seg.image_id) if seg else None
    mask = repo.delete_segment(seg_id)
    if mask:
        get_storage().delete(mask)

    if image:
        pipeline.refit(image.owner_id, image.project_id)
        pipeline.reclassify_pending(image.owner_id, image.project_id)

    return jsonify({"deleted": seg_id})


@bp.post("/segments/delete_batch")
def delete_segments_batch():
    """批次刪除選定的遮罩片段，連同其遮罩檔案。"""
    repo, pipeline = get_repo(), get_pipeline()
    data = request.get_json(silent=True) or {}
    seg_ids = data.get("segment_ids", [])
    if not seg_ids:
        abort(400, "無效的片段 ID 清單")

    first_image = None
    for seg_id in seg_ids:
        seg = get_owned_segment(str(seg_id))
        if first_image is None and seg:
            first_image = repo.get_image(seg.image_id)

    paths = repo.delete_segments_batch(seg_ids)
    for p in paths:
        get_storage().delete(p)

    if first_image:
        pipeline.refit(first_image.owner_id, first_image.project_id)
        pipeline.reclassify_pending(first_image.owner_id, first_image.project_id)

    return jsonify({"deleted_ids": seg_ids}), 200


@bp.get("/segments/<seg_id>/mask")
def segment_mask(seg_id: str):
    """回傳遮罩 PNG，給前端疊圖。"""
    seg = get_owned_segment(seg_id)
    if not seg or not seg.mask_path:
        abort(404)
    try:
        data = get_storage().read_bytes(seg.mask_path)
    except FileNotFoundError:
        abort(404)
    return send_file(
        io.BytesIO(data),
        mimetype="image/png",
        download_name=f"{seg.id}.png",
    )


@bp.get("/parameters")
def get_parameters():
    """取得當前動態參數值。"""
    pipeline = get_pipeline()
    return jsonify({
        "confidence_threshold": pipeline.config.confidence_threshold,
        "yolo_world_confidence": pipeline.config.yolo_world_confidence,
        "yolo_imgsz": pipeline.config.yolo_imgsz
    })


@bp.post("/parameters")
def update_parameters():
    """更新動態參數值，並重算未審核片段之信心送審狀態。"""
    repo, pipeline = get_repo(), get_pipeline()
    data = request.get_json(force=True, silent=True)
    if not isinstance(data, dict):
        return jsonify({"error": "無效的 JSON 請求"}), 400

    if "confidence_threshold" in data:
        raw_val = data["confidence_threshold"]
        if isinstance(raw_val, bool):
            return jsonify({"error": "confidence_threshold 必須為有效的數字"}), 400
        try:
            val = float(raw_val)
            if not (0.0 <= val <= 1.0):
                return jsonify({"error": "confidence_threshold 必須為 0.0 至 1.0 之間的數字"}), 400
        except (ValueError, TypeError):
            return jsonify({"error": "confidence_threshold 必須為有效的數字"}), 400
        pipeline.config.confidence_threshold = val
        repo.set_parameter("confidence_threshold", val)

    if "yolo_world_confidence" in data:
        raw_val = data["yolo_world_confidence"]
        if isinstance(raw_val, bool):
            return jsonify({"error": "yolo_world_confidence 必須為有效的數字"}), 400
        try:
            val = float(raw_val)
            if not (0.0 <= val <= 1.0):
                return jsonify({"error": "yolo_world_confidence 必須為 0.0 至 1.0 之間的數字"}), 400
        except (ValueError, TypeError):
            return jsonify({"error": "yolo_world_confidence 必須為有效的數字"}), 400
        pipeline.config.yolo_world_confidence = val
        repo.set_parameter("yolo_world_confidence", val)

    if "yolo_imgsz" in data:
        raw_val = data["yolo_imgsz"]
        if isinstance(raw_val, bool):
            return jsonify({"error": "yolo_imgsz 必須為有效的整數"}), 400
        try:
            num_val = float(raw_val)
        except (ValueError, TypeError):
            return jsonify({"error": "yolo_imgsz 必須為有效的整數"}), 400
        if num_val != int(num_val):
            return jsonify({"error": "yolo_imgsz 必須為有效的整數"}), 400
        val = int(num_val)
        # Ultralytics 會把非 32 倍數的值靜默調整，先擋掉避免實際生效值與顯示值不符
        if not (YOLO_IMGSZ_MIN <= val <= YOLO_IMGSZ_MAX) or val % 32 != 0:
            return jsonify({
                "error": (
                    f"yolo_imgsz 必須為 {YOLO_IMGSZ_MIN} 至 {YOLO_IMGSZ_MAX} 之間、"
                    "且為 32 的倍數"
                )
            }), 400
        pipeline.config.yolo_imgsz = val
        repo.set_parameter("yolo_imgsz", val)

    pipeline.reclassify_pending()

    return jsonify({
        "status": "success",
        "parameters": {
            "confidence_threshold": pipeline.config.confidence_threshold,
            "yolo_world_confidence": pipeline.config.yolo_world_confidence,
            "yolo_imgsz": pipeline.config.yolo_imgsz
        }
    })
