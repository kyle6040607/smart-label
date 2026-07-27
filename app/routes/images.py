"""影像上傳與瀏覽 API（提案 demo 第 1 步：上傳一批照片）。"""
from __future__ import annotations

import io
import mimetypes

from flask import Blueprint, abort, jsonify, request, send_file
from PIL import Image, ImageOps
from werkzeug.utils import secure_filename

from app.routes import get_config, get_repo, get_storage
from app.models import ImageRecord

bp = Blueprint("images", __name__, url_prefix="/api/images")


def _allowed(filename: str, allowed: tuple[str, ...]) -> bool:
    return "." in filename and filename.rsplit(".", 1)[1].lower() in allowed


@bp.post("")
def upload():
    """支援一次上傳多張。回傳建立的 ImageRecord 清單。"""
    import hashlib
    cfg, repo, storage = get_config(), get_repo(), get_storage()
    files = request.files.getlist("files") or request.files.getlist("file")
    if not files:
        abort(400, "沒有收到檔案（欄位名 files）")

    created = []
    duplicates = []
    repo_updated = False

    for f in files:
        if not f.filename or not _allowed(f.filename, cfg.allowed_ext):
            continue

        # 計算檔案的 SHA-256 雜湊值
        file_bytes = f.read()
        f.seek(0)
        file_hash = hashlib.sha256(file_bytes).hexdigest()

        # 比對是否已有相同雜湊值的照片已上傳
        is_duplicate = False
        for img in repo.list_images():
            existing_hash = getattr(img, "file_hash", "")
            # 針對歷史舊資料進行相容性雜湊值計算與補齊
            if not existing_hash and img.path:
                try:
                    existing_hash = hashlib.sha256(
                        storage.read_bytes(img.path)
                    ).hexdigest()
                    img.file_hash = existing_hash
                    repo_updated = True
                except Exception:
                    pass
            if existing_hash == file_hash:
                is_duplicate = True
                duplicates.append(f.filename)
                break

        if is_duplicate:
            continue

        extension = f.filename.rsplit(".", 1)[1].lower()
        name = secure_filename(f.filename)
        rec = ImageRecord(
            filename=name,
            file_hash=file_hash,
        )

        if not name:
            name = f"{rec.id}.{extension}"
            rec.filename = name

        # 在記憶體中校正圖片方向及縮圖，再交給目前選用的儲存後端。
        with Image.open(io.BytesIO(file_bytes)) as im:
            save_format = im.format or extension.upper()
            if save_format.upper() == "JPG":
                save_format = "JPEG"
            processed_image = ImageOps.exif_transpose(im)

            max_side = 1024

            if max(processed_image.size) > max_side:
                scale = max_side / max(processed_image.size)
                new_size = (
                    int(processed_image.size[0] * scale),
                    int(processed_image.size[1] * scale),
                )
                processed_image = processed_image.resize(
                    new_size,
                    Image.Resampling.LANCZOS,
                )

            rec.width, rec.height = processed_image.size

            output = io.BytesIO()
            processed_image.save(
                output,
                format=save_format,
            )

        content_type = (
            Image.MIME.get(save_format.upper())
            or f.mimetype
            or "application/octet-stream"
        )
        rec.path = storage.save_bytes(
            f"images/{rec.id}_{name}",
            output.getvalue(),
            content_type,
        )
        repo.add_image(rec)
        created.append(rec.to_dict())

    if repo_updated:
        repo._save()

    if duplicates and not created:
        return jsonify({"error": "圖片已上傳過，請勿重複上傳"}), 400

    if not created:
        abort(400, "沒有有效的影像檔")
    return jsonify(created), 201


@bp.get("")
def list_images():
    return jsonify([i.to_dict() for i in get_repo().list_images()])


@bp.get("/<image_id>/file")
def image_file(image_id: str):
    """回傳原圖，給前端 canvas 顯示。"""
    rec = get_repo().get_image(image_id)
    if not rec:
        abort(404)
    try:
        data = get_storage().read_bytes(rec.path)
    except FileNotFoundError:
        abort(404)
    return send_file(
        io.BytesIO(data),
        mimetype=mimetypes.guess_type(rec.filename)[0],
        download_name=rec.filename,
    )


@bp.delete("/<image_id>")
def delete_image(image_id: str):
    """刪除一張上傳的照片，連同它的遮罩片段與檔案一起清掉。"""
    repo = get_repo()
    if not repo.get_image(image_id):
        abort(404)
    paths = repo.delete_image(image_id)
    files_removed = sum(
        get_storage().delete(path)
        for path in paths
        if path
    )
    return jsonify({
        "deleted": image_id,
        "files_removed": files_removed,
    })


@bp.post("/delete_batch")
def delete_images_batch():
    """批次刪除照片，連同其遮罩與檔案。"""
    repo = get_repo()
    data = request.get_json(silent=True) or {}
    image_ids = data.get("image_ids", [])
    if not image_ids:
        abort(400, "無效的圖片 ID 清單")

    paths = repo.delete_images_batch(image_ids)
    files_removed = sum(
        get_storage().delete(path)
        for path in paths
        if path
    )

    return jsonify({
        "deleted_ids": image_ids,
        "files_removed": files_removed,
    }), 200
