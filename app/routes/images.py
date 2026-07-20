"""影像上傳與瀏覽 API（提案 demo 第 1 步：上傳一批照片）。"""
from __future__ import annotations

from pathlib import Path

from flask import Blueprint, abort, jsonify, request, send_file, session
from PIL import Image
from werkzeug.utils import secure_filename

from app.routes import get_config, get_repo
from app.routes.auth import api_login_required, get_current_user, owns, scope_owner_id
from app.models import ImageRecord

bp = Blueprint("images", __name__, url_prefix="/api/images")
bp.before_request(api_login_required)


def _allowed(filename: str, allowed: tuple[str, ...]) -> bool:
    return "." in filename and filename.rsplit(".", 1)[1].lower() in allowed


@bp.post("")
def upload():
    """支援一次上傳多張。回傳建立的 ImageRecord 清單。"""
    import hashlib
    cfg, repo = get_config(), get_repo()
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

        # 比對是否已有相同雜湊值的照片已上傳（只比對自己上傳過的，不比對別人的）
        is_duplicate = False
        for img in repo.list_images(owner_id=session["user_id"]):
            existing_hash = getattr(img, "file_hash", "")
            # 針對歷史舊資料進行相容性雜湊值計算與補齊
            if not existing_hash and img.path and Path(img.path).exists():
                try:
                    with open(img.path, "rb") as ef:
                        existing_hash = hashlib.sha256(ef.read()).hexdigest()
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

        name = secure_filename(f.filename)
        rec = ImageRecord(filename=name, file_hash=file_hash, owner_id=session["user_id"])
        dest = cfg.upload_dir / f"{rec.id}_{name}"
        f.save(dest)

        # 讀取並等比例縮小原圖（避免大圖撐開版面且加速 AI 運算）
        with Image.open(dest) as im:
            # 自動校正 EXIF 旋轉方向
            from PIL import ImageOps
            im_corrected = ImageOps.exif_transpose(im)
            
            max_side = 1024
            if max(im_corrected.size) > max_side:
                scale = max_side / max(im_corrected.size)
                new_size = (int(im_corrected.size[0] * scale), int(im_corrected.size[1] * scale))
                im_resized = im_corrected.resize(new_size, Image.Resampling.LANCZOS)
                
                # 儲存覆蓋原檔，保持原格式（或預設為 JPEG）
                save_format = im.format or "JPEG"
                im_resized.save(dest, format=save_format)
                rec.width, rec.height = new_size
            else:
                # 若不需縮小但有旋轉校正，重新存檔
                if im_corrected.size != im.size:
                    save_format = im.format or "JPEG"
                    im_corrected.save(dest, format=save_format)
                rec.width, rec.height = im_corrected.size

        rec.path = str(dest)
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
    owner_id = scope_owner_id(get_current_user())
    return jsonify([i.to_dict() for i in get_repo().list_images(owner_id=owner_id)])


@bp.get("/<image_id>/file")
def image_file(image_id: str):
    """回傳原圖，給前端 canvas 顯示。"""
    rec = get_repo().get_image(image_id)
    if not rec or not owns(get_current_user(), rec.owner_id):
        abort(404)
    return send_file(Path(rec.path))


@bp.delete("/<image_id>")
def delete_image(image_id: str):
    """刪除一張上傳的照片，連同它的遮罩片段與檔案一起清掉。"""
    repo = get_repo()
    rec = repo.get_image(image_id)
    if not rec or not owns(get_current_user(), rec.owner_id):
        abort(404)
    paths = repo.delete_image(image_id)          # 先從資料層移除
    for p in paths:                              # 再刪實體檔（原圖 + 遮罩 PNG）
        Path(p).unlink(missing_ok=True)
    return jsonify({"deleted": image_id, "files_removed": len(paths)})


@bp.post("/delete_batch")
def delete_images_batch():
    """批次刪除照片，連同其遮罩與檔案。只會刪自己擁有的（admin 不受限）。"""
    repo = get_repo()
    user = get_current_user()
    data = request.get_json(silent=True) or {}
    image_ids = data.get("image_ids", [])
    if not image_ids:
        abort(400, "無效的圖片 ID 清單")

    removed_files_count = 0
    deleted_ids = []

    for image_id in image_ids:
        rec = repo.get_image(image_id)
        if rec and owns(user, rec.owner_id):
            paths = repo.delete_image(image_id)
            for p in paths:
                Path(p).unlink(missing_ok=True)
            removed_files_count += len(paths)
            deleted_ids.append(image_id)

    return jsonify({"deleted_ids": deleted_ids, "files_removed": removed_files_count}), 200
