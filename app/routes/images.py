"""影像上傳與瀏覽 API（提案 demo 第 1 步：上傳一批照片）。"""
from __future__ import annotations

from pathlib import Path

from flask import Blueprint, abort, jsonify, request, send_file
from PIL import Image
from werkzeug.utils import secure_filename

from app.routes import get_config, get_repo
from app.models import ImageRecord

bp = Blueprint("images", __name__, url_prefix="/api/images")


def _allowed(filename: str, allowed: tuple[str, ...]) -> bool:
    return "." in filename and filename.rsplit(".", 1)[1].lower() in allowed


@bp.post("")
def upload():
    """支援一次上傳多張。回傳建立的 ImageRecord 清單。"""
    cfg, repo = get_config(), get_repo()
    files = request.files.getlist("files") or request.files.getlist("file")
    if not files:
        abort(400, "沒有收到檔案（欄位名 files）")

    created = []
    for f in files:
        if not f.filename or not _allowed(f.filename, cfg.allowed_ext):
            continue
        name = secure_filename(f.filename)
        rec = ImageRecord(filename=name)
        dest = cfg.upload_dir / f"{rec.id}_{name}"
        f.save(dest)
        with Image.open(dest) as im:
            rec.width, rec.height = im.size
        rec.path = str(dest)
        repo.add_image(rec)
        created.append(rec.to_dict())

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
    return send_file(Path(rec.path))


@bp.delete("/<image_id>")
def delete_image(image_id: str):
    """刪除一張上傳的照片，連同它的遮罩片段與檔案一起清掉。"""
    repo = get_repo()
    if not repo.get_image(image_id):
        abort(404)
    paths = repo.delete_image(image_id)          # 先從資料層移除
    for p in paths:                              # 再刪實體檔（原圖 + 遮罩 PNG）
        Path(p).unlink(missing_ok=True)
    return jsonify({"deleted": image_id, "files_removed": len(paths)})
