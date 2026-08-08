"""影像上傳與瀏覽 API（提案 demo 第 1 步：上傳一批照片）。"""
from __future__ import annotations

import hashlib
import io
import json
import mimetypes
import os
import tarfile
import tempfile
import zipfile

try:
    import py7zr
except ImportError:
    py7zr = None

from flask import Blueprint, Response, abort, jsonify, request, send_file, stream_with_context
from PIL import Image, ImageOps
from werkzeug.utils import secure_filename

from app.routes import (
    can_access_image,
    get_config,
    get_current_project_id,
    get_current_user_id,
    get_owned_image,
    get_repo,
    get_storage,
)
from app.models import ImageRecord

bp = Blueprint("images", __name__, url_prefix="/api/images")


cancelled_uploads: set[str] = set()


@bp.post("/cancel_upload")
def cancel_upload():
    """設定取消標記，中斷當前使用者正在執行的壓縮檔解壓/圖片處理迴圈。"""
    user_id = get_current_user_id()
    cancelled_uploads.add(user_id)
    return jsonify({"status": "cancelling"}), 200


def _allowed(filename: str, allowed: tuple[str, ...]) -> bool:
    return "." in filename and filename.rsplit(".", 1)[1].lower() in allowed


def _is_archive(filename: str, allowed_archive_ext: tuple[str, ...]) -> bool:
    if "." not in filename:
        return False
    lower = filename.lower()
    return any(lower.endswith(f".{ext}") for ext in allowed_archive_ext)


def _make_temp_dir():
    try:
        return tempfile.TemporaryDirectory(ignore_cleanup_errors=True)
    except TypeError:
        return tempfile.TemporaryDirectory()


def _extract_images_from_archive(
    filename: str,
    file_bytes: bytes,
    allowed_image_ext: tuple[str, ...],
    max_image_size: int,
) -> list[tuple[str, bytes]]:
    """從高效能壓縮檔 (.zip, .tar, .tar.gz 等) 提取符合副檔名與大小限制的影像檔案。"""
    extracted: list[tuple[str, bytes]] = []
    lower_name = filename.lower()

    def _should_keep(member_name: str) -> bool:
        base = os.path.basename(member_name)
        if not base or base.startswith(".") or member_name.startswith("__MACOSX"):
            return False
        if "." not in base:
            return False
        ext = base.rsplit(".", 1)[1].lower()
        return ext in allowed_image_ext

    # 1. ZIP 檔
    if lower_name.endswith(".zip"):
        try:
            with zipfile.ZipFile(io.BytesIO(file_bytes)) as zf:
                for member in zf.infolist():
                    if member.is_dir():
                        continue
                    if _should_keep(member.filename):
                        if member.file_size <= max_image_size:
                            try:
                                data = zf.read(member)
                                base_name = os.path.basename(member.filename)
                                extracted.append((base_name, data))
                            except Exception:
                                pass
        except Exception:
            pass

    # 2. TAR / GZ / BZ2 / XZ 檔
    elif any(
        lower_name.endswith(ext)
        for ext in (
            ".tar",
            ".tar.gz",
            ".tgz",
            ".tar.bz2",
            ".tbz2",
            ".tar.xz",
            ".txz",
            ".gz",
            ".bz2",
            ".xz",
        )
    ):
        try:
            with tarfile.open(fileobj=io.BytesIO(file_bytes), mode="r:*") as tf:
                for member in tf.getmembers():
                    if not member.isfile():
                        continue
                    if _should_keep(member.name):
                        if member.size <= max_image_size:
                            try:
                                f_obj = tf.extractfile(member)
                                if f_obj is not None:
                                    data = f_obj.read()
                                    base_name = os.path.basename(member.name)
                                    extracted.append((base_name, data))
                            except Exception:
                                pass
        except Exception:
            pass

    return extracted


def _collect_known_hashes(repo, storage, project_id: str | None = None) -> set[str]:
    """收齊指定專案中目前使用者看得到的照片雜湊值，供上傳去重比對。"""
    hashes = set()

    for img in repo.list_images(project_id=project_id):
        if not can_access_image(img):
            continue

        file_hash = getattr(img, "file_hash", "")
        if not file_hash and img.path:
            try:
                file_hash = hashlib.sha256(
                    storage.read_bytes(img.path)
                ).hexdigest()
            except Exception:
                continue
            repo.set_image_hash(img.id, file_hash)

        if file_hash:
            hashes.add(file_hash)

    return hashes


def _process_one_image(orig_filename: str, file_bytes: bytes, max_side: int = 1280) -> tuple[bytes, int, int, str]:
    extension = orig_filename.rsplit(".", 1)[1].lower() if "." in orig_filename else "png"
    default_mime = Image.MIME.get(extension.upper()) or f"image/{extension}"
    try:
        with Image.open(io.BytesIO(file_bytes)) as im:
            save_format = im.format or extension.upper()
            if save_format.upper() == "JPG":
                save_format = "JPEG"
            w, h = im.size

            exif = im.getexif()
            has_exif_trans = bool(exif and 0x0112 in exif and exif[0x0112] not in (0, 1))

            if max(w, h) <= max_side and not has_exif_trans:
                return file_bytes, w, h, default_mime

            processed_image = ImageOps.exif_transpose(im)
            if max(processed_image.size) > max_side:
                scale = max_side / max(processed_image.size)
                new_size = (
                    int(processed_image.size[0] * scale),
                    int(processed_image.size[1] * scale),
                )
                processed_image = processed_image.resize(
                    new_size,
                    Image.Resampling.BILINEAR,
                )

            out_w, out_h = processed_image.size
            output = io.BytesIO()
            processed_image.save(output, format=save_format, quality=85)
            mime = Image.MIME.get(save_format.upper()) or default_mime
            return output.getvalue(), out_w, out_h, mime
    except Exception:
        return file_bytes, 100, 100, default_mime


@bp.post("")
def upload():
    """支援一次上傳多張照片或壓縮檔 (zip/7z/tar)。支援一般 JSON 與 NDJSON 串流進度回傳。"""
    user_id = get_current_user_id()
    cancelled_uploads.discard(user_id)

    cfg, repo, storage = get_config(), get_repo(), get_storage()
    files = request.files.getlist("files") or request.files.getlist("file")
    if not files:
        abort(400, "沒有收到檔案（欄位名 files）")

    want_stream = request.args.get("stream") == "1" or "application/x-ndjson" in request.headers.get("Accept", "")

    items_to_process: list[tuple[str, bytes]] = []

    for f in files:
        if user_id in cancelled_uploads:
            cancelled_uploads.discard(user_id)
            break
        if not f.filename:
            continue

        raw_bytes = f.read()
        f.seek(0)

        if _is_archive(f.filename, cfg.allowed_archive_ext):
            extracted = _extract_images_from_archive(
                f.filename,
                raw_bytes,
                cfg.allowed_ext,
                cfg.max_image_size,
            )
            items_to_process.extend(extracted)
        elif _allowed(f.filename, cfg.allowed_ext):
            if len(raw_bytes) <= cfg.max_image_size:
                items_to_process.append((f.filename, raw_bytes))

    if not items_to_process:
        abort(400, "沒有有效的影像檔或解壓後無支援格式影像")

    total_items = len(items_to_process)
    project_id = request.form.get("project_id") or request.args.get("project_id") or get_current_project_id()
    known_hashes = _collect_known_hashes(repo, storage, project_id=project_id)

    def generate_upload_stream():
        created = []
        duplicates = []
        is_cancelled = False
        pending_records = []

        if want_stream:
            yield json.dumps({
                "event": "progress",
                "current": 0,
                "total": total_items,
                "created_count": 0,
                "filename": "",
            }, ensure_ascii=False) + "\n"

        batch_latest = []

        for idx, (orig_filename, file_bytes) in enumerate(items_to_process, start=1):
            if user_id in cancelled_uploads:
                cancelled_uploads.discard(user_id)
                is_cancelled = True
                break

            file_hash = hashlib.sha256(file_bytes).hexdigest()

            if file_hash in known_hashes:
                duplicates.append(orig_filename)
                if want_stream and (idx % 10 == 0 or idx == total_items or idx == 1):
                    yield json.dumps({
                        "event": "progress",
                        "current": idx,
                        "total": total_items,
                        "created_count": len(created),
                        "filename": orig_filename,
                        "latest_images": batch_latest,
                    }, ensure_ascii=False) + "\n"
                    batch_latest = []
                continue

            extension = orig_filename.rsplit(".", 1)[1].lower() if "." in orig_filename else "png"
            name = secure_filename(orig_filename) or f"uploaded.{extension}"

            rec = ImageRecord(
                owner_id=user_id,
                project_id=project_id,
                filename=name,
                file_hash=file_hash,
            )

            raw_to_save, width, height, content_type = _process_one_image(orig_filename, file_bytes)
            rec.width, rec.height = width, height

            rec.path = storage.save_bytes(
                f"images/{rec.id}_{name}",
                raw_to_save,
                content_type,
            )

            pending_records.append(rec)
            known_hashes.add(file_hash)
            rec_dict = rec.to_dict()
            created.append(rec_dict)
            batch_latest.append(rec_dict)

            if len(pending_records) >= 25 or idx == total_items:
                if hasattr(repo, "add_images"):
                    repo.add_images(pending_records)
                else:
                    for r in pending_records:
                        repo.add_image(r)
                pending_records.clear()

            if want_stream and (idx % 10 == 0 or idx == total_items or idx == 1):
                yield json.dumps({
                    "event": "progress",
                    "current": idx,
                    "total": total_items,
                    "created_count": len(created),
                    "filename": name,
                    "latest_images": batch_latest,
                }, ensure_ascii=False) + "\n"
                batch_latest = []

        if pending_records:
            if hasattr(repo, "add_images"):
                repo.add_images(pending_records)
            else:
                for r in pending_records:
                    repo.add_image(r)
            pending_records.clear()

        if want_stream:
            yield json.dumps({
                "event": "done",
                "cancelled": is_cancelled,
                "current": len(created),
                "total": total_items,
                "created": created,
                "duplicates": duplicates,
            }, ensure_ascii=False) + "\n"

    if want_stream:
        return Response(stream_with_context(generate_upload_stream()), mimetype="application/x-ndjson")

    # 非串流模式（相容原本的 API 與測試）
    created = []
    duplicates = []
    pending_records = []

    for idx, (orig_filename, file_bytes) in enumerate(items_to_process, start=1):
        if user_id in cancelled_uploads:
            cancelled_uploads.discard(user_id)
            break
        file_hash = hashlib.sha256(file_bytes).hexdigest()
        if file_hash in known_hashes:
            duplicates.append(orig_filename)
            continue
        extension = orig_filename.rsplit(".", 1)[1].lower() if "." in orig_filename else "png"
        name = secure_filename(orig_filename) or f"uploaded.{extension}"
        rec = ImageRecord(
            owner_id=user_id,
            project_id=project_id,
            filename=name,
            file_hash=file_hash,
        )
        raw_to_save, width, height, content_type = _process_one_image(orig_filename, file_bytes)
        rec.width, rec.height = width, height
        rec.path = storage.save_bytes(
            f"images/{rec.id}_{name}",
            raw_to_save,
            content_type,
        )
        pending_records.append(rec)
        created.append(rec.to_dict())

    if pending_records:
        if hasattr(repo, "add_images"):
            repo.add_images(pending_records)
        else:
            for r in pending_records:
                repo.add_image(r)

    if duplicates and not created:
        return jsonify({"error": "圖片已上傳過，請勿重複上傳"}), 400

    if not created:
        abort(400, "沒有有效的影像檔")
    return jsonify(created), 201


@bp.get("")
def list_images():
    project_id = request.args.get("project_id") or get_current_project_id()
    repo = get_repo()
    images = repo.list_images(project_id=project_id)
    accessible = [image for image in images if can_access_image(image)]

    result = []
    for image in accessible:
        d = image.to_dict()
        segs = repo.list_segments(image.id)
        d["segment_count"] = len(segs)
        tags = set()
        for s in segs:
            lbl = s.final_label or s.predicted_label
            if lbl:
                tags.add(lbl)
        d["segment_tags"] = list(tags)
        result.append(d)
    return jsonify(result)


@bp.get("/<image_id>")
def get_image_info(image_id: str):
    """取得單張照片的核心詮釋資料（ImageRecord）。"""
    rec = get_owned_image(image_id)
    return jsonify(rec.to_dict())


@bp.get("/<image_id>/file")
def image_file(image_id: str):
    """回傳原圖，給前端 canvas 顯示。"""
    rec = get_owned_image(image_id)
    try:
        data = get_storage().read_bytes(rec.path)
    except FileNotFoundError:
        abort(404)
    return send_file(
        io.BytesIO(data),
        mimetype=mimetypes.guess_type(rec.filename)[0] or "image/png",
        download_name=rec.filename,
        max_age=86400,
    )


@bp.delete("/<image_id>")
def delete_image(image_id: str):
    """刪除一張上傳的照片，連同它的遮罩片段與檔案一起清掉。"""
    repo = get_repo()
    get_owned_image(image_id)
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

    owned_ids = []
    for image_id in image_ids:
        image = repo.get_image(str(image_id))
        if image is not None and can_access_image(image):
            owned_ids.append(image.id)
    if len(owned_ids) != len(image_ids):
        abort(404)

    paths = repo.delete_images_batch(owned_ids)
    files_removed = sum(
        get_storage().delete(path)
        for path in paths
        if path
    )

    return jsonify({
        "deleted_ids": owned_ids,
        "files_removed": files_removed,
    }), 200
