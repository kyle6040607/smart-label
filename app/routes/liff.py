"""LIFF 頁面與標註任務路由。

頁面：
- GET /liff/
  LIFF 的共同 Endpoint URL，預設顯示建立標註任務頁面。
- GET /liff/create
  顯示建立標註任務頁面。
- GET /liff/upload
  舊版建立任務入口，與 /liff/create 共用頁面。
- GET /liff/tasks
  顯示進行中任務與歷史紀錄頁面。

任務 API：
- POST /liff/
  LIFF 共同入口的建立任務 API。
- POST /liff/create
  驗證 LINE 身分、儲存圖片並建立標註任務。
- POST /liff/upload
  舊版建立任務 API，與 /liff/create 共用處理流程。
- POST /liff/tasks
  驗證 LINE ID Token 並取得該使用者的任務清單。
- DELETE /liff/tasks/<task_id>
  驗證 LINE owner，清除終態任務及其專屬資料。
- POST /liff/tasks/<task_id>/append/context
  驗證追加圖片的任務並回傳固定 Prompt。
- POST /liff/uploads/init、/batch、/finalize
  建立暫存 session、分批上傳，最後只轉為 upload_ready。
- DELETE /liff/uploads/<session_id>/images/<image_id>
  在正式建立任務前移除一張暫存圖片。
- POST /liff/uploads/<session_id>/create-task
  建立專屬 Web Project、轉為 pending，並在 Commit 後觸發 Worker。

狀態與下載：
- GET /liff/tasks/<task_id>/status
  使用任務 token 查詢處理狀態。
- GET /liff/tasks/<task_id>/download
  使用任務 token 下載最新的 YOLO ZIP。

安全原則：
- 不信任前端傳入的 line_user_id。
- 使用 LINE 驗證結果中的 sub 識別使用者。
- 任務清單只回傳該 LINE 使用者自己的任務。
- 狀態查詢與 ZIP 下載必須驗證任務的隨機 token。
"""
import hashlib
import io
import re
import secrets
import time
from pathlib import Path

from PIL import Image, UnidentifiedImageError
from google.auth.exceptions import TransportError
from flask import (
    Blueprint,
    Response,
    current_app,
    jsonify,
    redirect,
    render_template,
    request,
    stream_with_context,
    url_for,
)

from werkzeug.utils import secure_filename

from app.models import AnnotationTask, ImageRecord
from app.routes import get_config, get_repo, get_storage
from app.services import line_login
from app.services.cloud_run_jobs import trigger_task_worker


bp = Blueprint(
    "liff",
    __name__,
    url_prefix="/liff",
)

_DOWNLOAD_CHUNK_SIZE = 1024 * 1024
_EXCLUDED_PAGE_SIZE = 20
_LIFF_UPLOAD_SESSION_TTL_SECONDS = 24 * 60 * 60
_LIFF_UPLOAD_BATCH_ID_PATTERN = re.compile(r"^[A-Za-z0-9_-]{1,64}$")


def _stream_file(storage, reference: str):
    """固定大小串流檔案，避免把整個 ZIP 載入記憶體。"""
    with storage.open_reader(reference) as reader:
        while chunk := reader.read(_DOWNLOAD_CHUNK_SIZE):
            yield chunk


def _liff_task_settings(cfg, project_id: str) -> dict:
    """建立不受 Web 動態參數影響的 LIFF 任務設定快照。"""
    return {
        "detection_confidence_threshold": cfg.liff_yolo_world_confidence,
        "export_confidence_threshold": cfg.liff_export_confidence_threshold,
        "yolo_imgsz": cfg.liff_yolo_imgsz,
        "model_name": "yolov8x-worldv2",
        "model_version": "v8.4.0",
        "project_id": project_id,
        "exclusion_rule": (
            "detection_confidence < export_confidence_threshold"
        ),
    }


def _resolve_liff_owner(repo, profile: dict) -> tuple[str, str, str]:
    """由已驗證的 LINE profile 決定 Web owner 與預設專案。"""
    line_user_id = profile["sub"]
    display_name = profile.get("name", "")
    user = repo.get_user_by_line_id(line_user_id)
    web_user_id = user.id if user is not None else ""
    web_project_id = (
        repo.get_or_create_default_project(web_user_id).id
        if web_user_id
        else ""
    )
    response_display_name = (
        user.display_name or user.username
        if user is not None
        else display_name
    )
    return web_user_id, web_project_id, response_display_name


def _resolve_liff_identity(repo, profile: dict) -> tuple[str, str]:
    """只解析 Web owner，不在暫存上傳階段建立或選取 Project。"""
    user = repo.get_user_by_line_id(profile["sub"])
    if user is None:
        return "", profile.get("name", "")
    return user.id, user.display_name or user.username


def _upload_snapshot(task: AnnotationTask) -> dict:
    return dict(task.settings_snapshot.get("upload") or {})


def _upload_expired(task: AnnotationTask) -> bool:
    expires_at = float(_upload_snapshot(task).get("expires_at", 0.0))
    return bool(expires_at and expires_at < time.time())


def _expire_stale_uploads(repo, storage, line_user_id: str) -> int:
    """清除同一 LINE 使用者逾期且未 finalize 的 LIFF 上傳。"""
    expired_count = 0
    for task in repo.list_tasks_by_line_user_id(line_user_id):
        if (
            task.status not in {"uploading", "upload_ready"}
            or not _upload_expired(task)
        ):
            continue

        paths = repo.delete_images_batch(task.image_ids)
        for path in paths:
            try:
                storage.delete(path)
            except Exception:  # noqa: BLE001 清理失敗不可阻擋新的上傳
                current_app.logger.warning(
                    "無法刪除逾期 LIFF 上傳檔案：%s",
                    path,
                    exc_info=True,
                )
        try:
            storage.delete_prefix(f"liff-uploads/{task.id}")
        except Exception:  # noqa: BLE001 清理失敗不可阻擋新的上傳
            current_app.logger.warning(
                "無法清除逾期 LIFF 上傳前綴：task_id=%s",
                task.id,
                exc_info=True,
            )

        task.image_ids = []
        task.status = "upload_expired"
        task.completion_reason = "upload_expired"
        task.error_message = "圖片上傳工作階段已過期"
        repo.update_task(task)
        expired_count += 1

    return expired_count


def _verify_liff_profile(id_token: str):
    """驗證 LIFF ID Token 並取得可信任的 LINE Profile，只採用 LINE 驗證結果裡的 sub"""
    if not id_token:
        return None, (jsonify({"ok": False, "message": "缺少 LINE ID Token"}), 401)

    cfg = get_config()

    if not cfg.line_login_channel_id:
        return None, (
            jsonify({"ok": False, "message": "伺服器尚未設定 LINE_LOGIN_CHANNEL_ID"}),
            500,
        )

    try:
        profile = line_login.verify_id_token(id_token, cfg.line_login_channel_id, None)
    except Exception as error:
        error_message = str(error)
        current_app.logger.exception("LIFF ID Token 驗證失敗：%s", error_message)

        if "IdToken expired" in error_message:
            return None, (
                jsonify(
                    {
                        "ok": False,
                        "code": "LINE_ID_TOKEN_EXPIRED",
                        "message": "LINE 登入資料已過期，正在重新登入",
                    }
                ),
                401,
            )

        return None, (
            jsonify(
                {
                    "ok": False,
                    "code": "LINE_ID_TOKEN_INVALID",
                    "message": "LINE 身分驗證失敗，請重新登入",
                }
            ),
            401,
        )

    if not profile.get("sub"):
        return None, (jsonify({"ok": False, "message": "LINE 未提供使用者識別碼"}), 401)

    return profile, None


def _task_summary(task: AnnotationTask) -> dict:
    """產生 LIFF 任務清單需要的安全資料。"""
    result = {
        "task_id": task.id,
        "prompt": task.prompt,
        "task_status": task.status,
        "image_count": len(task.image_ids),
        "processed_image_count": len(task.processed_image_ids),
        "dataset_version": task.dataset_version,
        "attempt_count": task.attempt_count,
        "segment_count": task.segment_count,
        "exported_count": task.exported_count,
        "excluded_count": task.excluded_count,
        "no_detection_count": len(task.no_detection_image_ids),
        "zip_available": bool(task.dataset_zip_path and task.dataset_version > 0),
        "completion_reason": task.completion_reason,
        "can_add_images": task.status == "completed",
        "can_delete": task.status in {"completed", "failed", "deleting"},
        "created_at": task.created_at,
        "updated_at": task.updated_at,
    }

    if task.status == "failed":
        result["error_message"] = task.error_message or "處理任務時發生錯誤"

    if task.dataset_zip_path and task.dataset_version > 0:
        result["download_url"] = url_for(
            "liff.download_task_dataset",
            task_id=task.id,
            token=task.download_token,
        )

    return result


@bp.get("/")
@bp.get("/upload")
@bp.get("/create")
def upload_page():
    """顯示 LIFF 圖片與 Prompt 上傳頁面。"""

    cfg = get_config()

    return render_template(
        "liff/upload.html",
        liff_id=cfg.liff_id,
        upload_batch_max_images=cfg.liff_upload_batch_max_images,
        upload_batch_max_bytes=cfg.liff_upload_batch_max_bytes,
        upload_max_total_bytes=cfg.liff_upload_max_total_bytes,
    )


@bp.get("/tasks")
def tasks_page():
    """顯示目前 LINE 使用者的標註任務。"""
    cfg = get_config()

    return render_template("liff/tasks.html", liff_id=cfg.liff_id)


@bp.post("/tasks")
def list_tasks():
    """
    驗證 LINE 身分後取得該使用者的任務清單。
    不接受前端傳入 line_user_id。
    只採用 LINE 驗證結果中的 sub。
    只查詢該 LINE 使用者自己的任務。
    回傳順序是最近更新的任務在前。
    """
    payload = request.get_json(silent=True)

    if not isinstance(payload, dict):
        payload = {}

    id_token = str(payload.get("id_token", "")).strip()
    profile, verification_error = _verify_liff_profile(id_token)

    if verification_error is not None:
        return verification_error

    line_user_id = profile["sub"]
    repo = get_repo()
    _expire_stale_uploads(repo, get_storage(), line_user_id)

    tasks = [
        task
        for task in repo.list_tasks_by_line_user_id(line_user_id)
        if task.status not in {
            "uploading",
            "upload_ready",
            "upload_expired",
            "upload_merged",
        }
    ]

    return jsonify(
        {
            "ok": True,
            "task_count": len(tasks),
            "tasks": [_task_summary(task) for task in tasks],
        }
    )


@bp.delete("/tasks/<task_id>")
def delete_task(task_id: str):
    """刪除目前 LINE 使用者擁有的終態任務及其專屬資料。"""
    payload = request.get_json(silent=True) or {}
    id_token = str(payload.get("id_token", "")).strip()
    profile, verification_error = _verify_liff_profile(id_token)
    if verification_error is not None:
        return verification_error

    repo = get_repo()
    task, deletion_state, paths = repo.prepare_liff_task_deletion(
        task_id,
        profile["sub"],
    )
    if deletion_state == "not_found":
        return jsonify({"ok": False, "message": "找不到標註任務"}), 404
    if deletion_state == "not_deletable":
        return (
            jsonify(
                {
                    "ok": False,
                    "code": "TASK_NOT_DELETABLE",
                    "message": "任務執行中，請等待完成後再刪除",
                    "task_status": task.status if task else "",
                }
            ),
            409,
        )

    storage = get_storage()
    try:
        for path in paths:
            storage.delete(path)
        for prefix in (
            f"liff-uploads/{task_id}",
            f"previews/tasks/{task_id}",
            f"datasets/{task_id}",
        ):
            storage.delete_prefix(prefix)
    except Exception:  # noqa: BLE001 保留 deleting 狀態供使用者重試
        current_app.logger.exception(
            "LIFF 任務檔案清理失敗，等待重試：task_id=%s",
            task_id,
        )
        return (
            jsonify(
                {
                    "ok": False,
                    "code": "TASK_DELETE_STORAGE_FAILED",
                    "message": "檔案清理暫時失敗，請稍後重試刪除",
                    "task_status": "deleting",
                }
            ),
            503,
        )

    deleted = repo.finalize_liff_task_deletion(task_id, profile["sub"])
    if deleted is None:
        return (
            jsonify(
                {
                    "ok": False,
                    "code": "TASK_DELETE_CONFLICT",
                    "message": "任務狀態已變更，請重新整理後再試",
                }
            ),
            409,
        )

    return jsonify(
        {
            "ok": True,
            "message": "標註任務已刪除",
            "task_id": task_id,
            **deleted,
        }
    )


@bp.post("/tasks/<task_id>/append/context")
def get_append_upload_context(task_id: str):
    """驗證 LINE owner，並提供追加頁面需要的不可竄改任務資訊。"""
    payload = request.get_json(silent=True) or {}
    id_token = str(payload.get("id_token", "")).strip()
    profile, verification_error = _verify_liff_profile(id_token)
    if verification_error is not None:
        return verification_error

    task = get_repo().get_task(task_id)
    if task is None or task.line_user_id != profile["sub"]:
        return jsonify({"ok": False, "message": "找不到標註任務"}), 404
    if task.status != "completed":
        return (
            jsonify(
                {
                    "ok": False,
                    "message": "只有已完成的任務可以新增照片",
                    "task_status": task.status,
                }
            ),
            409,
        )

    return jsonify(
        {
            "ok": True,
            "task_id": task.id,
            "prompt": task.prompt,
            "image_count": len(task.image_ids),
            "dataset_version": task.dataset_version,
        }
    )


@bp.post("/uploads/init")
def initialize_chunked_upload():
    """建立 LIFF 分批上傳 session；此時 Worker 尚不可領取。"""
    payload = request.get_json(silent=True) or {}
    id_token = str(payload.get("id_token", "")).strip()
    target_task_id = str(payload.get("target_task_id", "")).strip()

    try:
        expected_image_count = int(payload.get("expected_image_count", 0))
    except (TypeError, ValueError):
        expected_image_count = 0
    try:
        expected_total_bytes = int(payload.get("expected_total_bytes", 0))
    except (TypeError, ValueError):
        expected_total_bytes = -1

    cfg = get_config()
    if expected_image_count < 1:
        return (
            jsonify(
                {
                    "ok": False,
                    "message": "每個任務至少需選擇 1 張圖片",
                }
            ),
            400,
        )
    if (
        cfg.liff_upload_max_images > 0
        and expected_image_count > cfg.liff_upload_max_images
    ):
        return (
            jsonify(
                {
                    "ok": False,
                    "message": (
                        "每個任務最多選擇 "
                        f"{cfg.liff_upload_max_images} 張圖片"
                    ),
                }
            ),
            400,
        )
    if expected_total_bytes < 0:
        return jsonify({"ok": False, "message": "圖片總大小無效"}), 400
    if (
        cfg.liff_upload_max_total_bytes > 0
        and expected_total_bytes > cfg.liff_upload_max_total_bytes
    ):
        return (
            jsonify(
                {
                    "ok": False,
                    "message": "所選圖片總大小超過單一任務上限",
                    "max_total_bytes": cfg.liff_upload_max_total_bytes,
                }
            ),
            413,
        )

    profile, verification_error = _verify_liff_profile(id_token)
    if verification_error is not None:
        return verification_error

    repo = get_repo()
    _expire_stale_uploads(repo, get_storage(), profile["sub"])
    active_upload_count = sum(
        task.status in {"uploading", "upload_ready"}
        for task in repo.list_tasks_by_line_user_id(profile["sub"])
    )
    if (
        cfg.liff_upload_max_concurrent_sessions > 0
        and active_upload_count >= cfg.liff_upload_max_concurrent_sessions
    ):
        return (
            jsonify(
                {
                    "ok": False,
                    "message": "目前已有圖片正在上傳，請完成後再建立新任務",
                }
            ),
            429,
        )
    target_task = None
    if target_task_id:
        target_task = repo.get_task(target_task_id)
        if target_task is None or target_task.line_user_id != profile["sub"]:
            return jsonify({"ok": False, "message": "找不到標註任務"}), 404
        if target_task.status != "completed":
            return (
                jsonify(
                    {
                        "ok": False,
                        "message": "只有已完成的任務可以新增照片",
                        "task_status": target_task.status,
                    }
                ),
                409,
            )
        web_user_id = target_task.user_id
        linked_user = repo.get_user_by_line_id(profile["sub"])
        display_name = (
            linked_user.display_name or linked_user.username
            if linked_user is not None
            else profile.get("name", "")
        )
    else:
        web_user_id, display_name = _resolve_liff_identity(
            repo,
            profile,
        )
    expires_at = time.time() + _LIFF_UPLOAD_SESSION_TTL_SECONDS
    base_settings = (
        {
            key: value
            for key, value in target_task.settings_snapshot.items()
            if key != "upload"
        }
        if target_task is not None
        else {}
    )
    task = AnnotationTask(
        user_id=web_user_id,
        line_user_id=profile["sub"],
        prompt=target_task.prompt if target_task is not None else "",
        status="uploading",
    )
    # 暫存圖片先使用 session id 作為不可見的 staging project_id；
    # 正式建立任務時才新增 projects row，追加模式則改綁原 Project。
    task.project_id = task.id
    settings_snapshot = (
        base_settings
        if target_task is not None
        else _liff_task_settings(cfg, task.id)
    )
    settings_snapshot["upload"] = {
        "expected_image_count": expected_image_count,
        "expected_total_bytes": expected_total_bytes,
        "uploaded_bytes": 0,
        "completed_batches": {},
        "completed_batch_bytes": {},
        "expires_at": expires_at,
        "finalized_at": 0.0,
        "target_task_id": target_task_id,
        "image_bytes": {},
    }
    task.settings_snapshot = settings_snapshot
    repo.add_task(task)

    return (
        jsonify(
            {
                "ok": True,
                "session_id": task.id,
                "task_status": task.status,
                "expected_image_count": expected_image_count,
                "uploaded_count": 0,
                "uploaded_bytes": 0,
                "expires_at": expires_at,
                "display_name": display_name,
                "batch_max_images": cfg.liff_upload_batch_max_images,
                "batch_max_bytes": cfg.liff_upload_batch_max_bytes,
                "max_total_bytes": cfg.liff_upload_max_total_bytes,
                "append_mode": bool(target_task_id),
                "target_task_id": target_task_id,
            }
        ),
        201,
    )


@bp.post("/uploads/<session_id>/batch")
def upload_image_batch(session_id: str):
    """驗證並儲存一小批 LIFF 圖片；相同 batch_id 可安全重送。"""
    id_token = request.form.get("id_token", "").strip()
    batch_id = request.form.get("batch_id", "").strip()
    profile, verification_error = _verify_liff_profile(id_token)
    if verification_error is not None:
        return verification_error

    repo = get_repo()
    task = repo.get_task(session_id)
    if task is None or task.line_user_id != profile["sub"]:
        return jsonify({"ok": False, "message": "找不到上傳工作階段"}), 404
    if _upload_expired(task):
        _expire_stale_uploads(repo, get_storage(), profile["sub"])
        return jsonify({"ok": False, "message": "上傳工作階段已過期"}), 410
    if not _LIFF_UPLOAD_BATCH_ID_PATTERN.fullmatch(batch_id):
        return jsonify({"ok": False, "message": "無效的上傳批次識別碼"}), 400

    upload = _upload_snapshot(task)
    completed_batches = dict(upload.get("completed_batches") or {})
    if batch_id in completed_batches:
        completed_image_ids = list(completed_batches[batch_id])
        duplicate_client_ids = [
            str(client_id).strip()
            for client_id in request.form.getlist("client_ids")
        ]
        return jsonify(
            {
                "ok": True,
                "duplicate_batch": True,
                "batch_id": batch_id,
                "accepted_count": len(completed_image_ids),
                "items": [
                    {
                        "client_id": (
                            duplicate_client_ids[index]
                            if index < len(duplicate_client_ids)
                            else f"image-{index}"
                        ),
                        "image_id": image_id,
                        "filename": (
                            repo.get_image(image_id).filename
                            if repo.get_image(image_id) is not None
                            else ""
                        ),
                    }
                    for index, image_id in enumerate(completed_image_ids)
                ],
                "uploaded_count": len(task.image_ids),
                "uploaded_bytes": int(upload.get("uploaded_bytes", 0)),
                "expected_image_count": int(
                    upload.get("expected_image_count", 0)
                ),
            }
        )
    if task.status != "uploading":
        return jsonify({"ok": False, "message": "上傳工作階段已結束"}), 409

    images = [
        image
        for image in request.files.getlist("images")
        if image and image.filename
    ]
    if not images:
        return jsonify({"ok": False, "message": "此批次沒有圖片"}), 400
    client_ids = [
        str(client_id).strip()
        for client_id in request.form.getlist("client_ids")
    ]
    if client_ids and len(client_ids) != len(images):
        return jsonify({"ok": False, "message": "圖片與前端識別碼數量不符"}), 400
    if not client_ids:
        client_ids = [f"image-{index}" for index in range(len(images))]
    cfg = get_config()
    if len(images) > cfg.liff_upload_batch_max_images:
        return (
            jsonify(
                {
                    "ok": False,
                    "message": (
                        f"每批最多 {cfg.liff_upload_batch_max_images} 張圖片"
                    ),
                }
            ),
            413,
        )

    expected_image_count = int(upload.get("expected_image_count", 0))
    if len(task.image_ids) + len(images) > expected_image_count:
        return jsonify({"ok": False, "message": "上傳圖片數量超過預期"}), 409

    validated_images = []
    total_bytes = 0

    for index, (image, client_id) in enumerate(zip(images, client_ids)):
        extension = Path(image.filename).suffix.lower().lstrip(".")
        if not extension or extension not in cfg.allowed_ext:
            return (
                jsonify(
                    {
                        "ok": False,
                        "message": (
                            f"{image.filename} 格式不支援，"
                            f"允許格式：{', '.join(cfg.allowed_ext)}"
                        ),
                    }
                ),
                400,
            )

        image.stream.seek(0)
        image_bytes = image.stream.read()
        image.stream.seek(0)
        total_bytes += len(image_bytes)
        if len(image_bytes) > cfg.max_image_size:
            return (
                jsonify(
                    {
                        "ok": False,
                        "message": f"{image.filename} 超過單張圖片大小限制",
                    }
                ),
                413,
            )
        if total_bytes > cfg.liff_upload_batch_max_bytes:
            return (
                jsonify(
                    {
                        "ok": False,
                        "message": "此批次圖片總大小超過上傳限制",
                        "batch_max_bytes": cfg.liff_upload_batch_max_bytes,
                    }
                ),
                413,
            )

        try:
            with Image.open(io.BytesIO(image_bytes)) as uploaded_image:
                uploaded_image.load()
                width = uploaded_image.width
                height = uploaded_image.height
        except (Image.DecompressionBombError, UnidentifiedImageError, OSError):
            return (
                jsonify(
                    {
                        "ok": False,
                        "message": f"{image.filename} 不是有效的圖片檔案",
                    }
                ),
                400,
            )

        image_id = hashlib.sha256(
            f"{task.id}:{batch_id}:{index}".encode("utf-8")
        ).hexdigest()[:12]
        validated_images.append(
            (
                image_id,
                client_id,
                image,
                extension,
                image_bytes,
                width,
                height,
            )
        )

    uploaded_bytes = int(upload.get("uploaded_bytes", 0))
    expected_total_bytes = int(upload.get("expected_total_bytes", 0))
    if (
        cfg.liff_upload_max_total_bytes > 0
        and uploaded_bytes + total_bytes > cfg.liff_upload_max_total_bytes
    ):
        return (
            jsonify(
                {
                    "ok": False,
                    "message": "此任務累計圖片大小超過上限",
                    "max_total_bytes": cfg.liff_upload_max_total_bytes,
                }
            ),
            413,
        )
    if (
        expected_total_bytes > 0
        and uploaded_bytes + total_bytes > expected_total_bytes
    ):
        return jsonify({"ok": False, "message": "上傳內容超過預期大小"}), 409

    storage = get_storage()
    batch_image_ids = []
    batch_items = []
    image_bytes_by_id = {}
    for (
        image_id,
        client_id,
        image,
        extension,
        image_bytes,
        width,
        height,
    ) in validated_images:
        existing = repo.get_image(image_id)
        if existing is not None:
            if (
                existing.owner_id != task.user_id
                or existing.project_id != task.project_id
            ):
                return jsonify({"ok": False, "message": "圖片識別碼衝突"}), 409
            batch_image_ids.append(existing.id)
            batch_items.append(
                {
                    "client_id": client_id,
                    "image_id": existing.id,
                    "filename": existing.filename,
                }
            )
            image_bytes_by_id[existing.id] = len(image_bytes)
            continue

        safe_original_filename = secure_filename(image.filename)
        if not safe_original_filename:
            safe_original_filename = f"upload.{extension}"
        image_record = ImageRecord(
            id=image_id,
            owner_id=task.user_id,
            project_id=task.project_id,
            filename=safe_original_filename,
            width=width,
            height=height,
            file_hash=hashlib.sha256(image_bytes).hexdigest(),
        )
        image_record.path = storage.save_bytes(
            (
                f"liff-uploads/{task.id}/"
                f"{image_record.id}.{extension}"
            ),
            image_bytes,
            image.mimetype or "application/octet-stream",
        )
        repo.add_image(image_record)
        batch_image_ids.append(image_record.id)
        batch_items.append(
            {
                "client_id": client_id,
                "image_id": image_record.id,
                "filename": image_record.filename,
            }
        )
        image_bytes_by_id[image_record.id] = len(image_bytes)

    updated_task, recorded = repo.record_liff_upload_batch(
        task.id,
        profile["sub"],
        batch_id,
        batch_image_ids,
        total_bytes,
        image_bytes_by_id,
    )
    if updated_task is None:
        return jsonify({"ok": False, "message": "找不到上傳工作階段"}), 404
    if not recorded and updated_task.status != "uploading":
        return jsonify({"ok": False, "message": "上傳工作階段已結束"}), 409

    return (
        jsonify(
            {
                "ok": True,
                "duplicate_batch": not recorded,
                "batch_id": batch_id,
                "accepted_count": len(batch_image_ids),
                "items": batch_items,
                "uploaded_count": len(updated_task.image_ids),
                "uploaded_bytes": int(
                    _upload_snapshot(updated_task).get("uploaded_bytes", 0)
                ),
                "expected_image_count": expected_image_count,
            }
        ),
        201 if recorded else 200,
    )


@bp.post("/uploads/<session_id>/finalize")
def finalize_chunked_upload(session_id: str):
    """確認所有圖片已上傳，只轉為 upload_ready，不啟動 Worker。"""
    payload = request.get_json(silent=True) or {}
    id_token = str(payload.get("id_token", "")).strip()
    profile, verification_error = _verify_liff_profile(id_token)
    if verification_error is not None:
        return verification_error

    repo = get_repo()
    task = repo.get_task(session_id)
    if task is None or task.line_user_id != profile["sub"]:
        return jsonify({"ok": False, "message": "找不到上傳工作階段"}), 404
    if _upload_expired(task):
        _expire_stale_uploads(repo, get_storage(), profile["sub"])
        return jsonify({"ok": False, "message": "上傳工作階段已過期"}), 410

    upload = _upload_snapshot(task)
    expected_image_count = int(upload.get("expected_image_count", 0))
    if task.status == "uploading" and len(task.image_ids) != expected_image_count:
        return (
            jsonify(
                {
                    "ok": False,
                    "code": "UPLOAD_INCOMPLETE",
                    "message": "圖片尚未全部上傳完成",
                    "uploaded_count": len(task.image_ids),
                    "expected_image_count": expected_image_count,
                }
            ),
            409,
        )

    expected_total_bytes = int(upload.get("expected_total_bytes", 0))
    uploaded_bytes = int(upload.get("uploaded_bytes", 0))
    if (
        task.status == "uploading"
        and expected_total_bytes > 0
        and uploaded_bytes != expected_total_bytes
    ):
        return (
            jsonify(
                {
                    "ok": False,
                    "code": "UPLOAD_INCOMPLETE",
                    "message": "圖片資料尚未全部上傳完成",
                    "uploaded_bytes": uploaded_bytes,
                    "expected_total_bytes": expected_total_bytes,
                }
            ),
            409,
        )

    if task.status == "uploading":
        storage = get_storage()
        missing_image_ids = []
        for image_id in task.image_ids:
            image_record = repo.get_image(image_id)
            if (
                image_record is None
                or not image_record.path
                or not storage.exists(image_record.path)
            ):
                missing_image_ids.append(image_id)
        if missing_image_ids:
            return (
                jsonify(
                    {
                        "ok": False,
                        "code": "UPLOAD_INCOMPLETE",
                        "message": "部分圖片尚未完成儲存",
                        "missing_image_ids": missing_image_ids,
                    }
                ),
                409,
            )

    ready_task, transitioned = repo.mark_liff_upload_ready(
        task.id,
        profile["sub"],
    )
    if ready_task is None:
        return jsonify({"ok": False, "message": "找不到上傳工作階段"}), 404
    if not transitioned and ready_task.status != "upload_ready":
        return jsonify({"ok": False, "message": "圖片尚未全部上傳完成"}), 409

    return (
        jsonify(
            {
                "ok": True,
                "message": "圖片上傳完成，請確認清單後建立標註任務",
                "session_id": ready_task.id,
                "task_status": ready_task.status,
                "image_count": len(ready_task.image_ids),
                "uploaded_bytes": int(
                    _upload_snapshot(ready_task).get("uploaded_bytes", 0)
                ),
                "already_ready": not transitioned,
                "job_triggered": False,
            }
        ),
        200,
    )


@bp.delete("/uploads/<session_id>/images/<image_id>")
def delete_uploaded_image(session_id: str, image_id: str):
    """在建立標註任務前，刪除目前 LINE 使用者的一張暫存圖片。"""
    payload = request.get_json(silent=True) or {}
    id_token = str(payload.get("id_token", "")).strip()
    profile, verification_error = _verify_liff_profile(id_token)
    if verification_error is not None:
        return verification_error

    repo = get_repo()
    task = repo.get_task(session_id)
    if task is None or task.line_user_id != profile["sub"]:
        return jsonify({"ok": False, "message": "找不到上傳工作階段"}), 404
    if task.status != "upload_ready":
        return (
            jsonify(
                {
                    "ok": False,
                    "message": "只有尚未建立任務的已上傳圖片可以刪除",
                    "task_status": task.status,
                }
            ),
            409,
        )
    if image_id not in task.image_ids:
        return jsonify({"ok": False, "message": "找不到已上傳圖片"}), 404

    image = repo.get_image(image_id)
    if image is None or image.project_id != task.id:
        return jsonify({"ok": False, "message": "找不到已上傳圖片"}), 404

    # 暫存 session 不會被 Worker 使用；先做可重試的冪等檔案刪除，
    # 再於 Repository transaction 中移除資料列與 session reference。
    if image.path:
        try:
            get_storage().delete(image.path)
        except Exception:  # noqa: BLE001 DB reference 保留，使用者可重試
            current_app.logger.exception(
                "LIFF 暫存圖片刪除失敗：session_id=%s image_id=%s",
                session_id,
                image_id,
            )
            return jsonify({"ok": False, "message": "圖片刪除失敗，請重試"}), 503

    updated_task, _, removed = repo.remove_liff_upload_image(
        session_id,
        profile["sub"],
        image_id,
    )
    if updated_task is None:
        return jsonify({"ok": False, "message": "找不到上傳工作階段"}), 404
    if not removed:
        return jsonify({"ok": False, "message": "圖片狀態已變更"}), 409

    return jsonify(
        {
            "ok": True,
            "message": "圖片已移除",
            "image_id": image_id,
            "image_count": len(updated_task.image_ids),
            "uploaded_bytes": int(
                _upload_snapshot(updated_task).get("uploaded_bytes", 0)
            ),
        }
    )


@bp.post("/uploads/<session_id>/create-task")
def create_task_from_upload(session_id: str):
    """使用確認後的暫存圖片建立任務；成功 Commit 後才觸發 Worker。"""
    payload = request.get_json(silent=True) or {}
    id_token = str(payload.get("id_token", "")).strip()
    prompt = str(payload.get("prompt", "")).strip()
    requested_project_name = str(payload.get("project_name", "")).strip()
    profile, verification_error = _verify_liff_profile(id_token)
    if verification_error is not None:
        return verification_error

    repo = get_repo()
    session = repo.get_task(session_id)
    if session is None or session.line_user_id != profile["sub"]:
        return jsonify({"ok": False, "message": "找不到上傳工作階段"}), 404
    upload = _upload_snapshot(session)
    target_task_id = str(upload.get("target_task_id", ""))
    uploaded_image_count = len(session.image_ids)

    if not target_task_id:
        if not prompt:
            return jsonify({"ok": False, "message": "請輸入想要標註的物件"}), 400
        if len(prompt) > 200:
            return jsonify({"ok": False, "message": "標註內容不可超過 200 字"}), 400
    if len(requested_project_name) > 128:
        return jsonify({"ok": False, "message": "專案名稱不可超過 128 字"}), 400
    if session.status == "upload_ready" and not session.image_ids:
        return jsonify({"ok": False, "message": "請至少保留一張圖片"}), 409

    if target_task_id:
        finalized_task, transitioned = repo.finalize_liff_append_upload(
            session.id,
            profile["sub"],
        )
    else:
        project_name = requested_project_name or (
            f"{prompt[:40]} - {time.strftime('%Y/%m/%d %H:%M')}"
        )
        finalized_task, transitioned = repo.create_liff_annotation_task(
            session.id,
            profile["sub"],
            prompt,
            project_name,
        )

    if finalized_task is None:
        return jsonify({"ok": False, "message": "找不到上傳工作階段"}), 404
    already_created = bool(upload.get("created_at")) or (
        session.status == "upload_merged"
    )
    if not transitioned and not already_created:
        return (
            jsonify(
                {
                    "ok": False,
                    "message": "上傳工作階段目前無法建立標註任務",
                    "task_status": finalized_task.status,
                }
            ),
            409,
        )

    job_operation_name = ""
    if transitioned:
        try:
            job_operation_name = trigger_task_worker(get_config())
        except Exception:  # noqa: BLE001 pending 已 Commit，不可反向回滾
            current_app.logger.exception(
                "LIFF 任務已排隊，但 Cloud Run Job 觸發失敗：task_id=%s",
                finalized_task.id,
            )

    return (
        jsonify(
            {
                "ok": True,
                "message": (
                    "照片已新增，標註任務重新排隊"
                    if target_task_id
                    else "標註任務建立成功"
                ),
                "task_id": finalized_task.id,
                "project_id": finalized_task.project_id,
                "task_status": finalized_task.status,
                "image_count": len(finalized_task.image_ids),
                "added_image_count": int(
                    upload.get("merged_image_count", uploaded_image_count)
                ),
                "job_triggered": bool(job_operation_name),
                "already_created": not transitioned,
                "status_url": url_for(
                    "liff.task_status",
                    task_id=finalized_task.id,
                    token=finalized_task.download_token,
                ),
            }
        ),
        202 if transitioned else 200,
    )


@bp.post("/")
@bp.post("/upload")
@bp.post("/create")
def upload_task():
    """驗證 LINE 身分並儲存 LIFF 上傳的圖片與 Prompt。"""

    prompt = request.form.get("prompt", "").strip()
    id_token = request.form.get("id_token", "").strip()

    images = [
        image for image in request.files.getlist("images") if image and image.filename
    ]

    if not images:
        return jsonify({"ok": False, "message": "請至少選擇一張圖片"}), 400

    if not prompt:
        return jsonify({"ok": False, "message": "請輸入標註內容"}), 400

    profile, verification_error = _verify_liff_profile(id_token)

    if verification_error is not None:
        return verification_error

    line_user_id = profile["sub"]

    cfg = get_config()
    repo = get_repo()
    storage = get_storage()

    validated_images = []

    for image in images:
        extension = Path(image.filename).suffix.lower().lstrip(".")

        if not extension or extension not in cfg.allowed_ext:
            return (
                jsonify(
                    {
                        "ok": False,
                        "message": (
                            f"{image.filename} 格式不支援，"
                            f"允許格式：{', '.join(cfg.allowed_ext)}"
                        ),
                    }
                ),
                400,
            )
        try:
            image.stream.seek(0)
            image_bytes = image.stream.read()

            with Image.open(io.BytesIO(image_bytes)) as uploaded_image:
                uploaded_image.load()
                width = uploaded_image.width
                height = uploaded_image.height

        except (UnidentifiedImageError, OSError):
            image.stream.seek(0)
            return (
                jsonify(
                    {
                        "ok": False,
                        "message": f"{image.filename} 不是有效的圖片檔案",
                    }
                ),
                400,
            )

        image.stream.seek(0)
        validated_images.append(
            (
                image,
                extension,
                image_bytes,
                width,
                height,
            )
        )

    display_name = profile.get("name", "")

    # 查詢這位 LINE 使用者是否已綁定 Web 帳號
    user = repo.get_user_by_line_id(line_user_id)

    web_user_id = user.id if user is not None else ""
    web_project_id = (
        repo.get_or_create_default_project(web_user_id).id
        if web_user_id
        else ""
    )

    response_display_name = (
        user.display_name or user.username if user is not None else display_name
    )

    saved_images = []

    for (
        image,
        extension,
        image_bytes,
        width,
        height,
    ) in validated_images:
        safe_original_filename = secure_filename(image.filename)

        if not safe_original_filename:
            safe_original_filename = f"upload.{extension}"

        # 一併存下雜湊值，否則網頁端去重時得回頭把整張圖抓下來重算
        image_record = ImageRecord(
            owner_id=web_user_id,
            project_id=web_project_id,
            filename=safe_original_filename,
            file_hash=hashlib.sha256(image_bytes).hexdigest(),
        )

        saved_filename = f"{image_record.id}.{extension}"

        image_record.width = width
        image_record.height = height
        image_record.path = storage.save_bytes(
            f"images/{saved_filename}",
            image_bytes,
            image.mimetype or "application/octet-stream",
        )

        repo.add_image(image_record)

        saved_images.append(
            {
                "image_id": image_record.id,
                "original_filename": image.filename,
                "saved_filename": saved_filename,
                "width": image_record.width,
                "height": image_record.height,
            }
        )

    task = AnnotationTask(
        user_id=web_user_id,
        line_user_id=line_user_id,
        prompt=prompt,
        image_ids=[image_info["image_id"] for image_info in saved_images],
        settings_snapshot={
            "detection_confidence_threshold": cfg.liff_yolo_world_confidence,
            "export_confidence_threshold": cfg.liff_export_confidence_threshold,
            "yolo_imgsz": cfg.liff_yolo_imgsz,
            "model_name": "yolov8x-worldv2",
            "model_version": "v8.4.0",
            "project_id": web_project_id,
            "exclusion_rule": (
                "detection_confidence < export_confidence_threshold"
            ),
        },
    )

    repo.add_task(task)

    # 先確保任務已寫入資料庫，再要求 Cloud Run 啟動 Worker。觸發失敗時
    # 不能回滾上傳或刪除任務，讓人工執行或低頻 Scheduler 仍可補處理。
    job_operation_name = ""
    try:
        job_operation_name = trigger_task_worker(cfg)
    except Exception:  # noqa: BLE001 外部 API 失敗不可連帶讓任務建立失敗
        current_app.logger.exception(
            "LIFF 任務已建立，但 Cloud Run Job 觸發失敗：task_id=%s",
            task.id,
        )

    return (
        jsonify(
            {
                "ok": True,
                "message": "標註任務建立成功",
                "task_id": task.id,
                "task_status": task.status,
                "job_triggered": bool(job_operation_name),
                "status_url": url_for(
                    "liff.task_status",
                    task_id=task.id,
                    token=task.download_token,
                ),
                "user_id": web_user_id,
                "display_name": response_display_name,
                "prompt": prompt,
                "image_count": len(saved_images),
                "images": saved_images,
            }
        ),
        201,
    )

@bp.get("/tasks/<task_id>/status")
def task_status(task_id: str):
    """使用任務 token 查詢 LIFF 任務處理狀態。"""

    repo = get_repo()
    task = repo.get_task(task_id)

    if task is None:
        return jsonify({"ok": False, "message": "找不到標註任務"}), 404

    token = request.args.get("token", "")

    if not token or not secrets.compare_digest(token, task.download_token):
        return jsonify({"ok": False, "message": "查詢憑證無效"}), 403

    result = {
        "ok": True,
        "task_id": task.id,
        "task_status": task.status,
    }

    if task.status == "completed":
        if task.dataset_zip_path:
            result["message"] = "標註完成，可以下載 ZIP"
            result["download_url"] = url_for(
                "liff.download_task_dataset",
                task_id=task.id,
                token=task.download_token,
            )
        else:
            result["message"] = "標註完成，但沒有通過信心門檻的結果"
    elif task.status == "failed":
        result["message"] = "標註任務失敗"
        result["error_message"] = task.error_message or "處理任務時發生錯誤"
    elif task.status == "processing":
        result["message"] = "正在執行圖片標註"
    elif task.status == "retry_wait":
        result["message"] = "處理失敗，系統將自動重試"
    elif task.status == "deleting":
        result["message"] = "任務正在刪除；若長時間未完成，請重試刪除"
    else:
        result["message"] = "任務正在排隊等待處理"

    return jsonify(result)


@bp.post("/tasks/<task_id>/excluded")
def list_excluded_results(task_id: str):
    """驗證 LINE 身分後，以每頁 20 筆回傳低信心縮圖。"""
    payload = request.get_json(silent=True) or {}
    id_token = str(payload.get("id_token", "")).strip()
    profile, verification_error = _verify_liff_profile(id_token)
    if verification_error is not None:
        return verification_error

    task = get_repo().get_task(task_id)
    if task is None or task.line_user_id != profile["sub"]:
        return jsonify({"ok": False, "message": "找不到標註任務"}), 404

    try:
        page = max(1, int(payload.get("page", 1)))
    except (TypeError, ValueError):
        page = 1
    start = (page - 1) * _EXCLUDED_PAGE_SIZE
    selected = task.excluded_results[start:start + _EXCLUDED_PAGE_SIZE]
    items = [
        {
            "segment_id": str(item.get("segment_id", "")),
            "image_id": str(item.get("image_id", "")),
            "detection_confidence": float(
                item.get("detection_confidence", 0.0)
            ),
            "thumbnail_url": url_for(
                "liff.excluded_thumbnail",
                task_id=task.id,
                segment_id=str(item.get("segment_id", "")),
                token=task.download_token,
            ),
        }
        for item in selected
    ]
    return jsonify(
        {
            "ok": True,
            "page": page,
            "page_size": _EXCLUDED_PAGE_SIZE,
            "total": len(task.excluded_results),
            "has_more": start + len(selected) < len(task.excluded_results),
            "items": items,
        }
    )


@bp.get("/tasks/<task_id>/excluded/<segment_id>/thumbnail")
def excluded_thumbnail(task_id: str, segment_id: str):
    """使用任務隨機 token 串流後端產生的低信心 JPEG 縮圖。"""
    task = get_repo().get_task(task_id)
    if task is None:
        return jsonify({"ok": False, "message": "找不到標註任務"}), 404
    token = request.args.get("token", "")
    if not token or not secrets.compare_digest(token, task.download_token):
        return jsonify({"ok": False, "message": "縮圖憑證無效"}), 403
    item = next(
        (
            result for result in task.excluded_results
            if str(result.get("segment_id", "")) == segment_id
        ),
        None,
    )
    preview_path = str(item.get("preview_path", "")) if item else ""
    if not preview_path or not get_storage().exists(preview_path):
        return jsonify({"ok": False, "message": "縮圖不存在"}), 404
    return Response(
        stream_with_context(_stream_file(get_storage(), preview_path)),
        mimetype="image/jpeg",
        headers={"Cache-Control": "private, max-age=300"},
    )


@bp.get("/tasks/<task_id>/download")
def download_task_dataset(task_id: str):
    """使用任務的隨機 token 下載 YOLO ZIP。"""

    repo = get_repo()
    task = repo.get_task(task_id)

    if task is None:
        return jsonify({"ok": False, "message": "找不到標註任務"}), 404

    token = request.args.get("token", "")

    if not token or not secrets.compare_digest(token, task.download_token):
        return jsonify({"ok": False, "message": "下載憑證無效"}), 403

    if task.status != "completed":
        return (
            jsonify(
                {
                    "ok": False,
                    "message": "標註任務尚未完成",
                    "task_status": task.status,
                }
            ),
            409,
        )

    if not task.dataset_zip_path:
        return jsonify({"ok": False, "message": "任務沒有可下載的 ZIP"}), 404

    storage = get_storage()
    if not storage.exists(task.dataset_zip_path):
        return jsonify({"ok": False, "message": "ZIP 檔案不存在"}), 404

    download_name = f"seer_{task.id}_yolo.zip"
    try:
        signed_url = storage.generate_download_url(
            task.dataset_zip_path,
            download_name,
        )
    except (AttributeError, TransportError):
        current_app.logger.warning(
            "無法產生 GCS signed URL，改由應用程式串流：%s",
            task.dataset_zip_path,
            exc_info=True,
        )
        signed_url = None

    if signed_url is not None:
        return redirect(signed_url, code=302)

    return Response(
        stream_with_context(
            _stream_file(storage, task.dataset_zip_path)
        ),
        mimetype="application/zip",
        headers={
            "Content-Disposition": (
                f'attachment; filename="{download_name}"'
            ),
            "X-Accel-Buffering": "no",
        },
    )
