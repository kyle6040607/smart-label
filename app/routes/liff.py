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
- POST /liff/tasks/<task_id>/images
  預計用於對已完成任務追加圖片並重新排隊。

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
import os
import secrets
from pathlib import Path

from PIL import Image, UnidentifiedImageError
from flask import (
    Blueprint,
    current_app,
    jsonify,
    render_template,
    request,
    send_file,
    url_for,
)

from werkzeug.utils import secure_filename

from app.models import AnnotationTask, ImageRecord
from app.routes import get_config, get_repo
from app.services import line_login


bp = Blueprint(
    "liff",
    __name__,
    url_prefix="/liff",
)


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
        "can_add_images": task.status == "completed",
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

    return render_template("liff/upload.html", liff_id=cfg.liff_id)


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

    tasks = repo.list_tasks_by_line_user_id(line_user_id)

    return jsonify(
        {
            "ok": True,
            "task_count": len(tasks),
            "tasks": [_task_summary(task) for task in tasks],
        }
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

            with Image.open(image.stream) as uploaded_image:
                uploaded_image.verify()

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
        validated_images.append((image, extension))

    display_name = profile.get("name", "")

    # 查詢這位 LINE 使用者是否已綁定 Web 帳號
    user = repo.get_user_by_line_id(line_user_id)

    web_user_id = user.id if user is not None else ""

    response_display_name = (
        user.display_name or user.username if user is not None else display_name
    )

    upload_dir = cfg.upload_dir

    upload_dir.mkdir(parents=True, exist_ok=True)

    saved_images = []

    for image, extension in validated_images:
        safe_original_filename = secure_filename(image.filename)

        if not safe_original_filename:
            safe_original_filename = f"upload.{extension}"

        image_record = ImageRecord(filename=safe_original_filename)

        saved_filename = f"{image_record.id}.{extension}"

        save_path = upload_dir / saved_filename

        image.save(save_path)

        with Image.open(save_path) as saved_image:
            image_record.width = saved_image.width
            image_record.height = saved_image.height

        image_record.path = str(save_path)

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
    )

    repo.add_task(task)

    return (
        jsonify(
            {
                "ok": True,
                "message": "標註任務建立成功",
                "task_id": task.id,
                "task_status": task.status,
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
        result["message"] = "標註完成，可以下載 ZIP"
        result["download_url"] = url_for(
            "liff.download_task_dataset",
            task_id=task.id,
            token=task.download_token,
        )
    elif task.status == "failed":
        result["message"] = "標註任務失敗"
        result["error_message"] = task.error_message or "處理任務時發生錯誤"
    elif task.status == "processing":
        result["message"] = "正在執行圖片標註"
    else:
        result["message"] = "任務正在排隊等待處理"

    return jsonify(result)


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

    zip_path = Path(task.dataset_zip_path)

    if not zip_path.is_file():
        return jsonify({"ok": False, "message": "ZIP 檔案不存在"}), 404

    return send_file(
        zip_path,
        mimetype="application/zip",
        as_attachment=True,
        download_name=f"smart_label_{task.id}_yolo.zip",
    )
