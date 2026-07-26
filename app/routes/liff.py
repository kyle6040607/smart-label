import os
from pathlib import Path

from PIL import Image, UnidentifiedImageError
from flask import (
    Blueprint,
    current_app,
    jsonify,
    render_template,
    request,
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


@bp.get("/upload")
def upload_page():
    """顯示 LIFF 圖片與 Prompt 上傳頁面。"""

    cfg = get_config()

    return render_template(
        "liff/upload.html",
        liff_id=cfg.liff_id,
    )


@bp.post("/upload")
def upload_task():
    """驗證 LINE 身分並儲存 LIFF 上傳的圖片與 Prompt。"""

    prompt = request.form.get("prompt", "").strip()
    id_token = request.form.get("id_token", "").strip()

    images = [
        image
        for image in request.files.getlist("images")
        if image and image.filename
    ]

    if not images:
        return jsonify({
            "ok": False,
            "message": "請至少選擇一張圖片",
        }), 400

    if not prompt:
        return jsonify({
            "ok": False,
            "message": "請輸入標註內容",
        }), 400

    if not id_token:
        return jsonify({
            "ok": False,
            "message": "缺少 LINE ID Token",
        }), 401

    cfg = get_config()
    repo = get_repo()

    validated_images = []

    for image in images:
        extension = (
            Path(image.filename)
            .suffix
            .lower()
            .lstrip(".")
        )

        if not extension or extension not in cfg.allowed_ext:
            return jsonify({
                "ok": False,
                "message": (
                    f"{image.filename} 格式不支援，"
                    f"允許格式：{', '.join(cfg.allowed_ext)}"
                ),
            }), 400
        try:
            image.stream.seek(0)

            with Image.open(image.stream) as uploaded_image:
                uploaded_image.verify()

        except (UnidentifiedImageError, OSError):
            image.stream.seek(0)

            return jsonify({
                "ok": False,
                "message": f"{image.filename} 不是有效的圖片檔案",
            }), 400

        image.stream.seek(0)
        validated_images.append((image, extension))

    if not cfg.line_login_channel_id:
        return jsonify({
            "ok": False,
            "message": "伺服器尚未設定 LINE_LOGIN_CHANNEL_ID",
        }), 500

    current_app.logger.info(
        "LIFF Token 驗證使用的 Channel ID：%s",
        cfg.line_login_channel_id,
    )

    # 驗證 LIFF 傳來的 ID Token
    try:
        profile = line_login.verify_id_token(
            id_token,
            cfg.line_login_channel_id,
            None,
        )
    except Exception as error:
        error_message = str(error)

        current_app.logger.exception(
            "LIFF ID Token 驗證失敗：%s",
            error_message,
        )

        if "IdToken expired" in error_message:
            return jsonify({
                "ok": False,
                "code": "LINE_ID_TOKEN_EXPIRED",
                "message": "LINE 登入資料已過期，正在重新登入",
            }), 401

        return jsonify({
            "ok": False,
            "code": "LINE_ID_TOKEN_INVALID",
            "message": "LINE 身分驗證失敗，請重新登入",
        }), 401

    # 驗證結果中的 sub 才是可信任的 LINE 使用者 ID
    line_user_id = profile.get("sub")

    if not line_user_id:
        return jsonify({
            "ok": False,
            "message": "LINE 未提供使用者識別碼",
        }), 401

    display_name = profile.get("name", "")
    avatar_url = profile.get("picture", "")

    # 查詢這位 LINE 使用者是否已綁定 Web 帳號
    user = repo.get_user_by_line_id(line_user_id)

    if user is not None:
        profile_changed = False

        if display_name and user.display_name != display_name:
            user.display_name = display_name
            profile_changed = True

        if avatar_url and user.avatar_url != avatar_url:
            user.avatar_url = avatar_url
            profile_changed = True

        if profile_changed:
            repo.update_user(user)

    web_user_id = user.id if user is not None else ""

    response_display_name = (
        user.display_name or user.username
        if user is not None
        else display_name
    )

    upload_dir = cfg.upload_dir

    upload_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    saved_images = []

    for image, extension in validated_images:
        safe_original_filename = secure_filename(
            image.filename
        )

        if not safe_original_filename:
            safe_original_filename = (
                f"upload.{extension}"
            )

        image_record = ImageRecord(
            filename=safe_original_filename,
        )

        saved_filename = (
            f"{image_record.id}.{extension}"
        )

        save_path = upload_dir / saved_filename

        image.save(save_path)

        with Image.open(save_path) as saved_image:
            image_record.width = saved_image.width
            image_record.height = saved_image.height

        image_record.path = str(save_path)

        repo.add_image(image_record)

        saved_images.append({
            "image_id": image_record.id,
            "original_filename": image.filename,
            "saved_filename": saved_filename,
            "width": image_record.width,
            "height": image_record.height,
        })

    task = AnnotationTask(
        user_id=web_user_id,
        line_user_id=line_user_id,
        prompt=prompt,
        image_ids=[image_info["image_id"] for image_info in saved_images],
    )

    repo.add_task(task)

    return jsonify({
        "ok": True,
        "message": "標註任務建立成功",
        "task_id": task.id,
        "task_status": task.status,
        "user_id": web_user_id,
        "display_name": response_display_name,
        "prompt": prompt,
        "image_count": len(saved_images),
        "images": saved_images,
    }), 201
