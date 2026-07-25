import os
from pathlib import Path
from uuid import uuid4

from flask import (
    Blueprint,
    current_app,
    jsonify,
    render_template,
    request,
)
from werkzeug.utils import secure_filename


bp = Blueprint(
    "liff",
    __name__,
    url_prefix="/liff",
)


@bp.get("/upload")
def upload_page():
    """顯示 LIFF 圖片與 Prompt 上傳頁面。"""

    liff_id = os.getenv("LIFF_ID", "").strip()

    return render_template(
        "liff/upload.html",
        liff_id=liff_id,
    )


@bp.post("/upload")
def upload_task():
    """接收並儲存 LIFF 上傳的圖片與 Prompt。"""

    prompt = request.form.get("prompt", "").strip()

    # 排除沒有實際檔名的空檔案欄位
    images = [
        image
        for image in request.files.getlist("images")
        if image and image.filename
    ]

    # 先檢查圖片
    if not images:
        return jsonify({
            "ok": False,
            "message": "請至少選擇一張圖片",
        }), 400

    # 再檢查 Prompt
    if not prompt:
        return jsonify({
            "ok": False,
            "message": "請輸入標註內容",
        }), 400

    cfg = current_app.smart_config
    upload_dir = cfg.upload_dir

    # 正常情況 create_app() 已經會建立，
    # 這裡再確認一次可以避免資料夾被手動刪除後上傳失敗
    upload_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    saved_images = []

    for image in images:
        original_filename = secure_filename(image.filename)

        extension = (
            Path(original_filename)
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

        saved_filename = f"{uuid4().hex}.{extension}"
        save_path = upload_dir / saved_filename

        image.save(save_path)

        saved_images.append({
            "original_filename": image.filename,
            "saved_filename": saved_filename,
        })

    if not saved_images:
        return jsonify({
            "ok": False,
            "message": "沒有可儲存的圖片",
        }), 400

    return jsonify({
        "ok": True,
        "message": "圖片儲存成功",
        "prompt": prompt,
        "image_count": len(saved_images),
        "images": saved_images,
    }), 201