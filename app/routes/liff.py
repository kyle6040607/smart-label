import os
from flask import Blueprint, render_template

bp = Blueprint(
    "liff",
    __name__,
    url_prefix="/liff"
)

@bp.get("/upload")
def upload_page():
    """顯示 LIFF 圖片與 Prompt 上傳頁面。"""

    liff_id = os.getenv("LIFF_ID", "").strip()

    return render_template(
        "liff/upload.html",
        liff_id=liff_id,             
    )