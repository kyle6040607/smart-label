"""資料集匯出 API（專案的最終產出）。

把標好的「原圖 + 遮罩 + 類別」打包成下游可直接訓練的資料集 zip。
格式由 query 參數決定：?format=coco | yolo | mask。
"""
from __future__ import annotations

import io

from flask import Blueprint, abort, request, send_file

from app.routes import get_repo
from app.routes.auth import api_login_required, get_current_user, scope_owner_id
from app.services.exporter import FORMATS, build_dataset

bp = Blueprint("export", __name__, url_prefix="/api")
bp.before_request(api_login_required)


@bp.get("/export")
def export_dataset():
    """匯出資料集 zip。?format=coco（預設）| yolo | mask

    只包含自己標好的片段；admin 匯出全體資料。
    """
    fmt = (request.args.get("format") or "coco").lower()
    if fmt not in FORMATS:
        abort(400, f"未知格式：{fmt}（可用：{', '.join(FORMATS)}）")
    owner_id = scope_owner_id(get_current_user())
    data = build_dataset(get_repo(), fmt, owner_id=owner_id)
    return send_file(
        io.BytesIO(data),
        mimetype="application/zip",
        as_attachment=True,
        download_name=f"smart_label_dataset_{fmt}.zip",
    )
