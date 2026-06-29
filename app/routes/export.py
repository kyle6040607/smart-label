"""資料集匯出 API（專案的最終產出）。

把標好的「原圖 + 遮罩 + 類別」打包成下游可直接訓練的資料集 zip。
格式由 query 參數決定：?format=coco | yolo | mask。
"""
from __future__ import annotations

import io

from flask import Blueprint, abort, request, send_file

from app.routes import get_repo
from app.services.exporter import FORMATS, build_dataset

bp = Blueprint("export", __name__, url_prefix="/api")


@bp.get("/export")
def export_dataset():
    """匯出資料集 zip。?format=coco（預設）| yolo | mask"""
    fmt = (request.args.get("format") or "coco").lower()
    if fmt not in FORMATS:
        abort(400, f"未知格式：{fmt}（可用：{', '.join(FORMATS)}）")
    data = build_dataset(get_repo(), fmt)
    return send_file(
        io.BytesIO(data),
        mimetype="application/zip",
        as_attachment=True,
        download_name=f"smart_label_dataset_{fmt}.zip",
    )
