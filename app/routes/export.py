"""資料集匯出 API（專案的最終產出）。

把標好的「原圖 + 遮罩 + 類別」打包成下游可直接訓練的資料集 zip。
格式由 query 參數決定：?format=coco | yolo | mask。
"""
from __future__ import annotations

import io

from flask import Blueprint, abort, jsonify, request, send_file

from app.routes import can_access_image, get_current_project_id, get_repo, get_storage
from app.services.exporter import FORMATS, build_dataset

bp = Blueprint("export", __name__, url_prefix="/api")


def _get_project_filename_prefix(repo, project_id: str | None) -> str:
    """取得清洗後的專案檔名前綴。"""
    if project_id:
        proj = repo.get_project(project_id)
        if proj and getattr(proj, "name", None):
            clean_name = "".join(
                c if c.isalnum() or c in ("-", "_") else "_"
                for c in proj.name.strip()
            )
            if clean_name:
                return clean_name
    return "seer"


@bp.get("/export/preview")
def export_preview():
    """回傳資料集匯出預覽統計數據與目錄結構。"""
    repo = get_repo()
    project_id = request.args.get("project_id") or get_current_project_id()
    images = [img for img in repo.list_images(project_id=project_id) if can_access_image(img)]

    total_images = len(images)
    annotated_images = 0
    total_segments = 0
    label_counts: dict[str, int] = {}

    for img in images:
        segs = repo.list_segments(img.id)
        valid_segs = [s for s in segs if (s.final_label or s.predicted_label)]
        if valid_segs:
            annotated_images += 1
            total_segments += len(valid_segs)
            for s in valid_segs:
                lbl = s.final_label or s.predicted_label
                if lbl:
                    label_counts[lbl] = label_counts.get(lbl, 0) + 1

    prefix = _get_project_filename_prefix(repo, project_id)

    return jsonify({
        "total_images": total_images,
        "annotated_images": annotated_images,
        "total_segments": total_segments,
        "num_labels": len(label_counts),
        "label_counts": label_counts,
        "format_trees": {
            "coco": [
                f"{prefix}_COCO.zip",
                "├── images/              (包含了標記的原始圖檔)",
                "└── annotations.json     (包含 COCO 多邊形點與類別標籤的 JSON 檔)"
            ],
            "yolo": [
                f"{prefix}_YOLO.zip",
                "├── images/              (包含原始圖片檔)",
                "├── labels/              (包含了 txt 格式的多邊形邊界點)",
                "└── data.yaml            (YOLOv8 / YOLO11 訓練類別與路徑設定檔)"
            ],
            "mask": [
                f"{prefix}_MASK.zip",
                "├── images/              (包含原始圖片檔)",
                "├── masks/               (包含 PNG 二值化語意分割 Mask)",
                "└── classes.txt          (類別名稱對照表)"
            ]
        }
    })


@bp.get("/export")
def export_dataset():
    """匯出資料集 zip。?format=coco（預設）| yolo | mask"""
    fmt = (request.args.get("format") or "coco").lower()
    if fmt not in FORMATS:
        abort(400, f"未知格式：{fmt}（可用：{', '.join(FORMATS)}）")
    repo = get_repo()
    project_id = request.args.get("project_id") or get_current_project_id()
    image_ids = {
        image.id for image in repo.list_images(project_id=project_id) if can_access_image(image)
    }
    data = build_dataset(
        repo,
        fmt,
        image_ids=image_ids,
        storage=get_storage(),
    )
    prefix = _get_project_filename_prefix(repo, project_id)
    download_name = f"{prefix}_{fmt.upper()}.zip"

    return send_file(
        io.BytesIO(data),
        mimetype="application/zip",
        as_attachment=True,
        download_name=download_name,
    )
