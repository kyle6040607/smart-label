"""把標好的遮罩 + 類別打包成可訓練的資料集（這是專案的最終產出）。

對應提案第 4 週「跑出成果數據」——那份成果數據就是這裡匯出的資料集。
支援三種下游常見格式，使用者匯出時自選：

  - coco : COCO instance segmentation（images/ + annotations.json）
  - yolo : YOLOv8-seg（images/ + labels/*.txt + data.yaml）
  - mask : 語意分割 mask PNG（images/ + masks/*.png + classes.txt）

只收已有最終類別（final_label）的片段——那才算「標好」的資料。
"""
from __future__ import annotations

import io
import json
import zipfile
from pathlib import Path

import cv2
import numpy as np

from app.models import ImageRecord, Segment
from app.repository import Repository
from app.utils import imread

FORMATS = ("coco", "yolo", "mask")


# ---------- 共用小工具 ----------
def _safe_stem(image: ImageRecord) -> str:
    """zip 內檔名前綴：加 image id 避免不同圖同檔名互相覆蓋。"""
    stem = Path(image.filename).stem or "image"
    return f"{image.id}_{stem}"


def _arcname(image: ImageRecord) -> str:
    suffix = Path(image.filename).suffix or ".png"
    return f"{_safe_stem(image)}{suffix}"


def _image_size(image: ImageRecord, segs: list[Segment]) -> tuple[int, int]:
    """回傳 (w, h)：優先用紀錄值，沒有就從遮罩推回。"""
    if image.width and image.height:
        return image.width, image.height
    m = imread(segs[0].mask_path, cv2.IMREAD_GRAYSCALE)
    h, w = m.shape[:2]
    return w, h


def _mask_to_polygons(mask: np.ndarray) -> list[list[float]]:
    """二值遮罩 → 多邊形輪廓，每個輪廓是攤平的 [x1,y1,x2,y2,...]。"""
    contours, _ = cv2.findContours(
        (mask > 0).astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
    )
    polys: list[list[float]] = []
    for c in contours:
        if cv2.contourArea(c) < 1:
            continue
        poly = c.reshape(-1, 2).astype(float)
        if len(poly) >= 3:
            polys.append(poly.flatten().tolist())
    return polys


def _add_image_file(z: zipfile.ZipFile, image: ImageRecord) -> str:
    """把原圖塞進 zip 的 images/ 下，回傳 zip 內檔名。"""
    arc = _arcname(image)
    try:
        z.writestr(f"images/{arc}", Path(image.path).read_bytes())
    except OSError:
        pass  # 原圖檔不在就略過，標註仍照常寫
    return arc


def _owned(repo: Repository, seg: Segment, owner_id: str | None) -> bool:
    """雙重驗證：片段與所屬照片的 owner 都要對得上，防止資料不一致外洩。"""
    if owner_id is None:
        return True
    if seg.owner_id != owner_id:
        return False
    image = repo.get_image(seg.image_id)
    return image is not None and image.owner_id == owner_id


# ---------- 對外入口 ----------
def build_dataset(repo: Repository, fmt: str, owner_id: str | None = None) -> bytes:
    """收齊已標好的片段，打包成指定格式的 zip，回傳 bytes。

    owner_id 給定時只收該使用者的片段；None（如 admin）收全體。
    """
    if fmt not in FORMATS:
        raise ValueError(f"未知格式：{fmt}（可用：{', '.join(FORMATS)}）")

    labeled = [
        s for s in repo.list_segments(owner_id=owner_id)
        if s.final_label and _owned(repo, s, owner_id)
    ]
    labels = sorted({s.final_label for s in labeled if s.final_label})
    by_image: dict[str, list[Segment]] = {}
    for s in labeled:
        by_image.setdefault(s.image_id, []).append(s)

    buf = io.BytesIO()
    with zipfile.ZipFile(buf, "w", zipfile.ZIP_DEFLATED) as z:
        writer = {"coco": _write_coco, "yolo": _write_yolo, "mask": _write_mask}[fmt]
        writer(z, repo, by_image, labels)
    return buf.getvalue()


# ---------- COCO instance segmentation ----------
def _write_coco(z, repo, by_image, labels):
    cat_id = {name: i + 1 for i, name in enumerate(labels)}
    coco = {
        "images": [],
        "annotations": [],
        "categories": [{"id": cat_id[n], "name": n} for n in labels],
    }
    ann_id = 1
    for img_idx, (image_id, segs) in enumerate(by_image.items(), start=1):
        image = repo.get_image(image_id)
        if not image:
            continue
        w, h = _image_size(image, segs)
        file_name = _add_image_file(z, image)
        coco["images"].append(
            {"id": img_idx, "file_name": file_name, "width": w, "height": h}
        )
        for s in segs:
            mask = imread(s.mask_path, cv2.IMREAD_GRAYSCALE)
            if mask is None:
                continue
            x, y, bw, bh = s.bbox
            coco["annotations"].append({
                "id": ann_id,
                "image_id": img_idx,
                "category_id": cat_id[s.final_label],
                "bbox": [int(x), int(y), int(bw), int(bh)],
                "area": int(s.area),
                "segmentation": _mask_to_polygons(mask),
                "iscrowd": 0,
            })
            ann_id += 1
    z.writestr("annotations.json", json.dumps(coco, ensure_ascii=False, indent=2))


# ---------- YOLOv8 segmentation ----------
def _write_yolo(z, repo, by_image, labels):
    cls_idx = {name: i for i, name in enumerate(labels)}
    for image_id, segs in by_image.items():
        image = repo.get_image(image_id)
        if not image:
            continue
        w, h = _image_size(image, segs)
        _add_image_file(z, image)
        lines: list[str] = []
        for s in segs:
            mask = imread(s.mask_path, cv2.IMREAD_GRAYSCALE)
            if mask is None:
                continue
            for poly in _mask_to_polygons(mask):
                coords = " ".join(
                    f"{(v / w if i % 2 == 0 else v / h):.6f}"
                    for i, v in enumerate(poly)
                )
                lines.append(f"{cls_idx[s.final_label]} {coords}")
        body = "\n".join(lines)
        z.writestr(f"labels/{_safe_stem(image)}.txt", body + ("\n" if body else ""))

    yaml = (
        "train: images\n"
        "val: images\n"
        f"nc: {len(labels)}\n"
        "names:\n"
        + "".join(f"  {i}: {n}\n" for i, n in enumerate(labels))
    )
    z.writestr("data.yaml", yaml)


# ---------- 語意分割 mask PNG ----------
def _write_mask(z, repo, by_image, labels):
    cls_idx = {name: i + 1 for i, name in enumerate(labels)}  # 0 保留給背景
    for image_id, segs in by_image.items():
        image = repo.get_image(image_id)
        if not image:
            continue
        w, h = _image_size(image, segs)
        _add_image_file(z, image)
        canvas = np.zeros((h, w), np.uint8)
        # 大塊先畫、小塊後畫蓋上去，避免大區域吃掉重疊的小物件
        for s in sorted(segs, key=lambda s: s.area, reverse=True):
            mask = imread(s.mask_path, cv2.IMREAD_GRAYSCALE)
            if mask is None:
                continue
            if mask.shape[:2] != (h, w):
                mask = cv2.resize(mask, (w, h), interpolation=cv2.INTER_NEAREST)
            canvas[mask > 0] = cls_idx[s.final_label]
        ok, png = cv2.imencode(".png", canvas)
        if ok:
            z.writestr(f"masks/{_safe_stem(image)}.png", png.tobytes())

    classes = "0\t__background__\n" + "".join(f"{cls_idx[n]}\t{n}\n" for n in labels)
    z.writestr("classes.txt", classes)

def main():
    print("=" * 50)
    print("      YOLO Dataset Exporter")
    print("=" * 50)

    # 建立 Repository
    repo = Repository(Path("data/db.json"))

    # 顯示資料資訊
    images = repo.list_images()
    segments = repo.list_segments()
    labeled = [s for s in segments if s.final_label]

    print(f"圖片數量        : {len(images)}")
    print(f"Segment 數量    : {len(segments)}")
    print(f"已完成標註數量   : {len(labeled)}")

    if not labeled:
        print("\n❌ 沒有可匯出的資料")
        return

    print("\n開始建立 YOLO 資料集...")

    # 固定匯出 YOLO
    zip_bytes = build_dataset(repo, "yolo")

    output = Path("yolo_dataset.zip")
    output.write_bytes(zip_bytes)

    print("\n✅ 匯出成功")
    print(f"檔案：{output.resolve()}")


if __name__ == "__main__":
    main()
    