"""資料集匯出測試：標好的片段能打包成三種下游格式。"""
from __future__ import annotations

import io
import json
import zipfile

import cv2
import numpy as np

from app.config import Config
from app.services.exporter import build_dataset
from app.services.pipeline import Pipeline
from app.models import ImageRecord
from app.repository import Repository


def _make_image(path, color):
    img = np.full((120, 120, 3), 30, np.uint8)
    cv2.rectangle(img, (30, 30), (90, 90), color, -1)
    cv2.imwrite(str(path), img)


def _setup(tmp_path) -> Repository:
    """造一張圖、切塊、標一個類別 → 得到一個有 final_label 的片段。"""
    cfg = Config(
        base_dir=tmp_path, data_dir=tmp_path, upload_dir=tmp_path / "up",
        mask_dir=tmp_path / "mask", db_file=tmp_path / "store.json",
    )
    cfg.ensure_dirs()
    # 不用真模型跑測試（不受 .env 的 USE_REAL_SAM 影響）
    cfg.use_real_sam = False
    cfg.use_real_embedding = False
    repo = Repository(cfg.db_file)
    pipe = Pipeline(cfg, repo)
    p = cfg.upload_dir / "a.png"
    _make_image(p, (200, 50, 50))
    img = repo.add_image(ImageRecord(filename="a.png", path=str(p), width=120, height=120))
    segs = pipe.segment_image(img)
    pipe.add_example_from_segment(segs[0], "cat")
    return repo


def _zip(data: bytes) -> zipfile.ZipFile:
    return zipfile.ZipFile(io.BytesIO(data))


def test_export_coco(tmp_path):
    z = _zip(build_dataset(_setup(tmp_path), "coco"))
    names = z.namelist()
    assert "annotations.json" in names
    assert any(n.startswith("images/") for n in names)
    coco = json.loads(z.read("annotations.json"))
    assert len(coco["images"]) == 1
    assert len(coco["annotations"]) >= 1
    assert coco["categories"][0]["name"] == "cat"
    assert coco["annotations"][0]["segmentation"]  # 有多邊形


def test_export_yolo(tmp_path):
    z = _zip(build_dataset(_setup(tmp_path), "yolo"))
    names = z.namelist()
    assert "data.yaml" in names
    label_files = [n for n in names if n.startswith("labels/") and n.endswith(".txt")]
    assert label_files
    txt = z.read(label_files[0]).decode().strip()
    assert txt.split()[0] == "0"  # class index
    # 座標都正規化在 0~1
    coords = [float(v) for v in txt.split()[1:]]
    assert coords and all(0.0 <= c <= 1.0 for c in coords)


def test_export_mask(tmp_path):
    z = _zip(build_dataset(_setup(tmp_path), "mask"))
    names = z.namelist()
    assert "classes.txt" in names
    mask_files = [n for n in names if n.startswith("masks/")]
    assert mask_files
    png = np.frombuffer(z.read(mask_files[0]), np.uint8)
    m = cv2.imdecode(png, cv2.IMREAD_GRAYSCALE)
    assert m.max() == 1  # 單一類別 → 像素值 1（0 是背景）


def test_export_empty(tmp_path):
    """沒有任何標好的片段也要能匯出（空資料集，不爆炸）。"""
    cfg = Config(
        base_dir=tmp_path, data_dir=tmp_path, upload_dir=tmp_path / "up",
        mask_dir=tmp_path / "mask", db_file=tmp_path / "store.json",
    )
    cfg.ensure_dirs()
    repo = Repository(cfg.db_file)
    coco = json.loads(_zip(build_dataset(repo, "coco")).read("annotations.json"))
    assert coco["images"] == [] and coco["annotations"] == []
