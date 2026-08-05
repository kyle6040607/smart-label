"""YOLOv26x-seg 訓練服務與多租戶資料隔離測試。"""
from __future__ import annotations

import datetime
import os
from pathlib import Path
from unittest.mock import MagicMock, patch

import cv2
import numpy as np
import pytest

from app.config import Config
from app.models import AnnotationTask, ImageRecord, Segment
from app.ml.yolov26x_seg import prepare_yolo_dataset, train_yolov26x_seg
from app.repository import Repository


def _make_dummy_image_file(path: Path) -> None:
    img = np.full((100, 100, 3), 128, dtype=np.uint8)
    cv2.imwrite(str(path), img)


def _make_dummy_mask_file(path: Path) -> None:
    mask = np.zeros((100, 100), dtype=np.uint8)
    cv2.rectangle(mask, (20, 20), (80, 80), 255, -1)
    cv2.imwrite(str(path), mask)


def test_prepare_yolo_dataset_isolation(tmp_path):
    repo = Repository(tmp_path / "store.json")

    # 建立 User A / Project A 的圖片與遮罩
    img_a_path = tmp_path / "img_a.png"
    mask_a_path = tmp_path / "mask_a.png"
    _make_dummy_image_file(img_a_path)
    _make_dummy_mask_file(mask_a_path)

    img_a = repo.add_image(
        ImageRecord(
            owner_id="user_a",
            project_id="proj_a",
            filename="img_a.png",
            path=str(img_a_path),
            width=100,
            height=100,
        )
    )
    repo.add_segment(
        Segment(
            image_id=img_a.id,
            mask_path=str(mask_a_path),
            human_label="apple",
        )
    )

    # 建立 User B / Project B 的圖片與遮罩
    img_b_path = tmp_path / "img_b.png"
    mask_b_path = tmp_path / "mask_b.png"
    _make_dummy_image_file(img_b_path)
    _make_dummy_mask_file(mask_b_path)

    img_b = repo.add_image(
        ImageRecord(
            owner_id="user_b",
            project_id="proj_b",
            filename="img_b.png",
            path=str(img_b_path),
            width=100,
            height=100,
        )
    )
    repo.add_segment(
        Segment(
            image_id=img_b.id,
            mask_path=str(mask_b_path),
            human_label="banana",
        )
    )

    # 測試 prepare_yolo_dataset 只針對 user_a 和 proj_a
    work_dir = tmp_path / "work_a"
    yaml_path, labels = prepare_yolo_dataset(
        repo=repo,
        storage=None,
        user_id="user_a",
        project_id="proj_a",
        work_dir=work_dir,
    )

    assert yaml_path.is_file()
    assert labels == ["apple"]

    yaml_text = yaml_path.read_text(encoding="utf-8")
    assert "nc: 1" in yaml_text
    assert "apple" in yaml_text
    assert "banana" not in yaml_text

    # 檢查 labels/ 目錄中僅有 img_a 的標註
    txt_files = list((work_dir / "labels").glob("*.txt"))
    assert len(txt_files) == 1
    assert f"{img_a.id}_img_a.txt" in txt_files[0].name


def test_train_yolov26x_seg_timestamp_and_cleanup(tmp_path):
    repo = Repository(tmp_path / "store.json")

    img_path = tmp_path / "test.jpg"
    mask_path = tmp_path / "test_mask.png"
    _make_dummy_image_file(img_path)
    _make_dummy_mask_file(mask_path)

    img = repo.add_image(
        ImageRecord(
            owner_id="usr1",
            project_id="prj1",
            filename="test.jpg",
            path=str(img_path),
            width=100,
            height=100,
        )
    )
    repo.add_segment(
        Segment(
            image_id=img.id,
            mask_path=str(mask_path),
            human_label="cat",
        )
    )

    task = repo.add_task(
        AnnotationTask(
            user_id="usr1",
            project_id="prj1",
            prompt="yolo test",
        )
    )

    recorded_work_dirs = []

    # Mock YOLO.train 以免執行真正的重型模型訓練
    class MockYOLO:
        def __init__(self, model_path):
            pass

        def add_callback(self, event, func):
            pass

        def train(self, **kwargs):
            project_dir = Path(kwargs["project"])
            recorded_work_dirs.append(project_dir.parent)
            save_dir = project_dir / "exp"
            weights_dir = save_dir / "weights"
            weights_dir.mkdir(parents=True, exist_ok=True)
            fake_best = weights_dir / "best.pt"
            fake_best.write_bytes(b"dummy_weights")
            res = MagicMock()
            res.save_dir = str(save_dir)
            return res

    with patch("ultralytics.YOLO", MockYOLO):
        updated_task = train_yolov26x_seg(
            repo=repo,
            storage=None,
            task=task,
            epochs=1,
            base_model_path=str(img_path),  # 用存在檔假裝權重
        )

    assert updated_task.status == "completed"
    assert "yolo26x_seg_" in updated_task.best_model_path
    assert updated_task.best_model_path.endswith(".pt")

    # 檢查檔名包含日期時間 YYYYMMDD
    filename = Path(updated_task.best_model_path).name
    today_str = datetime.datetime.now().strftime("%Y%m%d")
    assert today_str in filename

    # 驗證工作目錄被 100% 清理自動刪除
    assert len(recorded_work_dirs) == 1
    work_dir = recorded_work_dirs[0]
    assert not work_dir.exists(), f"暫存工作目錄未被刪除: {work_dir}"
