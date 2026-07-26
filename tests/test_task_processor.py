import zipfile

import cv2
import numpy as np
import pytest

from app.config import Config
from app.models import AnnotationTask, ImageRecord
from app.repository import Repository
from app.services.pipeline import Pipeline
from app.services.task_processor import process_task


def test_process_task_creates_yolo_zip(tmp_path):
    cfg = Config(
        base_dir=tmp_path,
        data_dir=tmp_path / "data",
        upload_dir=tmp_path / "uploads",
        mask_dir=tmp_path / "masks",
        db_file=tmp_path / "store.json",
    )
    cfg.use_real_sam = False
    cfg.use_real_embedding = False
    cfg.ensure_dirs()

    repo = Repository(cfg.db_file)
    pipeline = Pipeline(cfg, repo)

    image_path = cfg.upload_dir / "cat.png"
    pixels = np.zeros(
        (120, 120, 3),
        dtype=np.uint8,
    )
    cv2.rectangle(
        pixels,
        (30, 30),
        (90, 90),
        (255, 255, 255),
        -1,
    )
    assert cv2.imwrite(
        str(image_path),
        pixels,
    )

    image = repo.add_image(
        ImageRecord(
            filename="cat.png",
            path=str(image_path),
            width=120,
            height=120,
        )
    )

    task = repo.add_task(
        AnnotationTask(
            prompt="cat",
            image_ids=[image.id],
        )
    )
    task.status = "processing"
    repo.update_task(task)

    updated_task = process_task(
        repo,
        pipeline,
        task,
        cfg.data_dir / "tasks",
    )

    zip_path = (
        cfg.data_dir/ "tasks"/ task.id/ "dataset.zip"
    )

    assert updated_task.status == "completed"
    assert updated_task.error_message == ""
    assert updated_task.dataset_zip_path == str(zip_path)
    assert zip_path.exists()

    saved_task = repo.get_task(task.id)
    assert saved_task is not None
    assert saved_task.status == "completed"

    with zipfile.ZipFile(zip_path) as dataset:
        names = dataset.namelist()

    assert "data.yaml" in names
    assert any(
        name.startswith("images/")
        for name in names
    )
    assert any(
        name.startswith("labels/")
        for name in names
    )

def test_process_task_rejects_empty_detection(tmp_path):
    class EmptyPipeline:
        def segment_text(
            self,
            image,
            prompt,
        ):
            return []

    repo = Repository(
        tmp_path / "store.json"
    )
    image = repo.add_image(
        ImageRecord(
            filename="empty.jpg",
        )
    )
    task = repo.add_task(
        AnnotationTask(
            prompt="fried chicken",
            image_ids=[image.id],
            status="processing",
        )
    )

    with pytest.raises(
        ValueError,
        match="找不到符合標註內容的物件",
    ):
        process_task(
            repo,
            EmptyPipeline(),
            task,
            tmp_path / "tasks",
        )

    assert not (
        tmp_path
        / "tasks"
        / task.id
        / "dataset.zip"
    ).exists()