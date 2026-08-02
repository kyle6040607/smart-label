import zipfile
from pathlib import Path

import cv2
import numpy as np
import pytest

from app.config import Config
from app.models import AnnotationTask, ImageRecord
from app.models import Segment
from app.repository import Repository
from app.services.pipeline import Pipeline
from app.services.task_processor import cleanup_task_attempt, process_task
from app.storage import LocalStorage, StorageService


def test_cleanup_task_attempt_removes_only_stale_attempt_artifacts(tmp_path):
    repo = Repository(tmp_path / "store.json")
    storage = StorageService(
        local=LocalStorage(
            data_dir=tmp_path / "data",
            upload_dir=tmp_path / "uploads",
            mask_dir=tmp_path / "masks",
        )
    )
    stale_mask = storage.save_bytes("masks/stale.png", b"mask")
    current_mask = storage.save_bytes("masks/current.png", b"mask")
    stale_preview = storage.save_bytes(
        "previews/tasks/task-1/attempts/stale/segment.jpg",
        b"preview",
    )
    current_preview = storage.save_bytes(
        "previews/tasks/task-1/attempts/current/segment.jpg",
        b"preview",
    )
    stale_zip = storage.save_bytes(
        "datasets/task-1/attempts/stale/dataset_v1.zip",
        b"zip",
    )
    current_zip = storage.save_bytes(
        "datasets/task-1/attempts/current/dataset_v1.zip",
        b"zip",
    )
    stale_segment = repo.add_segment(
        Segment(
            mask_path=stale_mask,
            annotation_task_id="task-1",
            task_attempt_token="stale",
        )
    )
    current_segment = repo.add_segment(
        Segment(
            mask_path=current_mask,
            annotation_task_id="task-1",
            task_attempt_token="current",
        )
    )

    cleanup_task_attempt(repo, storage, "task-1", "stale")

    remaining_ids = {segment.id for segment in repo.list_segments()}
    assert stale_segment.id not in remaining_ids
    assert current_segment.id in remaining_ids
    assert not storage.exists(stale_mask)
    assert storage.exists(current_mask)
    assert not storage.exists(stale_preview)
    assert storage.exists(current_preview)
    assert not storage.exists(stale_zip)
    assert storage.exists(current_zip)


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
    cfg.use_gcs = False
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
        cfg.data_dir / "tasks" / task.id / "dataset_v1.zip"
    )

    assert updated_task.status == "completed"
    assert updated_task.error_message == ""
    assert updated_task.dataset_zip_path == str(zip_path)
    assert updated_task.dataset_version == 1
    assert updated_task.processed_image_ids == [image.id]
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
            **kwargs,
        ):
            del image, prompt, kwargs
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
        / "dataset_v1.zip"
    ).exists()


def test_process_task_excludes_low_confidence_and_creates_thumbnail(tmp_path):
    class LowConfidencePipeline:
        def segment_text(self, image, prompt, **kwargs):
            del prompt
            segment = Segment(
                image_id=image.id,
                bbox=(0, 0, 20, 20),
                area=400,
                predicted_label="cat",
                detection_confidence=0.49,
                annotation_task_id=kwargs["annotation_task_id"],
                task_attempt_token=kwargs["task_attempt_token"],
            )
            repo.add_segment(segment)
            return [segment]

    image_path = tmp_path / "cat.png"
    pixels = np.zeros((30, 30, 3), dtype=np.uint8)
    assert cv2.imwrite(str(image_path), pixels)
    repo = Repository(tmp_path / "store.json")
    image = repo.add_image(
        ImageRecord(
            filename="cat.png",
            path=str(image_path),
            width=30,
            height=30,
        )
    )
    task = repo.add_task(
        AnnotationTask(
            prompt="cat",
            image_ids=[image.id],
            status="processing",
            settings_snapshot={
                "detection_confidence_threshold": 0.15,
                "export_confidence_threshold": 0.5,
                "yolo_imgsz": 640,
            },
        )
    )
    result = process_task(
        repo,
        LowConfidencePipeline(),
        task,
        tmp_path / "tasks",
    )
    assert result.status == "completed"
    assert result.exported_count == 0
    assert result.excluded_count == 1
    assert result.dataset_zip_path == ""
    assert result.completion_reason == "all_segments_below_confidence"
    assert Path(result.excluded_results[0]["preview_path"]).exists()


def test_process_task_rejects_local_fallback_when_gcs_is_enabled(
    tmp_path,
):
    class GcsPipelineWithoutStorage:
        class PipelineConfig:
            use_gcs = True

        config = PipelineConfig()

        def segment_text(self, image, prompt):
            pytest.fail("storage 驗證應在處理圖片前發生")

    repo = Repository(tmp_path / "store.json")
    task = repo.add_task(
        AnnotationTask(
            prompt="cat",
            image_ids=["image-1"],
            status="processing",
        )
    )

    with pytest.raises(RuntimeError, match="USE_GCS=1"):
        process_task(
            repo,
            GcsPipelineWithoutStorage(),
            task,
            tmp_path / "tasks",
        )


def test_process_task_rejects_explicit_local_storage_in_gcs_mode(
    tmp_path,
):
    class GcsPipeline:
        class PipelineConfig:
            use_gcs = True

        config = PipelineConfig()

    class LocalStorageStub:
        backend_name = "local"

    repo = Repository(tmp_path / "store.json")
    task = repo.add_task(
        AnnotationTask(
            prompt="cat",
            image_ids=["image-1"],
            status="processing",
        )
    )

    with pytest.raises(RuntimeError, match="GCS storage"):
        process_task(
            repo,
            GcsPipeline(),
            task,
            tmp_path / "tasks",
            storage=LocalStorageStub(),
        )


def test_process_task_only_processes_new_images(
    tmp_path,
    monkeypatch,
):
    class RecordingPipeline:
        def __init__(self):
            self.image_ids = []

        def segment_text(self, image, prompt, **kwargs):
            del prompt
            self.image_ids.append(image.id)
            segment = Segment(
                image_id=image.id,
                predicted_label="cat",
                detection_confidence=0.9,
                annotation_task_id=kwargs.get("annotation_task_id", ""),
                task_attempt_token=kwargs.get("task_attempt_token", ""),
            )
            repo.add_segment(segment)
            return [segment]

    dataset_image_ids = []

    def fake_write_dataset(
        repo,
        fmt,
        output,
        image_ids=None,
        storage=None,
        **kwargs,
    ):
        del repo, fmt, storage, kwargs
        dataset_image_ids.append(set(image_ids or set()))
        output.write(
            f"version-{len(dataset_image_ids)}".encode()
        )

    monkeypatch.setattr(
        "app.services.task_processor.write_dataset",
        fake_write_dataset,
    )

    repo = Repository(tmp_path / "store.json")
    pipeline = RecordingPipeline()

    first_image = repo.add_image(
        ImageRecord(filename="first.jpg")
    )
    second_image = repo.add_image(
        ImageRecord(filename="second.jpg")
    )

    task = repo.add_task(
        AnnotationTask(
            prompt="cat",
            image_ids=[first_image.id],
            status="processing",
        )
    )

    output_dir = tmp_path / "tasks"

    first_result = process_task(
        repo,
        pipeline,
        task,
        output_dir,
    )

    first_zip = (
        output_dir
        / task.id
        / "dataset_v1.zip"
    )

    assert pipeline.image_ids == [first_image.id]
    assert first_result.dataset_version == 1
    assert first_result.processed_image_ids
