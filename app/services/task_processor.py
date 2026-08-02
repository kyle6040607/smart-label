"""執行單一 LIFF 任務，並以 attempt fencing 發布資料集結果。"""

from __future__ import annotations

import io
from pathlib import Path
from typing import Callable

from PIL import Image

from app.models import AnnotationTask, Segment
from app.repository import Repository
from app.services.exporter import write_dataset
from app.services.pipeline import Pipeline
from app.storage import LocalStorage, StorageService


class TaskLeaseLostError(RuntimeError):
    """目前 Worker 的 claim token 已失效，不得再發布結果。"""


def _remove_segments(
    repo: Repository,
    storage: StorageService,
    segments: list[Segment],
) -> None:
    if not segments:
        return
    mask_paths = repo.delete_segments_batch([segment.id for segment in segments])
    for mask_path in mask_paths:
        storage.delete(mask_path)


def cleanup_task_attempt(
    repo: Repository,
    storage: StorageService,
    task_id: str,
    attempt_token: str,
) -> None:
    """只清掉指定失敗 attempt 的 Segment，不碰 Web 或其他 attempt。"""
    segments = [
        segment
        for segment in repo.list_segments()
        if segment.annotation_task_id == task_id
        and segment.task_attempt_token == attempt_token
    ]
    _remove_segments(repo, storage, segments)


def _save_excluded_preview(
    storage: StorageService,
    task: AnnotationTask,
    segment: Segment,
    attempt_token: str,
    image,
) -> str:
    if image is None:
        raise FileNotFoundError(segment.image_id)
    image_bytes = storage.read_bytes(image.path)
    with Image.open(io.BytesIO(image_bytes)) as source:
        source.load()
        x, y, width, height = segment.bbox
        left = max(0, int(x))
        top = max(0, int(y))
        right = min(source.width, left + max(1, int(width)))
        bottom = min(source.height, top + max(1, int(height)))
        crop = source.convert("RGB").crop((left, top, right, bottom))
        crop.thumbnail((512, 512))
        with io.BytesIO() as output:
            crop.save(output, format="JPEG", quality=80, optimize=True)
            preview_bytes = output.getvalue()
    safe_attempt = attempt_token or "legacy"
    return storage.save_bytes(
        (
            f"previews/tasks/{task.id}/attempts/{safe_attempt}/"
            f"{segment.id}.jpg"
        ),
        preview_bytes,
        "image/jpeg",
    )


def _snapshot_settings(task: AnnotationTask, pipeline: Pipeline) -> dict:
    if task.settings_snapshot:
        return dict(task.settings_snapshot)
    config = getattr(pipeline, "config", None)
    return {
        "detection_confidence_threshold": float(
            getattr(config, "liff_yolo_world_confidence", 0.15)
        ),
        "export_confidence_threshold": float(
            getattr(config, "liff_export_confidence_threshold", 0.5)
        ),
        "yolo_imgsz": int(getattr(config, "liff_yolo_imgsz", 640)),
        "model_name": "yolov8x-worldv2",
        "model_version": "v8.4.0",
        "exclusion_rule": "detection_confidence < export_confidence_threshold",
    }


def process_task(
    repo: Repository,
    pipeline: Pipeline,
    task: AnnotationTask,
    output_dir: Path,
    *,
    storage: StorageService | None = None,
    ensure_lease: Callable[[], None] | None = None,
) -> AnnotationTask:
    """執行 LIFF 任務；耗時 I/O 全在 claim transaction 之外。"""
    if storage is None:
        storage = getattr(pipeline, "storage", None)

    pipeline_config = getattr(pipeline, "config", None)
    if (
        bool(getattr(pipeline_config, "use_gcs", False))
        and (storage is None or storage.backend_name != "gcs")
    ):
        raise RuntimeError("USE_GCS=1，但 task_processor 沒有取得 GCS storage")

    if storage is None:
        storage = StorageService(
            local=LocalStorage(
                data_dir=output_dir.parent,
                upload_dir=output_dir.parent / "uploads",
                mask_dir=output_dir.parent / "masks",
                dataset_dir=output_dir,
            )
        )

    ensure_lease = ensure_lease or (lambda: None)
    claim_token = task.claim_token
    settings = _snapshot_settings(task, pipeline)
    task.settings_snapshot = settings
    export_threshold = float(settings["export_confidence_threshold"])

    # Worker 是獨立行程，可安全套用任務快照，不會改到 Web service 的全域參數。
    if pipeline_config is not None:
        pipeline_config.yolo_world_confidence = float(
            settings["detection_confidence_threshold"]
        )
        pipeline_config.yolo_imgsz = int(settings["yolo_imgsz"])

    # 每個任務開始前重新讀取該 owner 最新 few-shot classifier。
    if task.user_id and hasattr(pipeline, "refit"):
        pipeline.refit(task.user_id)

    processed_image_ids = set(task.processed_image_ids)
    pending_image_ids = [
        image_id for image_id in task.image_ids
        if image_id not in processed_image_ids
    ]
    if not pending_image_ids:
        raise ValueError("任務沒有尚未處理的圖片")

    # 追加圖片時保留舊結果，只移除這批圖片的舊 LIFF attempt 資料。
    prior_previews = []
    for result in task.excluded_results:
        if result.get("image_id") in pending_image_ids:
            preview_path = str(result.get("preview_path", ""))
            if preview_path:
                storage.delete(preview_path)
        else:
            prior_previews.append(result)
    task.excluded_results = prior_previews

    no_detection = set(task.no_detection_image_ids)
    for image_id in pending_image_ids:
        ensure_lease()
        image = repo.get_image(image_id)
        if image is None:
            raise ValueError(f"找不到任務圖片：{image_id}")

        old_task_segments = [
            segment for segment in repo.list_segments(image_id)
            if segment.annotation_task_id == task.id
        ]
        _remove_segments(repo, storage, old_task_segments)

        segments = pipeline.segment_text(
            image,
            task.prompt,
            annotation_task_id=task.id,
            task_attempt_token=claim_token,
        )
        if segments:
            no_detection.discard(image_id)
        else:
            no_detection.add(image_id)
        ensure_lease()

    task.no_detection_image_ids = sorted(no_detection)

    task_segments = [
        segment for segment in repo.list_segments()
        if (
            segment.image_id in set(task.image_ids)
            and (
                (
                    segment.annotation_task_id == task.id
                    and (
                        segment.image_id in processed_image_ids
                        or segment.task_attempt_token == claim_token
                    )
                )
                or (
                    not segment.annotation_task_id
                    and segment.image_id in processed_image_ids
                )
            )
        )
    ]
    if not task_segments:
        raise ValueError(f"找不到符合標註內容的物件：{task.prompt}")

    # 升級前完成的 LIFF 任務沒有 task_id 欄位；僅針對已記錄為該任務
    # processed 的圖片補歸屬，確保追加圖片產生的新 ZIP 仍包含舊結果。
    for segment in task_segments:
        if not segment.annotation_task_id:
            segment.annotation_task_id = task.id
            repo.update_segment(segment)

    excluded = [
        segment for segment in task_segments
        if segment.detection_confidence < export_threshold
    ]
    exportable = [
        segment for segment in task_segments
        if segment.final_label
        and segment.detection_confidence >= export_threshold
    ]

    excluded_results = list(task.excluded_results)
    for segment in excluded:
        if any(
            item.get("segment_id") == segment.id
            for item in excluded_results
        ):
            continue
        # 僅在完成前建立小型 JPEG；原圖與 mask 保持 private。
        image = repo.get_image(segment.image_id)
        preview_path = _save_excluded_preview(
            storage,
            task,
            segment,
            claim_token,
            image,
        )
        excluded_results.append(
            {
                "segment_id": segment.id,
                "image_id": segment.image_id,
                "detection_confidence": segment.detection_confidence,
                "preview_path": preview_path,
            }
        )

    task.segment_count = len(task_segments)
    task.exported_count = len(exportable)
    task.excluded_count = len(excluded)
    task.excluded_results = excluded_results
    task.processed_image_ids = list(dict.fromkeys([
        *task.processed_image_ids,
        *pending_image_ids,
    ]))

    ensure_lease()
    if exportable:
        next_dataset_version = task.dataset_version + 1
        attempt_prefix = (
            f"attempts/{claim_token}/" if claim_token else ""
        )
        object_name = (
            f"datasets/{task.id}/{attempt_prefix}"
            f"dataset_v{next_dataset_version}.zip"
        )
        with storage.open_writer(object_name, "application/zip") as (
            output,
            dataset_reference,
        ):
            write_dataset(
                repo,
                "yolo",
                output=output,
                image_ids=set(task.image_ids),
                storage=storage,
                annotation_task_id=task.id,
                minimum_detection_confidence=export_threshold,
                segment_ids={segment.id for segment in exportable},
            )
        task.dataset_version = next_dataset_version
        task.dataset_zip_path = dataset_reference
        task.completion_reason = "export_ready"
    else:
        # 全部低信心或無有效 label 時，成功完成但不產生空 ZIP。
        task.dataset_zip_path = ""
        task.completion_reason = (
            "all_segments_below_confidence"
            if len(excluded) == len(task_segments)
            else "no_exportable_segments"
        )

    ensure_lease()
    completed = repo.complete_claimed_task(task, claim_token)
    if completed is None:
        raise TaskLeaseLostError("任務 lease 已失效，放棄發布 attempt 結果")
    return completed
