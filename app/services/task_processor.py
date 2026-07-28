from pathlib import Path

from app.models import AnnotationTask
from app.repository import Repository
from app.services.exporter import write_dataset
from app.services.pipeline import Pipeline
from app.storage import LocalStorage, StorageService


def process_task(
    repo: Repository,
    pipeline: Pipeline,
    task: AnnotationTask,
    output_dir: Path,
    *,
    storage: StorageService | None = None,
) -> AnnotationTask:
    """執行單一 LIFF 標註任務並產生 YOLO ZIP。"""
    if storage is None:
        storage = getattr(pipeline, "storage", None)

    pipeline_config = getattr(pipeline, "config", None)
    if (
        bool(getattr(pipeline_config, "use_gcs", False))
        and (
            storage is None
            or storage.backend_name != "gcs"
        )
    ):
        raise RuntimeError(
            "USE_GCS=1，但 task_processor 沒有取得 GCS storage"
        )

    if storage is None:
        storage = StorageService(
            local=LocalStorage(
                data_dir=output_dir.parent,
                upload_dir=output_dir.parent / "uploads",
                mask_dir=output_dir.parent / "masks",
                dataset_dir=output_dir,
            )
        )

    processed_image_ids = set(task.processed_image_ids)
    pending_image_ids = [
        image_id
        for image_id in task.image_ids
        if image_id not in processed_image_ids
    ]

    if not pending_image_ids:
        raise ValueError("任務沒有尚未處理的圖片")

    segment_count = 0
    for image_id in pending_image_ids:
        image = repo.get_image(image_id)

        if image is None:
            raise ValueError(
                f"找不到任務圖片：{image_id}"
            )
        existing_segments = repo.list_segments(image_id)
        #重試時先清掉未完成圖片的舊 segments 和 mask 檔，不會重複累加。
        if existing_segments:
            mask_paths = repo.delete_segments_batch(
                [
                    segment.id
                    for segment in existing_segments
                ]
            )

            for mask_path in mask_paths:
                storage.delete(mask_path)

        segments = pipeline.segment_text(
            image,
            task.prompt,
        )
        segment_count += len(segments)

    if segment_count == 0:
        raise ValueError(
            f"找不到符合標註內容的物件：{task.prompt}"
        )

    next_dataset_version = task.dataset_version + 1
    object_name = (
        f"datasets/{task.id}/"
        f"dataset_v{next_dataset_version}.zip"
    )
    with storage.open_writer(
        object_name,
        "application/zip",
    ) as (output, dataset_reference):
        write_dataset(
            repo,
            "yolo",
            output=output,
            image_ids=set(task.image_ids),
            storage=storage,
        )

    task.processed_image_ids = list(dict.fromkeys([
        *task.processed_image_ids,
        *pending_image_ids,
    ]))
    task.dataset_version = next_dataset_version
    task.dataset_zip_path = dataset_reference
    task.status = "completed"
    task.error_message = ""

    return repo.update_task(task)
