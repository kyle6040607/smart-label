from pathlib import Path

from app.models import AnnotationTask
from app.repository import Repository
from app.services.exporter import build_dataset
from app.services.pipeline import Pipeline


def process_task(
    repo: Repository,
    pipeline: Pipeline,
    task: AnnotationTask,
    output_dir: Path,
) -> AnnotationTask:
    """執行單一 LIFF 標註任務並產生 YOLO ZIP。"""
    segment_count = 0
    for image_id in task.image_ids:
        image = repo.get_image(image_id)

        if image is None:
            raise ValueError(
                f"找不到任務圖片：{image_id}"
            )

        segments = pipeline.segment_text(
            image,
            task.prompt,
        )
        segment_count += len(segments)

    if segment_count == 0:
        raise ValueError(
            f"找不到符合標註內容的物件：{task.prompt}"
        )

    zip_bytes = build_dataset(
        repo,
        "yolo",
        image_ids=set(task.image_ids),
    )

    task_dir = output_dir / task.id
    task_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    zip_path = task_dir / "dataset.zip"
    zip_path.write_bytes(zip_bytes)

    task.dataset_zip_path = str(zip_path)
    task.status = "completed"
    task.error_message = ""

    return repo.update_task(task)