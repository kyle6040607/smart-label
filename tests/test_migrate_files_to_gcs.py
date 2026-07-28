from pathlib import Path

from app.models import AnnotationTask, ImageRecord, Segment
from app.repository import Repository
from scripts.migrate_files_to_gcs import (
    count_local_references,
    migrate,
)


class RecordingGcsStorage:
    backend_name = "gcs"

    def __init__(self):
        self.uploads: dict[str, bytes] = {}

    def save_file(
        self,
        object_name: str,
        source: Path,
        content_type: str | None = None,
    ) -> str:
        del content_type
        self.uploads[object_name] = source.read_bytes()
        return f"gs://test-bucket/{object_name}"


def test_migration_uploads_and_rewrites_all_supported_paths(
    tmp_path,
):
    repo = Repository(tmp_path / "store.json")
    image_path = tmp_path / "uploads" / "cat.jpg"
    mask_path = tmp_path / "masks" / "mask.png"
    zip_path = tmp_path / "tasks" / "task-1" / "dataset.zip"

    for path, content in (
        (image_path, b"image"),
        (mask_path, b"mask"),
        (zip_path, b"zip"),
    ):
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(content)

    image = repo.add_image(
        ImageRecord(
            id="image-1",
            filename="cat.jpg",
            path=str(image_path),
        )
    )
    segment = repo.add_segment(
        Segment(
            id="segment-1",
            image_id=image.id,
            mask_path=str(mask_path),
        )
    )
    task = repo.add_task(
        AnnotationTask(
            id="task-1",
            dataset_version=2,
            dataset_zip_path=str(zip_path),
        )
    )
    storage = RecordingGcsStorage()

    result = migrate(
        repo,
        storage,
        base_dir=tmp_path,
    )

    assert result.migrated == 3
    assert result.skipped == 0
    assert result.failed == 0
    assert repo.get_image(image.id).path == (
        "gs://test-bucket/images/image-1.jpg"
    )
    assert repo.get_segment(segment.id).mask_path == (
        "gs://test-bucket/masks/image-1_segment-1.png"
    )
    assert repo.get_task(task.id).dataset_zip_path == (
        "gs://test-bucket/datasets/task-1/dataset_v2.zip"
    )
    assert count_local_references(repo) == 0

    rerun = migrate(
        repo,
        storage,
        base_dir=tmp_path,
    )

    assert rerun.migrated == 0
    assert rerun.skipped == 3
    assert rerun.failed == 0


def test_migration_dry_run_reports_missing_source_without_db_change(
    tmp_path,
):
    repo = Repository(tmp_path / "store.json")
    image = repo.add_image(
        ImageRecord(
            id="missing-image",
            filename="missing.jpg",
            path=str(tmp_path / "missing.jpg"),
        )
    )

    result = migrate(
        repo,
        None,
        base_dir=tmp_path,
        dry_run=True,
    )

    assert result.migrated == 0
    assert result.failed == 1
    assert result.orphaned == 0
    assert count_local_references(repo) == 1
    assert repo.get_image(image.id).path == image.path
