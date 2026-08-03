from __future__ import annotations

import io
import zipfile
from pathlib import Path

import cv2
import numpy as np
import pytest
from google.api_core.exceptions import NotFound
from PIL import Image

from app import create_app
from app.config import Config
from app.models import ImageRecord
from app.repository import Repository
from app.services.exporter import build_dataset
from app.services.pipeline import Pipeline
from app.storage import GCSStorage, LocalStorage, StorageService


class FakeBlob:
    def __init__(self, objects: dict[str, bytes], name: str):
        self.objects = objects
        self.name = name

    def upload_from_string(
        self,
        data: bytes,
        content_type: str | None = None,
    ) -> None:
        del content_type
        self.objects[self.name] = bytes(data)

    def upload_from_filename(
        self,
        filename: str,
        content_type: str | None = None,
    ) -> None:
        del content_type
        self.objects[self.name] = Path(filename).read_bytes()

    def download_as_bytes(self) -> bytes:
        if self.name not in self.objects:
            raise NotFound("missing")
        return self.objects[self.name]

    def open(self, mode: str, **kwargs):
        del kwargs
        if mode == "rb":
            if self.name not in self.objects:
                raise FileNotFoundError(self.name)
            return io.BytesIO(self.objects[self.name])
        if mode == "wb":
            objects = self.objects
            name = self.name

            class CommittingWriter(io.BytesIO):
                def close(self):
                    if not self.closed:
                        objects[name] = self.getvalue()
                    super().close()

            return CommittingWriter()
        raise ValueError(mode)

    def generate_signed_url(self, **kwargs) -> str:
        del kwargs
        return f"https://storage.example/{self.name}?signed=1"

    def delete(self) -> None:
        if self.name not in self.objects:
            raise NotFound("missing")
        del self.objects[self.name]

    def exists(self) -> bool:
        return self.name in self.objects


class FakeBucket:
    def __init__(self):
        self.objects: dict[str, bytes] = {}

    def blob(self, name: str) -> FakeBlob:
        return FakeBlob(self.objects, name)

    def list_blobs(self, prefix: str):
        return [
            FakeBlob(self.objects, name)
            for name in list(self.objects)
            if name.startswith(prefix)
        ]


class FakeClient:
    def __init__(self):
        self.buckets: dict[str, FakeBucket] = {}

    def bucket(self, name: str) -> FakeBucket:
        return self.buckets.setdefault(name, FakeBucket())


def make_storage(
    tmp_path,
    *,
    use_gcs: bool,
) -> StorageService:
    local = LocalStorage(
        data_dir=tmp_path / "data",
        upload_dir=tmp_path / "uploads",
        mask_dir=tmp_path / "masks",
    )
    gcs = GCSStorage(
        bucket_name="test-bucket",
        project_id="test-project",
        client=FakeClient(),
    )
    return StorageService(
        local=local,
        gcs=gcs,
        use_gcs=use_gcs,
    )


def test_local_storage_keeps_existing_directories(tmp_path):
    storage = make_storage(tmp_path, use_gcs=False)

    image_path = storage.save_bytes(
        "images/cat.png",
        b"image",
        "image/png",
    )
    mask_path = storage.save_bytes(
        "masks/cat.png",
        b"mask",
        "image/png",
    )
    dataset_path = storage.save_bytes(
        "datasets/task-1/dataset.zip",
        b"zip",
        "application/zip",
    )

    assert image_path == str(tmp_path / "uploads" / "cat.png")
    assert mask_path == str(tmp_path / "masks" / "cat.png")
    assert dataset_path == str(
        tmp_path / "data" / "tasks" / "task-1" / "dataset.zip"
    )
    assert storage.read_bytes(image_path) == b"image"


def test_storage_can_stream_and_upload_existing_file(tmp_path):
    source = tmp_path / "source.zip"
    source.write_bytes(b"streamed-zip")
    storage = make_storage(tmp_path, use_gcs=True)

    reference = storage.save_file(
        "datasets/task-1/dataset.zip",
        source,
        "application/zip",
    )

    assert reference == (
        "gs://test-bucket/datasets/task-1/dataset.zip"
    )
    with storage.open_reader(reference) as reader:
        assert reader.read(4) == b"stre"
        assert reader.read() == b"amed-zip"


def test_storage_can_write_zip_stream_and_generate_signed_url(
    tmp_path,
):
    storage = make_storage(tmp_path, use_gcs=True)

    with storage.open_writer(
        "datasets/task-2/dataset.zip",
        "application/zip",
    ) as (writer, reference):
        writer.write(b"streamed-output")

    assert storage.read_bytes(reference) == b"streamed-output"
    assert storage.generate_download_url(
        reference,
        "dataset.zip",
    ) == (
        "https://storage.example/"
        "datasets/task-2/dataset.zip?signed=1"
    )


def test_local_stream_writer_does_not_publish_partial_file(
    tmp_path,
):
    storage = make_storage(tmp_path, use_gcs=False)
    target = (
        tmp_path
        / "data"
        / "tasks"
        / "task-3"
        / "dataset.zip"
    )

    with pytest.raises(RuntimeError, match="zip failed"):
        with storage.open_writer(
            "datasets/task-3/dataset.zip",
            "application/zip",
        ) as (writer, reference):
            assert reference == str(target)
            writer.write(b"partial")
            raise RuntimeError("zip failed")

    assert not target.exists()
    assert not target.with_suffix(".zip.tmp").exists()


def test_gcs_storage_and_legacy_local_path_can_coexist(tmp_path):
    storage = make_storage(tmp_path, use_gcs=True)

    gcs_path = storage.save_bytes(
        "images/cat.png",
        b"gcs-image",
        "image/png",
    )
    local_path = storage.local.save_bytes(
        "images/legacy.png",
        b"local-image",
        "image/png",
    )

    assert gcs_path == "gs://test-bucket/images/cat.png"
    assert storage.read_bytes(gcs_path) == b"gcs-image"
    assert storage.read_bytes(local_path) == b"local-image"
    assert storage.delete(gcs_path) is True
    assert storage.delete(gcs_path) is False


@pytest.mark.parametrize("use_gcs", [False, True])
def test_storage_delete_prefix_only_removes_selected_attempt(
    tmp_path,
    use_gcs,
):
    storage = make_storage(tmp_path, use_gcs=use_gcs)
    stale = storage.save_bytes(
        "previews/tasks/task-1/attempts/stale/a.jpg",
        b"stale",
    )
    current = storage.save_bytes(
        "previews/tasks/task-1/attempts/current/b.jpg",
        b"current",
    )

    assert storage.delete_prefix(
        "previews/tasks/task-1/attempts/stale"
    ) == 1
    assert not storage.exists(stale)
    assert storage.exists(current)


def test_pipeline_rejects_missing_storage_when_gcs_is_enabled(
    tmp_path,
):
    cfg = Config(
        base_dir=tmp_path,
        data_dir=tmp_path / "data",
        upload_dir=tmp_path / "uploads",
        mask_dir=tmp_path / "masks",
        db_file=tmp_path / "store.json",
    )
    cfg.use_gcs = True
    repo = Repository(cfg.db_file)

    with pytest.raises(
        RuntimeError,
        match="USE_GCS=1",
    ):
        Pipeline(cfg, repo)


def test_pipeline_rejects_local_backend_when_gcs_is_enabled(
    tmp_path,
):
    cfg = Config(
        base_dir=tmp_path,
        data_dir=tmp_path / "data",
        upload_dir=tmp_path / "uploads",
        mask_dir=tmp_path / "masks",
        db_file=tmp_path / "store.json",
    )
    cfg.use_gcs = True
    repo = Repository(cfg.db_file)
    local_storage = make_storage(tmp_path, use_gcs=False)

    with pytest.raises(
        RuntimeError,
        match="backend 不是 GCS",
    ):
        Pipeline(cfg, repo, storage=local_storage)


def test_pipeline_and_export_read_images_and_masks_from_gcs(tmp_path):
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

    pixels = np.zeros((40, 40, 3), dtype=np.uint8)
    pixels[10:30, 10:30] = (255, 255, 255)
    ok, encoded = cv2.imencode(".png", pixels)
    assert ok

    storage = make_storage(tmp_path, use_gcs=True)
    image_path = storage.save_bytes(
        "images/cat.png",
        encoded.tobytes(),
        "image/png",
    )

    repo = Repository(cfg.db_file)
    image = repo.add_image(
        ImageRecord(
            filename="cat.png",
            path=image_path,
            width=40,
            height=40,
        )
    )
    pipeline = Pipeline(cfg, repo, storage=storage)

    segments = pipeline.segment_text(image, "cat")

    assert len(segments) == 1
    assert segments[0].mask_path.startswith(
        "gs://test-bucket/masks/"
    )

    dataset = build_dataset(
        repo,
        "yolo",
        storage=storage,
    )
    with zipfile.ZipFile(io.BytesIO(dataset)) as archive:
        names = archive.namelist()

    assert "data.yaml" in names
    assert any(name.startswith("images/") for name in names)
    assert any(name.startswith("labels/") for name in names)


def test_web_upload_segment_download_and_delete_use_gcs(tmp_path):
    cfg = Config(
        base_dir=tmp_path,
        data_dir=tmp_path / "data",
        upload_dir=tmp_path / "uploads",
        mask_dir=tmp_path / "masks",
        db_file=tmp_path / "store.json",
    )
    cfg.db_backend = "json"
    cfg.use_gcs = False
    cfg.use_real_sam = False
    cfg.use_real_embedding = False
    cfg.ensure_dirs()

    app = create_app(cfg)
    app.config["TESTING"] = True

    storage = make_storage(tmp_path, use_gcs=True)
    app.storage = storage
    app.pipeline.storage = storage

    image_file = io.BytesIO()
    Image.new(
        "RGB",
        (40, 40),
        color=(255, 0, 0),
    ).save(image_file, format="PNG")
    image_file.seek(0)

    client = app.test_client()
    user = app.repo.get_user_by_username(cfg.default_admin_user)
    assert user is not None
    with client.session_transaction() as session:
        session["user_id"] = user.id
        session["username"] = user.username

    upload_response = client.post(
        "/api/images",
        data={"files": (image_file, "cat.png")},
        content_type="multipart/form-data",
    )
    assert upload_response.status_code == 201

    image_data = upload_response.get_json()[0]
    image_id = image_data["id"]
    image_record = app.repo.get_image(image_id)
    assert image_record is not None
    assert image_record.path.startswith(
        "gs://test-bucket/images/"
    )

    file_response = client.get(
        f"/api/images/{image_id}/file"
    )
    assert file_response.status_code == 200
    assert file_response.data.startswith(b"\x89PNG")

    segment_response = client.post(
        f"/api/images/{image_id}/segment_point",
        json={"x": 20, "y": 20},
    )
    assert segment_response.status_code == 201
    segment_data = segment_response.get_json()
    assert segment_data["mask_path"].startswith(
        "gs://test-bucket/masks/"
    )

    mask_response = client.get(
        f"/api/segments/{segment_data['id']}/mask"
    )
    assert mask_response.status_code == 200
    assert mask_response.data.startswith(b"\x89PNG")

    delete_response = client.delete(
        f"/api/images/{image_id}"
    )
    assert delete_response.status_code == 200
    assert delete_response.get_json()["files_removed"] == 2
    assert not storage.exists(image_record.path)
