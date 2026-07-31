"""測試壓縮包 (zip / tar.gz / 7z) 影像資料集上傳與自動解壓提取。"""
from __future__ import annotations

import io
import tarfile
import zipfile
import pytest
from PIL import Image

try:
    import py7zr
except ImportError:
    py7zr = None

from app import create_app
from app.config import Config


def _create_sample_png_bytes(color: tuple[int, int, int] = (255, 0, 0)) -> bytes:
    buf = io.BytesIO()
    Image.new("RGB", (32, 32), color=color).save(buf, format="PNG")
    return buf.getvalue()


def _make_zip_archive(files_dict: dict[str, bytes]) -> bytes:
    buf = io.BytesIO()
    with zipfile.ZipFile(buf, "w", zipfile.ZIP_DEFLATED) as zf:
        for filename, data in files_dict.items():
            zf.writestr(filename, data)
    return buf.getvalue()


def _make_tar_gz_archive(files_dict: dict[str, bytes]) -> bytes:
    buf = io.BytesIO()
    with tarfile.open(fileobj=buf, mode="w:gz") as tf:
        for filename, data in files_dict.items():
            ti = tarfile.TarInfo(name=filename)
            ti.size = len(data)
            tf.addfile(ti, io.BytesIO(data))
    return buf.getvalue()


def _make_7z_archive(files_dict: dict[str, bytes]) -> bytes:
    if py7zr is None:
        pytest.skip("py7zr 未安裝")
    buf = io.BytesIO()
    with py7zr.SevenZipFile(buf, mode="w") as sz:
        for filename, data in files_dict.items():
            sz.writestr(data, filename)
    return buf.getvalue()


@pytest.fixture
def app_client(tmp_path):
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
    client = app.test_client()

    # login admin user
    user = app.repo.get_user_by_username(cfg.default_admin_user)
    with client.session_transaction() as session:
        session["user_id"] = user.id
        session["username"] = user.username

    return app, client


def test_upload_zip_archive(app_client):
    app, client = app_client
    img1 = _create_sample_png_bytes((255, 0, 0))
    img2 = _create_sample_png_bytes((0, 255, 0))

    zip_bytes = _make_zip_archive({
        "dataset/cat.png": img1,
        "dataset/subfolder/dog.png": img2,
        "dataset/notes.txt": b"ignore me",
        "__MACOSX/._cat.png": b"apple metadata",
    })

    data = {
        "files": (io.BytesIO(zip_bytes), "test_dataset.zip"),
    }

    res = client.post("/api/images", data=data, content_type="multipart/form-data")
    assert res.status_code == 201
    json_data = res.get_json()
    assert isinstance(json_data, list)
    assert len(json_data) == 2

    filenames = {item["filename"] for item in json_data}
    assert "cat.png" in filenames
    assert "dog.png" in filenames
    assert "notes.txt" not in filenames


def test_upload_tar_gz_archive(app_client):
    app, client = app_client
    img1 = _create_sample_png_bytes((0, 0, 255))

    tar_bytes = _make_tar_gz_archive({
        "train/image_01.png": img1,
        "train/readme.md": b"# Readme",
    })

    data = {
        "files": (io.BytesIO(tar_bytes), "dataset.tar.gz"),
    }

    res = client.post("/api/images", data=data, content_type="multipart/form-data")
    assert res.status_code == 201
    json_data = res.get_json()
    assert len(json_data) == 1
    assert json_data[0]["filename"] == "image_01.png"


def test_upload_7z_archive(app_client):
    if py7zr is None:
        pytest.skip("py7zr 未安裝")

    app, client = app_client
    img1 = _create_sample_png_bytes((128, 128, 0))

    sz_bytes = _make_7z_archive({
        "data/sample.png": img1,
    })

    data = {
        "files": (io.BytesIO(sz_bytes), "dataset.7z"),
    }

    res = client.post("/api/images", data=data, content_type="multipart/form-data")
    assert res.status_code == 201
    json_data = res.get_json()
    assert len(json_data) == 1
    assert json_data[0]["filename"] == "sample.png"
