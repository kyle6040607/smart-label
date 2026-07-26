import io
from pathlib import Path

import pytest
from PIL import Image

from app import create_app
from app.config import Config
from app.services import line_login


@pytest.fixture
def app(tmp_path):
    cfg = Config(
        base_dir=tmp_path,
        data_dir=tmp_path,
        upload_dir=tmp_path / "uploads",
        mask_dir=tmp_path / "masks",
        db_file=tmp_path / "store.json",
    )

    cfg.db_backend = "json"
    cfg.line_login_channel_id = "test-channel-id"
    cfg.liff_id = "test-liff-id"
    cfg.use_real_sam = False
    cfg.use_real_embedding = False
    cfg.ensure_dirs()

    application = create_app(cfg)
    application.config["TESTING"] = True

    return application


def make_png_file() -> io.BytesIO:
    image_file = io.BytesIO()

    Image.new(
        "RGB",
        (20, 10),
        color=(255, 0, 0),
    ).save(
        image_file,
        format="PNG",
    )

    image_file.seek(0)

    return image_file


def test_liff_upload_creates_annotation_task(
    app,
    monkeypatch,
):
    monkeypatch.setattr(
        line_login,
        "verify_id_token",
        lambda *args: {
            "sub": "U-liff-upload-test",
            "name": "LIFF 測試使用者",
            "picture": "",
        },
    )

    client = app.test_client()

    response = client.post(
        "/liff/upload",
        data={
            "prompt": "請標註圖片中的貓咪",
            "id_token": "fake-id-token",
            "images": (
                make_png_file(),
                "cat.png",
            ),
        },
        content_type="multipart/form-data",
    )

    assert response.status_code == 201

    result = response.get_json()

    assert result is not None
    assert result["ok"] is True
    assert result["task_status"] == "pending"
    assert result["image_count"] == 1
    assert result["user_id"] == ""

    task = app.repo.get_task(
        result["task_id"]
    )

    assert task is not None
    assert task.line_user_id == "U-liff-upload-test"
    assert task.user_id == ""
    assert task.prompt == "請標註圖片中的貓咪"
    assert len(task.image_ids) == 1

    image_record = app.repo.get_image(
        task.image_ids[0]
    )

    assert image_record is not None
    assert image_record.width == 20
    assert image_record.height == 10
    assert Path(image_record.path).exists()