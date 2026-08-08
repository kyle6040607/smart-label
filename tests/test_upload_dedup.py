"""上傳去重：舊資料補算雜湊值時不能依賴 JSON repo 專屬的 _save。"""
from __future__ import annotations

import dataclasses
import hashlib
import io

from PIL import Image

from app import create_app
from app.config import Config
from app.models import ImageRecord, Project


class MySQLLikeRepo:
    """代理 JSON repo，但模仿 MySQLRepository 的兩個關鍵差異。

    1. 沒有 _save。
    2. list_images() 每次都回傳新物件（DB round-trip），
       所以對回傳值做的記憶體修改不會留到下一次查詢。
    """

    def __init__(self, inner, allow_save: bool = False):
        object.__setattr__(self, "_inner", inner)
        object.__setattr__(self, "_allow_save", allow_save)

    def list_images(self, project_id: str | None = None) -> list[ImageRecord]:
        return [
            dataclasses.replace(image)
            for image in self._inner.list_images(project_id=project_id)
        ]

    def __getattr__(self, name):
        if name == "_save" and not self._allow_save:
            raise AttributeError(
                "MySQLRepository 沒有 _save"
            )
        return getattr(self._inner, name)


class CountingStorage:
    """記錄 read_bytes 被呼叫幾次，用來確認補算沒有每個檔案重跑一遍。"""

    def __init__(self, inner):
        object.__setattr__(self, "_inner", inner)
        object.__setattr__(self, "reads", 0)

    def read_bytes(self, reference: str) -> bytes:
        object.__setattr__(self, "reads", self.reads + 1)
        return self._inner.read_bytes(reference)

    def __getattr__(self, name):
        return getattr(self._inner, name)


def _png_bytes(color: tuple[int, int, int]) -> bytes:
    buffer = io.BytesIO()
    Image.new("RGB", (16, 16), color=color).save(
        buffer,
        format="PNG",
    )
    return buffer.getvalue()


def _make_app(tmp_path):
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
    return app, cfg


def _login(app, client):
    user = app.repo.get_user_by_username(
        app.smart_config.default_admin_user
    )
    assert user is not None
    with client.session_transaction() as session:
        session["user_id"] = user.id
        session["username"] = user.username
    return user


def _seed_legacy_image(app, owner_id: str, data: bytes) -> ImageRecord:
    """建一筆沒有 file_hash 的舊資料（LIFF 上傳就長這樣）。"""
    record = ImageRecord(owner_id=owner_id, filename="legacy.png")
    record.path = app.storage.save_bytes(
        f"images/{record.id}_legacy.png",
        data,
        "image/png",
    )
    app.repo.add_image(record)
    return record


def test_backfill_persists_without_repo_save(tmp_path):
    app, _ = _make_app(tmp_path)
    client = app.test_client()
    user = _login(app, client)

    data = _png_bytes((255, 0, 0))
    legacy = _seed_legacy_image(app, user.id, data)
    assert legacy.file_hash == ""

    # repo 沒有 _save 也要能跑完，不可以炸 AttributeError
    app.repo = MySQLLikeRepo(app.repo)

    response = client.post(
        "/api/images",
        data={"files": (io.BytesIO(data), "again.png")},
        content_type="multipart/form-data",
    )

    assert response.status_code == 400
    assert response.get_json()["error"] == "圖片已上傳過，請勿重複上傳"

    stored = app.repo.get_image(legacy.id)
    assert stored is not None
    assert stored.file_hash == hashlib.sha256(data).hexdigest()


def test_backfill_reads_each_legacy_image_once_per_request(tmp_path):
    app, _ = _make_app(tmp_path)
    client = app.test_client()
    user = _login(app, client)

    for color in ((255, 0, 0), (0, 255, 0)):
        _seed_legacy_image(app, user.id, _png_bytes(color))

    # allow_save=True：這個測試只想量讀取次數，別被 _save 的錯誤蓋掉
    app.repo = MySQLLikeRepo(app.repo, allow_save=True)
    counting = CountingStorage(app.storage)
    app.storage = counting

    response = client.post(
        "/api/images",
        data={
            "files": [
                (io.BytesIO(_png_bytes((0, 0, 255))), "a.png"),
                (io.BytesIO(_png_bytes((0, 0, 254))), "b.png"),
                (io.BytesIO(_png_bytes((0, 0, 253))), "c.png"),
            ]
        },
        content_type="multipart/form-data",
    )

    assert response.status_code == 201
    assert len(response.get_json()) == 3
    # 兩張舊圖各讀一次即可，不該是 3 個檔案 x 2 張舊圖 = 6 次
    assert counting.reads == 2


def test_duplicate_within_single_batch_is_rejected(tmp_path):
    app, _ = _make_app(tmp_path)
    client = app.test_client()
    _login(app, client)

    data = _png_bytes((10, 20, 30))
    response = client.post(
        "/api/images",
        data={
            "files": [
                (io.BytesIO(data), "one.png"),
                (io.BytesIO(data), "two.png"),
            ]
        },
        content_type="multipart/form-data",
    )

    assert response.status_code == 201
    assert len(response.get_json()) == 1


def test_same_image_upload_to_different_projects_succeeds(tmp_path):
    app, _ = _make_app(tmp_path)
    client = app.test_client()
    user = _login(app, client)

    p1 = app.repo.add_project(Project(owner_id=user.id, name="Project 1"))
    p2 = app.repo.add_project(Project(owner_id=user.id, name="Project 2"))

    data = _png_bytes((123, 45, 67))

    # Upload image to Project 1
    res1 = client.post(
        "/api/images",
        data={
            "project_id": p1.id,
            "files": (io.BytesIO(data), "photo.png"),
        },
        content_type="multipart/form-data",
    )
    assert res1.status_code == 201
    assert len(res1.get_json()) == 1
    assert res1.get_json()[0]["project_id"] == p1.id

    # Upload same image to Project 2 (should succeed)
    res2 = client.post(
        "/api/images",
        data={
            "project_id": p2.id,
            "files": (io.BytesIO(data), "photo.png"),
        },
        content_type="multipart/form-data",
    )
    assert res2.status_code == 201
    assert len(res2.get_json()) == 1
    assert res2.get_json()[0]["project_id"] == p2.id

    # Upload same image to Project 1 again (should be rejected as duplicate)
    res3 = client.post(
        "/api/images",
        data={
            "project_id": p1.id,
            "files": (io.BytesIO(data), "photo.png"),
        },
        content_type="multipart/form-data",
    )
    assert res3.status_code == 400
    assert res3.get_json()["error"] == "圖片已上傳過，請勿重複上傳"

