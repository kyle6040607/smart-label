"""擁有者隔離測試：一般使用者只能碰自己的資料，admin 可以看到/管理所有人的。"""
from __future__ import annotations

import io
import json
import zipfile

import pytest
from PIL import Image as PILImage
from werkzeug.security import generate_password_hash

from app import create_app
from app.config import Config
from app.models import User


@pytest.fixture
def app_and_users(tmp_path):
    cfg = Config(
        base_dir=tmp_path, data_dir=tmp_path, upload_dir=tmp_path / "up",
        mask_dir=tmp_path / "mask", db_file=tmp_path / "store.json",
    )
    cfg.ensure_dirs()
    cfg.use_real_sam = False
    cfg.use_real_embedding = False
    app = create_app(cfg)
    app.config["TESTING"] = True
    with app.app_context():
        repo = app.repo
        alice = repo.add_user(User(username="alice", password_hash=generate_password_hash("pw"), role="user"))
        bob = repo.add_user(User(username="bob", password_hash=generate_password_hash("pw"), role="user"))
        yield app, alice, bob


def login(client, username, password="pw"):
    r = client.post("/login", data={"username": username, "password": password})
    assert r.status_code == 302, r.status_code


def _png_bytes(color, size=(60, 60)):
    buf = io.BytesIO()
    PILImage.new("RGB", size, color).save(buf, format="PNG")
    return buf.getvalue()


def upload_image(client, color=(200, 50, 50)):
    r = client.post(
        "/api/images",
        data={"files": (io.BytesIO(_png_bytes(color)), "a.png")},
        content_type="multipart/form-data",
    )
    assert r.status_code == 201, r.get_json()
    return r.get_json()[0]["id"]


def make_segment(client, image_id):
    r = client.post(f"/api/images/{image_id}/segment_point", json={"x": 20, "y": 20})
    assert r.status_code == 201, r.get_json()
    return r.get_json()["id"]


def test_images_isolated(app_and_users):
    app, alice, bob = app_and_users
    c = app.test_client()

    login(c, "alice")
    img_id = upload_image(c)
    assert len(c.get("/api/images").get_json()) == 1
    c.get("/logout")

    login(c, "bob")
    assert c.get("/api/images").get_json() == []
    assert c.get(f"/api/images/{img_id}/file").status_code == 404
    assert c.delete(f"/api/images/{img_id}").status_code == 404
    c.get("/logout")

    login(c, "sa", "sa")
    assert len(c.get("/api/images").get_json()) == 1
    assert c.get(f"/api/images/{img_id}/file").status_code == 200


def test_segments_isolated(app_and_users):
    app, alice, bob = app_and_users
    c = app.test_client()

    login(c, "alice")
    img_id = upload_image(c)
    seg_id = make_segment(c, img_id)
    c.get("/logout")

    login(c, "bob")
    assert c.post(f"/api/images/{img_id}/segment_point", json={"x": 20, "y": 20}).status_code == 404
    assert c.get(f"/api/segments/{seg_id}/mask").status_code == 404
    assert c.delete(f"/api/segments/{seg_id}").status_code == 404
    assert c.get("/api/review/queue").get_json() == []
    assert c.get("/api/stats").get_json()["total_segments"] == 0
    c.get("/logout")

    login(c, "sa", "sa")
    assert c.get(f"/api/segments/{seg_id}/mask").status_code == 200
    assert len(c.get("/api/review/queue").get_json()) == 1
    assert c.get("/api/stats").get_json()["total_segments"] == 1


def test_labels_independent_per_user(app_and_users):
    """分類器/範例各自維護：A 標的類別，B 完全看不到，也不會影響 B 的分類器。"""
    app, alice, bob = app_and_users
    c = app.test_client()

    login(c, "alice")
    a_img = upload_image(c, (200, 40, 40))
    a_seg = make_segment(c, a_img)
    assert c.post(f"/api/segments/{a_seg}/label", json={"label": "cat"}).status_code == 201
    assert c.get("/api/labels").get_json() == ["cat"]
    c.get("/logout")

    login(c, "bob")
    assert c.get("/api/labels").get_json() == []
    b_img = upload_image(c, (40, 40, 200))
    b_seg = make_segment(c, b_img)
    assert c.post(f"/api/segments/{b_seg}/label", json={"label": "dog"}).status_code == 201
    assert c.get("/api/labels").get_json() == ["dog"]
    c.get("/logout")

    login(c, "sa", "sa")
    assert set(c.get("/api/labels").get_json()) == {"cat", "dog"}


def test_delete_label_scoped_and_admin_cross_user(app_and_users):
    app, alice, bob = app_and_users
    c = app.test_client()

    login(c, "alice")
    a_img = upload_image(c)
    a_seg = make_segment(c, a_img)
    c.post(f"/api/segments/{a_seg}/label", json={"label": "cat"})
    c.get("/logout")

    login(c, "bob")
    # bob 沒有 "cat" 類別，刪不到（也刪不到別人的）
    assert c.delete("/api/labels/cat").status_code == 404
    c.get("/logout")

    login(c, "sa", "sa")
    # admin 不帶 owner_id，只會刪自己的（sa 沒有 cat）
    assert c.delete("/api/labels/cat").status_code == 404
    # admin 帶 ?owner_id= 才能刪指定使用者的類別
    r = c.delete(f"/api/labels/cat?owner_id={alice.id}")
    assert r.status_code == 200
    assert r.get_json()["examples_removed"] == 1


def test_examples_response_has_no_sensitive_fields(app_and_users):
    app, alice, bob = app_and_users
    c = app.test_client()

    login(c, "alice")
    a_img = upload_image(c)
    a_seg = make_segment(c, a_img)
    c.post(f"/api/segments/{a_seg}/label", json={"label": "cat"})
    examples = c.get("/api/examples").get_json()
    assert len(examples) == 1
    assert set(examples[0].keys()) == {"id", "label", "created_at"}


def test_export_scoped(app_and_users):
    app, alice, bob = app_and_users
    c = app.test_client()

    login(c, "alice")
    a_img = upload_image(c, (200, 40, 40))
    a_seg = make_segment(c, a_img)
    c.post(f"/api/segments/{a_seg}/label", json={"label": "cat"})

    def coco_images(resp):
        z = zipfile.ZipFile(io.BytesIO(resp.data))
        return json.loads(z.read("annotations.json"))["images"]

    r = c.get("/api/export?format=coco")
    assert r.status_code == 200
    assert len(coco_images(r)) == 1
    c.get("/logout")

    login(c, "bob")
    r = c.get("/api/export?format=coco")
    assert coco_images(r) == []
    c.get("/logout")

    login(c, "sa", "sa")
    r = c.get("/api/export?format=coco")
    assert len(coco_images(r)) == 1
