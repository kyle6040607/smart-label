import json
import pytest
from app import create_app
from app.config import Config
from app.models import ImageRecord, Segment

@pytest.fixture
def client(tmp_path):
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
        with app.test_client() as client:
            user = app.repo.get_user_by_username(cfg.default_admin_user)
            assert user is not None
            with client.session_transaction() as sess:
                sess["user_id"] = user.id
                sess["username"] = user.username
            yield client

def test_parameters_get_and_post(client):
    # 1. Test GET /api/parameters
    res = client.get("/api/parameters")
    assert res.status_code == 200
    data = res.get_json()
    assert data["confidence_threshold"] == 0.6
    assert data["yolo_world_confidence"] == 0.4
    
    # 2. Test POST /api/parameters
    res = client.post("/api/parameters", json={
        "confidence_threshold": 0.5,
        "yolo_world_confidence": 0.3
    })
    assert res.status_code == 200
    data = res.get_json()
    assert data["status"] == "success"
    assert data["parameters"]["confidence_threshold"] == 0.5
    assert data["parameters"]["yolo_world_confidence"] == 0.3
    
    # Verify GET returns the updated values
    res = client.get("/api/parameters")
    assert res.status_code == 200
    data = res.get_json()
    assert data["confidence_threshold"] == 0.5
    assert data["yolo_world_confidence"] == 0.3

def test_reclassify_pending_on_parameter_update(client, tmp_path):
    from flask import current_app
    import cv2
    repo = current_app.repo
    pipeline = current_app.pipeline
    
    # Create mock mask image on disk since cv2.imread is called in reclassify_pending
    mask_dir = tmp_path / "mask"
    mask_path = mask_dir / "img1_seg1.png"
    # Write a simple black 10x10 png image
    import numpy as np
    dummy_img = np.zeros((10, 10), dtype=np.uint8)
    cv2.imwrite(str(mask_path), dummy_img)
    
    # Create mock original image
    img_dir = tmp_path / "up"
    img_path = img_dir / "img1.png"
    cv2.imwrite(str(img_path), np.zeros((10, 10, 3), dtype=np.uint8))
    
    img = repo.add_image(ImageRecord(id="img1", filename="img1.png", path=str(img_path)))
    
    # Add example to train classifier and make it ready
    from app.models import LabelExample
    repo.add_example(LabelExample(label="cat", feature=[0.1]*512))
    pipeline.refit()
    assert pipeline.classifier.ready is True
    
    # Mock classifier to predict 0.55 confidence for the category "cat"
    pipeline.classifier.predict = lambda feat: {"cat": 0.55}
    
    # Add a pending segment
    seg = Segment(id="seg1", image_id="img1", mask_path=str(mask_path))
    repo.add_segment(seg)
    
    # Update threshold to 0.60 -> confidence (0.55) < threshold (0.60) -> needs_review = True
    res = client.post("/api/parameters", json={"confidence_threshold": 0.60})
    assert res.status_code == 200
    
    seg_updated = repo.get_segment("seg1")
    assert seg_updated.needs_review is True
    
    # Update threshold to 0.50 -> confidence (0.55) >= threshold (0.50) -> needs_review = False
    res = client.post("/api/parameters", json={"confidence_threshold": 0.50})
    assert res.status_code == 200
    
    seg_updated = repo.get_segment("seg1")
    assert seg_updated.needs_review is False

def test_parameters_validation(client):
    # Test invalid string input -> 400
    res = client.post("/api/parameters", json={"confidence_threshold": "invalid"})
    assert res.status_code == 400
    assert "error" in res.get_json()

    # Test boolean input -> 400
    res = client.post("/api/parameters", json={"confidence_threshold": True})
    assert res.status_code == 400

    # Test out of range input (> 1.0) -> 400
    res = client.post("/api/parameters", json={"confidence_threshold": 1.5})
    assert res.status_code == 400

    # Test out of range input (< 0.0) -> 400
    res = client.post("/api/parameters", json={"yolo_world_confidence": -0.2})
    assert res.status_code == 400

def test_parameters_persistence(client, tmp_path):
    # Post parameter update
    res = client.post("/api/parameters", json={
        "confidence_threshold": 0.75,
        "yolo_world_confidence": 0.25,
        "yolo_imgsz": 1280
    })
    assert res.status_code == 200

    # Re-initialize app with same db_file path
    cfg = Config(
        base_dir=tmp_path, data_dir=tmp_path, upload_dir=tmp_path / "up",
        mask_dir=tmp_path / "mask", db_file=tmp_path / "store.json",
    )
    new_app = create_app(cfg)
    assert new_app.pipeline.config.confidence_threshold == 0.75
    assert new_app.pipeline.config.yolo_world_confidence == 0.25
    # 存進 parameters 表是 float，還原時必須轉回 int，否則會傳 1280.0 給 Ultralytics
    assert new_app.pipeline.config.yolo_imgsz == 1280
    assert isinstance(new_app.pipeline.config.yolo_imgsz, int)


def test_yolo_imgsz_get_and_post(client):
    res = client.get("/api/parameters")
    assert res.get_json()["yolo_imgsz"] == 640

    res = client.post("/api/parameters", json={"yolo_imgsz": 1280})
    assert res.status_code == 200
    assert res.get_json()["parameters"]["yolo_imgsz"] == 1280

    assert client.get("/api/parameters").get_json()["yolo_imgsz"] == 1280


def test_yolo_imgsz_validation(client):
    # 非 32 的倍數 -> 400（Ultralytics 會靜默調整，必須先擋掉）
    assert client.post("/api/parameters", json={"yolo_imgsz": 700}).status_code == 400
    # 超出範圍 -> 400
    assert client.post("/api/parameters", json={"yolo_imgsz": 64}).status_code == 400
    assert client.post("/api/parameters", json={"yolo_imgsz": 4096}).status_code == 400
    # 非整數 -> 400
    assert client.post("/api/parameters", json={"yolo_imgsz": 640.5}).status_code == 400
    assert client.post("/api/parameters", json={"yolo_imgsz": "big"}).status_code == 400
    assert client.post("/api/parameters", json={"yolo_imgsz": True}).status_code == 400
    # 被擋下來之後不可留下副作用
    assert client.get("/api/parameters").get_json()["yolo_imgsz"] == 640

