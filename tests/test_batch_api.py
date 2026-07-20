import json
import pytest
from pathlib import Path
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
    # 我們不希望用真的 SAM/embedding 來跑測試，強制設為 mock 模式
    cfg.use_real_sam = False
    cfg.use_real_embedding = False
    
    app = create_app(cfg)
    app.config["TESTING"] = True
    with app.app_context():
        with app.test_client() as client:
            # API 現在要求登入；用種好的預設 admin 帳號登入，批次刪除測試才能通過擁有者檢查
            client.post("/login", data={
                "username": cfg.default_admin_user,
                "password": cfg.default_admin_password,
            })
            yield client

def test_delete_images_batch(client, tmp_path):
    # 造一些測試圖片檔
    up_dir = tmp_path / "up"
    img_path1 = up_dir / "img1.png"
    img_path2 = up_dir / "img2.png"
    img_path1.write_bytes(b"test1")
    img_path2.write_bytes(b"test2")
    
    # 建立 Repo
    from flask import current_app
    repo = current_app.repo
    
    # 新增圖片到資料庫
    img1 = repo.add_image(ImageRecord(id="img1", filename="img1.png", path=str(img_path1)))
    img2 = repo.add_image(ImageRecord(id="img2", filename="img2.png", path=str(img_path2)))
    
    # 驗證檔案存在
    assert img_path1.exists()
    assert img_path2.exists()
    
    # 呼叫 API 進行批次刪除
    res = client.post("/api/images/delete_batch", json={
        "image_ids": ["img1", "img2"]
    })
    assert res.status_code == 200
    data = res.get_json()
    assert "deleted_ids" in data
    assert "img1" in data["deleted_ids"]
    assert "img2" in data["deleted_ids"]
    
    # 驗證資料庫中已刪除
    assert repo.get_image("img1") is None
    assert repo.get_image("img2") is None
    
    # 驗證實體檔案已刪除
    assert not img_path1.exists()
    assert not img_path2.exists()

def test_delete_segments_batch(client, tmp_path):
    # 造一些測試遮罩檔
    mask_dir = tmp_path / "mask"
    mask_path1 = mask_dir / "mask1.png"
    mask_path2 = mask_dir / "mask2.png"
    mask_path1.write_bytes(b"mask1")
    mask_path2.write_bytes(b"mask2")
    
    # 建立 Repo
    from flask import current_app
    repo = current_app.repo
    
    # 新增遮罩到資料庫
    seg1 = repo.add_segment(Segment(id="seg1", image_id="img1", mask_path=str(mask_path1)))
    seg2 = repo.add_segment(Segment(id="seg2", image_id="img1", mask_path=str(mask_path2)))
    
    # 驗證檔案存在
    assert mask_path1.exists()
    assert mask_path2.exists()
    
    # 呼叫 API 進行批次刪除
    res = client.post("/api/segments/delete_batch", json={
        "segment_ids": ["seg1", "seg2"]
    })
    assert res.status_code == 200
    data = res.get_json()
    assert "deleted_ids" in data
    assert "seg1" in data["deleted_ids"]
    assert "seg2" in data["deleted_ids"]
    
    # 驗證資料庫中已刪除
    assert repo.get_segment("seg1") is None
    assert repo.get_segment("seg2") is None
    
    # 驗證實體檔案已刪除
    assert not mask_path1.exists()
    assert not mask_path2.exists()
