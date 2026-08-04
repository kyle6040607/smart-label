import io
import pytest
from PIL import Image
from app import create_app
from app.config import Config
from app.models import ImageRecord, Project


@pytest.fixture
def client(tmp_path):
    cfg = Config(
        base_dir=tmp_path,
        data_dir=tmp_path,
        upload_dir=tmp_path / "up",
        mask_dir=tmp_path / "mask",
        db_file=tmp_path / "store.json",
    )
    cfg.ensure_dirs()
    cfg.db_backend = "json"
    cfg.use_real_sam = False
    cfg.use_real_embedding = False
    cfg.use_gcs = False

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


def test_list_and_create_projects(client):
    # 1. 取得專案列表，應該自動建立並回傳預設專案
    res = client.get("/api/projects")
    assert res.status_code == 200
    data = res.get_json()
    assert "active_project_id" in data
    assert len(data["projects"]) == 1
    assert data["projects"][0]["name"] == "預設專案"

    default_proj_id = data["active_project_id"]

    # 2. 建立新專案 Alpha
    res = client.post("/api/projects", json={"name": "專案 Alpha", "mode": "novice"})
    assert res.status_code == 201
    alpha_proj = res.get_json()
    assert alpha_proj["name"] == "專案 Alpha"
    alpha_id = alpha_proj["id"]

    # 3. 再次列表，活躍專案應已切換至 Alpha
    res = client.get("/api/projects")
    data = res.get_json()
    assert data["active_project_id"] == alpha_id
    assert len(data["projects"]) == 2


def _make_dummy_image_bytes() -> bytes:
    img = Image.new("RGB", (10, 10), color="red")
    buf = io.BytesIO()
    img.save(buf, format="PNG")
    return buf.getvalue()


def test_project_data_isolation(client, tmp_path):
    # 建立 Alpha 專案並選擇
    res_alpha = client.post("/api/projects", json={"name": "專案 Alpha"})
    alpha_id = res_alpha.get_json()["id"]
    client.post(f"/api/projects/{alpha_id}/select")

    # 於 Alpha 專案上傳圖片
    img_data = _make_dummy_image_bytes()
    file1 = (io.BytesIO(img_data), "alpha_img.png")
    res_up1 = client.post("/api/images", data={"files": file1})
    assert res_up1.status_code == 201

    # 檢查 Alpha 專案下有一張圖
    res_imgs_alpha = client.get("/api/images")
    imgs_alpha = res_imgs_alpha.get_json()
    assert len(imgs_alpha) == 1
    assert imgs_alpha[0]["filename"] == "alpha_img.png"
    assert imgs_alpha[0]["project_id"] == alpha_id

    # 增加一個低信心 Segments 到 Alpha 專案
    from flask import current_app
    from app.models import Segment
    repo = current_app.repo
    repo.add_segment(Segment(id="seg-alpha", image_id=imgs_alpha[0]["id"], needs_review=True))

    # 在 Alpha 專案下，review/queue 應該包含 seg-alpha，stats 總數為 1
    queue_alpha = client.get("/api/review/queue").get_json()
    assert len(queue_alpha) == 1
    stats_alpha = client.get("/api/stats").get_json()
    assert stats_alpha["total_segments"] == 1
    assert stats_alpha["need_review"] == 1

    # 建立 Beta 專案並切換
    res_beta = client.post("/api/projects", json={"name": "專案 Beta"})
    beta_id = res_beta.get_json()["id"]

    # 檢查 Beta 專案下圖片列表、審核佇列與統計皆為空（資料隔離）
    res_imgs_beta = client.get("/api/images")
    imgs_beta = res_imgs_beta.get_json()
    assert len(imgs_beta) == 0

    queue_beta = client.get("/api/review/queue").get_json()
    assert len(queue_beta) == 0

    stats_beta = client.get("/api/stats").get_json()
    assert stats_beta["total_segments"] == 0
    assert stats_beta["need_review"] == 0

    # 切換回 Alpha 專案
    client.post(f"/api/projects/{alpha_id}/select")
    res_imgs_alpha2 = client.get("/api/images")
    assert len(res_imgs_alpha2.get_json()) == 1
    assert len(client.get("/api/review/queue").get_json()) == 1


def test_rename_and_delete_project(client):
    # 建立專案
    res = client.post("/api/projects", json={"name": "待刪專案"})
    proj_id = res.get_json()["id"]

    # 重新命名
    res_ren = client.put(f"/api/projects/{proj_id}", json={"name": "更新名稱專案"})
    assert res_ren.status_code == 200
    assert res_ren.get_json()["name"] == "更新名稱專案"

    # 刪除專案
    res_del = client.delete(f"/api/projects/{proj_id}")
    assert res_del.status_code == 200
    del_data = res_del.get_json()
    assert del_data["deleted_id"] == proj_id

    # 驗證查無此專案
    res_get = client.get(f"/api/projects/{proj_id}")
    assert res_get.status_code == 404


def test_duplicate_project_name_prevention(client):
    # 1. 建立第一個專案 "Project One"
    res1 = client.post("/api/projects", json={"name": "Project One"})
    assert res1.status_code == 201

    # 2. 嘗試建立同名專案 (完全相同或大小寫不同)，應被拒絕
    res2 = client.post("/api/projects", json={"name": "Project One"})
    assert res2.status_code == 400
    assert "專案名稱已存在" in res2.get_json()["error"]

    res2_case = client.post("/api/projects", json={"name": "project one"})
    assert res2_case.status_code == 400
    assert "專案名稱已存在" in res2_case.get_json()["error"]

    # 3. 建立第二個不同名稱的專案 "Project Two"
    res3 = client.post("/api/projects", json={"name": "Project Two"})
    assert res3.status_code == 201
    proj_two_id = res3.get_json()["id"]

    # 4. 嘗試將 "Project Two" 重新命名為 "Project One"，應被拒絕
    res_rename_dupe = client.put(f"/api/projects/{proj_two_id}", json={"name": "Project One"})
    assert res_rename_dupe.status_code == 400
    assert "專案名稱已存在" in res_rename_dupe.get_json()["error"]

    # 5. 重新命名為未使用的名字，應成功
    res_rename_ok = client.put(f"/api/projects/{proj_two_id}", json={"name": "Project Two Updated"})
    assert res_rename_ok.status_code == 200
    assert res_rename_ok.get_json()["name"] == "Project Two Updated"

