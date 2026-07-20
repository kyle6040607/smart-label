"""批量分割 job API 測試：建立、驗證、進度、單張失敗不中斷、重啟標記中斷。"""
from __future__ import annotations

import cv2
import numpy as np
import pytest
from flask import current_app

from app import create_app
from app.config import Config
from app.models import ImageRecord, SegmentJob
from app.repository import Repository


def _make_image(dir_path, name, color=(200, 50, 50)):
    """造一張帶單一色塊的測試圖（mock SAM 要能真的讀檔）。"""
    img = np.full((120, 120, 3), 30, np.uint8)
    cv2.rectangle(img, (30, 30), (90, 90), color, -1)
    p = dir_path / name
    cv2.imwrite(str(p), img)
    return str(p)


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
            # jobs API 有登入保護，直接塞 session（不走登入表單）
            with client.session_transaction() as sess:
                sess["user_id"] = "test-user"
            yield client
    app.job_runner.shutdown()


def _add_image(tmp_path, image_id, name):
    repo = current_app.repo
    path = _make_image(tmp_path / "up", name)
    return repo.add_image(ImageRecord(id=image_id, filename=name, path=path, width=120, height=120))


def test_create_job_and_complete(client, tmp_path):
    _add_image(tmp_path, "img1", "a.png")
    _add_image(tmp_path, "img2", "b.png")

    res = client.post("/api/segment_jobs", json={"image_ids": ["img1", "img2"]})
    assert res.status_code == 202
    job = res.get_json()
    assert job["status"] == "queued"
    assert job["total"] == 2

    current_app.job_runner.join(timeout=30)

    res = client.get(f"/api/segment_jobs/{job['id']}")
    assert res.status_code == 200
    done = res.get_json()
    assert done["status"] == "done"
    assert done["done"] == 2
    assert done["failed"] == []
    # 兩張圖都真的產生了片段
    assert len(current_app.repo.list_segments("img1")) >= 1
    assert len(current_app.repo.list_segments("img2")) >= 1


def test_create_job_with_prompt(client, tmp_path):
    _add_image(tmp_path, "img1", "a.png")

    res = client.post("/api/segment_jobs", json={"image_ids": ["img1"], "prompt": "cat"})
    assert res.status_code == 202
    job_id = res.get_json()["id"]

    current_app.job_runner.join(timeout=30)

    done = client.get(f"/api/segment_jobs/{job_id}").get_json()
    assert done["status"] == "done"
    segs = current_app.repo.list_segments("img1")
    assert len(segs) == 1
    assert segs[0].predicted_label == "cat"  # mock 模式的固定行為


def test_validation_happens_before_job_starts(client, tmp_path):
    # 空 image_ids → 400
    assert client.post("/api/segment_jobs", json={"image_ids": []}).status_code == 400
    assert client.post("/api/segment_jobs", json={}).status_code == 400
    # 不存在的圖 → 404
    assert client.post("/api/segment_jobs", json={"image_ids": ["nope"]}).status_code == 404
    # prompt 超長 → 400
    _add_image(tmp_path, "img1", "a.png")
    res = client.post("/api/segment_jobs", json={"image_ids": ["img1"], "prompt": "x" * 201})
    assert res.status_code == 400
    # 以上全都不該留下任何 job
    assert client.get("/api/segment_jobs").get_json() == []


def test_single_failure_does_not_abort_batch(client, tmp_path):
    _add_image(tmp_path, "good", "a.png")
    # 壞圖：路徑不存在，pipeline 讀檔會炸
    current_app.repo.add_image(ImageRecord(id="bad", filename="ghost.png", path=str(tmp_path / "up" / "ghost.png")))

    res = client.post("/api/segment_jobs", json={"image_ids": ["bad", "good"]})
    job_id = res.get_json()["id"]

    current_app.job_runner.join(timeout=30)

    done = client.get(f"/api/segment_jobs/{job_id}").get_json()
    assert done["status"] == "done"
    assert done["done"] == 2  # 兩張都算處理過（含失敗）
    assert [f["image_id"] for f in done["failed"]] == ["bad"]
    # 排在壞圖後面的好圖仍完成分割
    assert len(current_app.repo.list_segments("good")) >= 1


def test_scope_unprocessed_only_targets_images_without_segments(client, tmp_path):
    _add_image(tmp_path, "seen", "a.png")
    _add_image(tmp_path, "fresh", "b.png")
    # 先把 seen 分割掉
    client.post("/api/segment_jobs", json={"image_ids": ["seen"]})
    current_app.job_runner.join(timeout=30)

    res = client.post("/api/segment_jobs", json={"scope": "unprocessed"})
    assert res.status_code == 202
    assert res.get_json()["image_ids"] == ["fresh"]
    current_app.job_runner.join(timeout=30)

    # 全部都處理過之後再送 → 400
    res = client.post("/api/segment_jobs", json={"scope": "unprocessed"})
    assert res.status_code == 400
    # scope 亂給 → 400
    assert client.post("/api/segment_jobs", json={"scope": "everything"}).status_code == 400


def test_list_jobs_newest_first(client, tmp_path):
    _add_image(tmp_path, "img1", "a.png")
    id1 = client.post("/api/segment_jobs", json={"image_ids": ["img1"]}).get_json()["id"]
    current_app.job_runner.join(timeout=30)
    id2 = client.post("/api/segment_jobs", json={"image_ids": ["img1"]}).get_json()["id"]
    current_app.job_runner.join(timeout=30)

    jobs = client.get("/api/segment_jobs").get_json()
    assert [j["id"] for j in jobs] == [id2, id1]
    assert client.get("/api/segment_jobs/missing").status_code == 404


def test_api_requires_login(client, tmp_path):
    """未登入 → 三個端點都要回 401，不能開工。"""
    with client.session_transaction() as sess:
        sess.clear()
    assert client.post("/api/segment_jobs", json={"scope": "all"}).status_code == 401
    assert client.get("/api/segment_jobs").status_code == 401
    assert client.get("/api/segment_jobs/whatever").status_code == 401


def test_reject_new_job_while_one_is_active(client, tmp_path):
    """已有 queued/running 的 job → 409，不讓多個 job 疊在唯一的 worker 上。"""
    _add_image(tmp_path, "img1", "a.png")
    repo = current_app.repo
    # 直接塞一個 running job（不經 runner），時序完全可控
    repo.add_job(SegmentJob(id="busy", image_ids=["img1"], status="running"))

    res = client.post("/api/segment_jobs", json={"image_ids": ["img1"]})
    assert res.status_code == 409

    # busy 結束後就能再送
    busy = repo.get_job("busy")
    busy.status = "done"
    repo.update_job(busy)
    assert client.post("/api/segment_jobs", json={"image_ids": ["img1"]}).status_code == 202
    current_app.job_runner.join(timeout=30)


def test_unexpected_error_marks_job_interrupted(client, tmp_path, monkeypatch):
    """迴圈外的非預期錯誤（如存檔失敗）不能讓 job 卡在 running。"""
    _add_image(tmp_path, "img1", "a.png")
    repo = current_app.repo

    real_update = repo.update_job
    calls = {"n": 0}

    def flaky_update(job):
        calls["n"] += 1
        if calls["n"] == 1:  # 第一次（寫 running 狀態）就炸
            raise RuntimeError("模擬存檔失敗")
        return real_update(job)

    monkeypatch.setattr(repo, "update_job", flaky_update)
    res = client.post("/api/segment_jobs", json={"image_ids": ["img1"]})
    job_id = res.get_json()["id"]

    current_app.job_runner.join(timeout=30)
    assert repo.get_job(job_id).status == "interrupted"


def test_futures_cleaned_up_after_completion(client, tmp_path):
    """future 完成後要從 _futures 移除，長時間執行不累積。"""
    _add_image(tmp_path, "img1", "a.png")
    client.post("/api/segment_jobs", json={"image_ids": ["img1"]})
    runner = current_app.job_runner
    runner.join(timeout=30)
    assert runner._futures == {}


def test_unfinished_jobs_marked_interrupted_after_restart(tmp_path):
    """模擬伺服器重啟：存檔裡殘留 running 的 job，重載後要標成 interrupted。"""
    db = tmp_path / "store.json"
    repo = Repository(db)
    repo.add_job(SegmentJob(id="j1", image_ids=["x"], status="running"))
    repo.add_job(SegmentJob(id="j2", image_ids=["x"], status="done", done=1))

    reloaded = Repository(db)
    assert reloaded.get_job("j1").status == "interrupted"
    assert reloaded.get_job("j2").status == "done"
