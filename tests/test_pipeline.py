"""端到端冒煙測試：不需真模型，驗證整條人機協作流程跑得通。"""
from __future__ import annotations

import numpy as np
import cv2

from app.config import Config
from app.services.pipeline import Pipeline
from app.models import ImageRecord
from app.repository import Repository


def _make_image(tmp_path, name, color):
    """造一張帶單一色塊的測試圖。"""
    img = np.full((120, 120, 3), 30, np.uint8)
    cv2.rectangle(img, (30, 30), (90, 90), color, -1)
    p = tmp_path / name
    cv2.imwrite(str(p), img)
    return str(p)


def _pipeline(tmp_path) -> tuple[Pipeline, Repository, Config]:
    cfg = Config(
        base_dir=tmp_path, data_dir=tmp_path, upload_dir=tmp_path / "up",
        mask_dir=tmp_path / "mask", db_file=tmp_path / "store.json",
    )
    cfg.ensure_dirs()
    repo = Repository(cfg.db_file)
    return Pipeline(cfg, repo), repo, cfg


def test_segment_produces_masks(tmp_path):
    pipe, repo, cfg = _pipeline(tmp_path)
    path = _make_image(cfg.upload_dir, "a.png", (200, 50, 50))
    img = repo.add_image(ImageRecord(filename="a.png", path=path, width=120, height=120))
    segs = pipe.segment_image(img)
    assert len(segs) >= 1
    # 還沒範例 → 全部送審
    assert all(s.needs_review for s in segs)


def test_active_learning_loop(tmp_path):
    pipe, repo, cfg = _pipeline(tmp_path)
    # 兩張不同顏色的圖，各自切塊
    red = repo.add_image(ImageRecord(filename="r.png",
        path=_make_image(cfg.upload_dir, "r.png", (200, 40, 40)), width=120, height=120))
    blue = repo.add_image(ImageRecord(filename="b.png",
        path=_make_image(cfg.upload_dir, "b.png", (40, 40, 200)), width=120, height=120))
    rseg = pipe.segment_image(red)[0]
    bseg = pipe.segment_image(blue)[0]

    # 標兩個種子 → 觸發回訓
    pipe.add_example_from_segment(rseg, "red")
    pipe.add_example_from_segment(bseg, "blue")

    # 分類器就緒，統計有自動接受比例
    assert pipe.classifier.ready
    stats = repo.stats()
    assert stats["num_labels"] == 2
    assert 0.0 <= stats["auto_ratio"] <= 1.0

    # 刪掉建錯的類別：範例消失、用它標的片段退回送審、分類器不再就緒
    assert rseg.human_label == "red" and rseg.reviewed
    n = pipe.delete_label("red")
    assert n == 1
    assert repo.labels() == ["blue"]
    assert not pipe.classifier.ready  # 只剩一個類別
    again = repo.get_segment(rseg.id)
    assert again.human_label is None and not again.reviewed and again.needs_review
    # 未審片段的舊預測也要被清掉，不能殘留刪掉的類別
    assert again.predicted_label is None and again.probs == {}
    # 刪不存在的類別 → 0
    assert pipe.delete_label("nope") == 0
