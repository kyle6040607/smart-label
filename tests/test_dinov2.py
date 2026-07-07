from __future__ import annotations

import numpy as np
import pytest
import torch
from app.ml.embedding import DinoEmbedder


def test_dino_embedder_initialization(monkeypatch):
    """測試 DinoEmbedder 是否能成功初始化。"""
    # embedder = DinoEmbedder(model_name="facebook/dinov2-base")
    # print(f"\n[DinoEmbedder] Running model: {embedder.model_name}")

    class _DummyModel:
        def eval(self):
            return self
        def to(self, device):
            return self
        def __call__(self, x):
            return torch.ones((x.shape[0], 768))
            #return torch.zeros((x.shape[0], 768)) 正常跑請使用這個 現在都使用 假model進行測試確認流程正確 
    monkeypatch.setattr(torch.hub, "load", lambda *args, **kwargs: _DummyModel())
    embedder = DinoEmbedder(model_name="facebook/dinov2-base", device="cpu")
    assert embedder.dim == 768
    assert embedder.model is not None


def test_dino_embedder_encode(monkeypatch):
    """測試 DinoEmbedder 能否對影像與遮罩進行特徵提取，且維度與 L2 常態化皆正確。"""
    #embedder = DinoEmbedder(model_name="facebook/dinov2-base")
    #print(f"\n[DinoEmbedder] Running model: {embedder.model_name}")
    class _DummyModel:
        def eval(self):
            return self
        def to(self, device):
            return self
        def __call__(self, x):
            return torch.ones((x.shape[0], 768))
            #return torch.zeros((x.shape[0], 768)) 正常跑請使用這個 現在都使用 假model進行測試確認流程正確 
    monkeypatch.setattr(torch.hub, "load", lambda *args, **kwargs: _DummyModel())
    embedder = DinoEmbedder(model_name="facebook/dinov2-base", device="cpu")
    
    # 建立 224x224x3 的隨機影像 (BGR 格式)
    image = np.random.randint(0, 256, (224, 224, 3), dtype=np.uint8)
    
    # 建立一個圓形遮罩
    mask = np.zeros((224, 224), dtype=np.uint8)
    for y in range(40, 180):
        for x in range(40, 180):
            if (x - 110) ** 2 + (y - 110) ** 2 <= 60 ** 2:
                mask[y, x] = 255
                
    feat = embedder.encode(image, mask)
    
    # 檢查維度是否為 768 且是一維的
    assert feat.shape == (768,)
    assert feat.dtype == np.float64
    
    # 檢查是否為 L2 normalized (模長應接近 1)
    norm = np.linalg.norm(feat)
    assert pytest.approx(norm, abs=1e-5) == 1.0
