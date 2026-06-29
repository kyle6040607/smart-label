"""特徵抽取

把「SAM 切出來的物件」轉成特徵向量，餵給後面的小分類器。
重點：用凍結的預訓練模型（DINOv2 / CLIP）抽特徵，不訓練它——
難的部分別人做完了，我們只訓練最後一小層。

- MockEmbedder：用顏色直方圖 + HOG-ish 統計當特徵，免下載模型即可跑。
- DinoEmbedder：真正接 DINOv2 / CLIP（USE_REAL_EMBEDDING=1 時啟用）。
"""
from __future__ import annotations

from typing import Protocol

import cv2
import numpy as np


class Embedder(Protocol):
    dim: int

    def encode(self, image: np.ndarray, mask: np.ndarray) -> np.ndarray:
        """把 image 中 mask 圈起來的物件編成一維特徵向量。"""
        ...


def _crop_to_bbox(image: np.ndarray, mask: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    ys, xs = np.where(mask > 0)
    if len(xs) == 0:
        return image, mask
    x0, x1, y0, y1 = xs.min(), xs.max() + 1, ys.min(), ys.max() + 1
    return image[y0:y1, x0:x1], mask[y0:y1, x0:x1]


class MockEmbedder:
    """免模型的替身：用顏色直方圖 + 形狀統計組特徵向量。

    維度固定、可重現，足以讓 few-shot 分類器跑出有意義的相對信心。
    最終會被 DinoEmbedder 取代。
    """

    dim: int = 50  # 16*3 顏色 + 2 形狀

    def encode(self, image: np.ndarray, mask: np.ndarray) -> np.ndarray:
        crop, m = _crop_to_bbox(image, mask)
        if crop.ndim == 2:
            crop = cv2.cvtColor(crop, cv2.COLOR_GRAY2RGB)
        m3 = (m > 0)

        feats: list[float] = []
        # 每個通道 16-bin 直方圖（只統計遮罩內像素）
        for c in range(3):
            chan = crop[:, :, c][m3]
            hist, _ = np.histogram(chan, bins=16, range=(0, 256))
            feats.extend(hist.astype(np.float64))
        # 形狀：面積占比、長寬比
        area = float(m3.sum())
        h, w = m.shape[:2]
        feats.append(area / (h * w + 1e-6))
        feats.append(w / (h + 1e-6))

        v = np.asarray(feats, dtype=np.float64)
        n = np.linalg.norm(v)
        return v / n if n > 0 else v  # L2 normalize，方便做 cosine/kNN


class DinoEmbedder:
    """真正的 DINOv2 / CLIP 特徵（提案第 6 頁，凍結不訓練）。

        uv add torch torchvision transformers
    這裡先留骨架，第 2 週接上即可（USE_REAL_EMBEDDING=1）。
    """

    dim: int = 768

    def __init__(self, model_name: str = "facebook/dinov2-base", device: str = "cpu"):
        # TODO: 載入凍結的 DINOv2 / CLIP
        # from transformers import AutoImageProcessor, AutoModel
        # self.proc = AutoImageProcessor.from_pretrained(model_name)
        # self.model = AutoModel.from_pretrained(model_name).eval().to(device)
        raise NotImplementedError(
            "DinoEmbedder 尚未接上——先用 MockEmbedder，第 2 週補完特徵抽取。"
        )

    def encode(self, image: np.ndarray, mask: np.ndarray) -> np.ndarray:  # pragma: no cover
        raise NotImplementedError


def build_embedder(use_real_embedding: bool) -> Embedder:
    if use_real_embedding:
        return DinoEmbedder()
    return MockEmbedder()
