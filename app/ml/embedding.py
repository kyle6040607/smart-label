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
import torch
import torchvision.transforms as T



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
    """真正的 DINOv2 特徵（凍結不訓練）。

    使用 PyTorch Hub 載入預訓練模型。
    """

    dim: int = 768

    def __init__(self, model_name: str = "facebook/dinov2-base", device: str | None = None):
        if device is None:
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        else:
            self.device = torch.device(device)

        # 把預設的 Hugging Face 模型名稱對應到 PyTorch Hub 模組名
        hub_model = "dinov2_vitb14"
        if "dinov2-small" in model_name or "vits14" in model_name:
            hub_model = "dinov2_vits14"
            self.dim = 384
        elif "dinov2-base" in model_name or "vitb14" in model_name:
            hub_model = "dinov2_vitb14"
            self.dim = 768
        else:
            raise ValueError(
                f"Unsupported DinoV2 model_name: {model_name!r} (expected dinov2-small/vits14 or dinov2-base/vitb14)"
            )

        # 載入模型
        self.model = torch.hub.load("facebookresearch/dinov2", hub_model)
        self.model.eval().to(self.device)
        self._normalize = T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])

    def encode(self, image: np.ndarray, mask: np.ndarray) -> np.ndarray:
        # 1. 取得遮罩的 bounding box
        crop, crop_mask = _crop_to_bbox(image, mask)

        # 2. 套用遮罩，將物件以外的背景去背（設為黑色）
        crop_masked = crop * (crop_mask > 0)[:, :, np.newaxis]

        # 3. 確保為 RGB 3 通道
        if crop_masked.ndim == 2:
            crop_masked = cv2.cvtColor(crop_masked, cv2.COLOR_GRAY2RGB)

        # 4. 縮放到 DinoV2 期望的 224x224（14 的倍數）
        crop_resized = cv2.resize(crop_masked, (224, 224), interpolation=cv2.INTER_AREA)

        # 5. 轉換為 PyTorch Tensor，縮放到 [0, 1] 並進行 ImageNet 常態化
        tensor = torch.from_numpy(crop_resized).permute(2, 0, 1).float() / 255.0
        tensor = self._normalize(tensor).unsqueeze(0).to(self.device)

        # 6. 推論取得特徵向量
        with torch.no_grad():
            features = self.model(tensor)

        # 7. 轉回 1D float64 numpy array，並做 L2 Normalization
        feat = features.squeeze(0).cpu().numpy().astype(np.float64)
        norm = np.linalg.norm(feat)
        return feat / norm if norm > 0 else feat


def build_embedder(use_real_embedding: bool) -> Embedder:
    if use_real_embedding:
        return DinoEmbedder()
    return MockEmbedder()
