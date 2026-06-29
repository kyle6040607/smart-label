"""SAM 切割

定義一個 Segmenter 介面：吃一張圖，吐出一堆遮罩。
- MockSegmenter：用 OpenCV 做簡單的分水嶺/連通元件，免下載模型即可跑通整條流程。
- SamSegmenter：真正接 Meta 的 segment-anything（USE_REAL_SAM=1 時啟用）。

把介面與實作分開，團隊第 1 週先用 mock 把上傳→點選→出遮罩跑起來，
之後把 SamSegmenter 補完即可，上層完全不用改。
"""
from __future__ import annotations

from typing import Protocol

import cv2
import numpy as np


class MaskDict(dict):
    """一塊遮罩的描述。

    keys: mask (np.ndarray bool/uint8), bbox (x,y,w,h), area (int)
    """


class Segmenter(Protocol):
    def segment(self, image: np.ndarray) -> list[MaskDict]:
        """對整張圖做自動切割，回傳多塊遮罩。"""
        ...

    def segment_at(self, image: np.ndarray, point: tuple[int, int]) -> MaskDict:
        """在使用者點擊的座標切出單一物件（互動式提示）。"""
        ...


def _to_mask_dict(mask: np.ndarray) -> MaskDict:
    mask = (mask > 0).astype(np.uint8)
    ys, xs = np.where(mask)
    if len(xs) == 0:
        return MaskDict(mask=mask, bbox=(0, 0, 0, 0), area=0)
    x, y, w, h = int(xs.min()), int(ys.min()), int(xs.max() - xs.min() + 1), int(ys.max() - ys.min() + 1)
    return MaskDict(mask=mask, bbox=(x, y, w, h), area=int(mask.sum()))


class MockSegmenter:
    """免模型的替身：用傳統影像處理切塊，讓流程立刻能跑。

    這不是最終品質，只是把 pipeline 接通用的 placeholder。
    """

    def __init__(self, max_masks: int = 12, min_area_ratio: float = 0.004, flood_tol: int = 12):
        self.max_masks = max_masks
        self.min_area_ratio = min_area_ratio
        self.flood_tol = flood_tol

    def segment(self, image: np.ndarray) -> list[MaskDict]:
        gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY) if image.ndim == 3 else image
        h, w = gray.shape[:2]
        min_area = int(self.min_area_ratio * h * w)

        # 模糊 + Otsu + 形態學，產生粗略前景區塊
        blur = cv2.GaussianBlur(gray, (5, 5), 0)
        _, th = cv2.threshold(blur, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        kernel = np.ones((3, 3), np.uint8)
        opened = cv2.morphologyEx(th, cv2.MORPH_OPEN, kernel, iterations=2)

        n, labels, statsm, _ = cv2.connectedComponentsWithStats(opened)
        masks: list[tuple[int, MaskDict]] = []
        for i in range(1, n):  # 0 是背景
            area = int(statsm[i, cv2.CC_STAT_AREA])
            if area < min_area:
                continue
            masks.append((area, _to_mask_dict((labels == i).astype(np.uint8))))

        masks.sort(key=lambda t: t[0], reverse=True)
        return [m for _, m in masks[: self.max_masks]]

    def segment_at(self, image: np.ndarray, point: tuple[int, int]) -> MaskDict:
        """以點擊點做 floodFill，模擬 SAM 的點提示。"""
        h, w = image.shape[:2]
        flood = np.zeros((h + 2, w + 2), np.uint8)
        img = image.copy()
        x, y = point
        t = (self.flood_tol,) * 3
        cv2.floodFill(img, flood, (x, y), (255, 255, 255), t, t,
                      flags=cv2.FLOODFILL_FIXED_RANGE)
        mask = flood[1:-1, 1:-1]
        return _to_mask_dict(mask)


class SamSegmenter:
    """真正的 SAM（提案第 5 頁）。

    需要安裝 segment-anything 與下載 checkpoint：
        uv add segment-anything torch torchvision
        # 下載 sam_vit_h_4b8939.pth 放到 models/
    這裡先留骨架，第 1 週把 TODO 補完即可啟用（USE_REAL_SAM=1）。
    """

    def __init__(self, checkpoint: str, model_type: str = "vit_h", device: str = "cpu"):
        # TODO: 載入 SAM
        # from segment_anything import sam_model_registry, SamAutomaticMaskGenerator, SamPredictor
        # sam = sam_model_registry[model_type](checkpoint=checkpoint).to(device)
        # self.auto = SamAutomaticMaskGenerator(sam)
        # self.predictor = SamPredictor(sam)
        raise NotImplementedError(
            "SamSegmenter 尚未接上——先用 MockSegmenter，第 1 週補完模型載入。"
        )

    def segment(self, image: np.ndarray) -> list[MaskDict]:  # pragma: no cover
        raise NotImplementedError

    def segment_at(self, image: np.ndarray, point: tuple[int, int]) -> MaskDict:  # pragma: no cover
        raise NotImplementedError


def build_segmenter(
    use_real_sam: bool,
    *,
    max_masks: int = 12,
    min_area_ratio: float = 0.004,
    flood_tol: int = 12,
    checkpoint: str = "models/sam_vit_h_4b8939.pth",
) -> Segmenter:
    if use_real_sam:
        return SamSegmenter(checkpoint=checkpoint)
    return MockSegmenter(max_masks, min_area_ratio, flood_tol)
