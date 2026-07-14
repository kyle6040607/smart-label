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

    def segment_by_box(self, image: np.ndarray, bbox: list[float]) -> MaskDict:
        """在給定的 Bounding Box 內切出單一物件。"""
        ...


def _to_mask_dict(mask: np.ndarray) -> MaskDict:
    mask = (mask > 0).astype(np.uint8)
    ys, xs = np.where(mask)
    if len(xs) == 0:
        return MaskDict(mask=mask, bbox=(0, 0, 0, 0), area=0)
    x, y, w, h = int(xs.min()), int(ys.min()), int(xs.max() - xs.min() + 1), int(ys.max() - ys.min() + 1)
    return MaskDict(mask=mask, bbox=(x, y, w, h), area=int(mask.sum()))


def _resize_if_needed(image: np.ndarray, max_side: int = 1024) -> tuple[np.ndarray, float, bool, int, int]:
    """若影像長邊大於 max_side，則等比例縮小。

    回傳: (處理後的影像, 縮放比例, 是否進行了縮放, 原始高度, 原始寬度)
    """
    try:
        orig_h, orig_w = image.shape[:2]
        need_resize = max(orig_h, orig_w) > max_side
        if need_resize:
            scale = max_side / max(orig_h, orig_w)
            new_w, new_h = int(orig_w * scale), int(orig_h * scale)
            # 防範怪圖導致寬高為 0
            new_w = max(1, new_w)
            new_h = max(1, new_h)
            resized_img = cv2.resize(image, (new_w, new_h), interpolation=cv2.INTER_AREA)
            return resized_img, scale, True, orig_h, orig_w
        return image, 1.0, False, orig_h, orig_w
    except Exception as e:
        raise ValueError(f"影像等比例縮放失敗 ({e})，已略過此影像。")


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

    def segment_by_box(self, image: np.ndarray, bbox: list[float]) -> MaskDict:
        """以 BBox 當作遮罩，模擬 SAM 的框提示。"""
        h, w = image.shape[:2]
        x1, y1, x2, y2 = map(int, bbox)
        mask = np.zeros((h, w), dtype=np.uint8)
        mask[max(0, y1):min(h, y2), max(0, x1):min(w, x2)] = 1
        return _to_mask_dict(mask)


class SamSegmenter:
    def __init__(
        self,
        checkpoint: str,
        model_type: str = "vit_t",
        device: str = "cuda",
        points_per_side: int = 32,
        min_mask_region_area: int = 100,
    ):
        import torch
        from mobile_sam import sam_model_registry, SamAutomaticMaskGenerator, SamPredictor

        # 檢查 GPU 是否可用，若指定 cuda 但不可用，則降級為 cpu
        if torch.cuda.is_available():
            self.device = "cuda"
            print("使用 GPU 訓練")
        else:
            self.device = "cpu"
            print("GPU 不可用，改用 CPU")

        if model_type not in sam_model_registry:
            supported = ", ".join(sorted(sam_model_registry))
            raise ValueError(
                f"不支援的 SAM_MODEL_TYPE: {model_type!r}；可用類型: {supported}"
            )

        print(f"--- MobileSAM 初始化中 ---")
        print(f"指定架構: {model_type}")
        print(f"配置設備: {self.device}")

        self.sam = sam_model_registry[model_type](checkpoint=checkpoint)
        self.sam.to(device=self.device)
        self.sam.eval()

        self.auto = SamAutomaticMaskGenerator(
            self.sam,
            points_per_side=points_per_side,
            min_mask_region_area=min_mask_region_area,
        )
        self.predictor = SamPredictor(self.sam)

        print(f"MobileSAM 載入成功，已就緒！")

    def segment(self, image: np.ndarray) -> list[MaskDict]:
        try:
            infer_image, _, need_resize, orig_h, orig_w = _resize_if_needed(image)
        except ValueError as e:
            print(f"警告：{e}")
            return []  # 發生錯誤，略過此張圖，直接回傳空遮罩列表

        raw_masks = self.auto.generate(infer_image)
        masks: list[MaskDict] = []
        for rm in raw_masks:
            mask = rm["segmentation"].astype(np.uint8)
            # 若有進行縮放，需將 mask 用最近鄰插值法 resize 回原圖尺寸
            if need_resize:
                mask = cv2.resize(mask, (orig_w, orig_h), interpolation=cv2.INTER_NEAREST)
            
            md = _to_mask_dict(mask)
            # 防呆：只加入大於 0 像素的有效遮罩
            if md["area"] > 0:
                masks.append(md)
        return masks

    def segment_at(self, image: np.ndarray, point: tuple[int, int]) -> MaskDict:
        orig_h, orig_w = image.shape[:2]
        x, y = point

        try:
            infer_image, scale, need_resize, _, _ = _resize_if_needed(image)
        except ValueError as e:
            print(f"警告：{e}")
            # 發生錯誤，略過此張圖，直接回傳空遮罩
            return _to_mask_dict(np.zeros((orig_h, orig_w), dtype=np.uint8))

        if need_resize:
            x, y = int(x * scale), int(y * scale)

        self.predictor.set_image(infer_image)
        input_point = np.array([[x, y]])
        input_label = np.array([1])

        masks, scores, logits = self.predictor.predict(
            point_coords=input_point,
            point_labels=input_label,
            multimask_output=True,
        )

        # 挑選分數最高的遮罩
        best_idx = np.argmax(scores)
        best_mask = masks[best_idx].astype(np.uint8)

        # 防止空遮罩：若預測出的遮罩像素和為 0，則拋出異常
        if best_mask.sum() == 0:
            raise ValueError("SAM 在此點擊位置沒有找到任何物件。")

        # 若有進行縮放，需將 mask 用最近鄰插值法 resize 回原圖尺寸
        if need_resize:
            best_mask = cv2.resize(best_mask, (orig_w, orig_h), interpolation=cv2.INTER_NEAREST)

        return _to_mask_dict(best_mask)

    def segment_by_box(self, image: np.ndarray, bbox: list[float]) -> MaskDict:
        orig_h, orig_w = image.shape[:2]
        x1, y1, x2, y2 = bbox

        try:
            infer_image, scale, need_resize, _, _ = _resize_if_needed(image)
        except ValueError as e:
            print(f"警告：{e}")
            return _to_mask_dict(np.zeros((orig_h, orig_w), dtype=np.uint8))

        if need_resize:
            x1, y1, x2, y2 = x1 * scale, y1 * scale, x2 * scale, y2 * scale

        self.predictor.set_image(infer_image)
        input_box = np.array([x1, y1, x2, y2])

        masks, scores, logits = self.predictor.predict(
            box=input_box,
            multimask_output=False,
        )

        best_mask = masks[0].astype(np.uint8)

        if best_mask.sum() == 0:
            raise ValueError("SAM 在此框選範圍內沒有找到任何物件。")

        if need_resize:
            best_mask = cv2.resize(best_mask, (orig_w, orig_h), interpolation=cv2.INTER_NEAREST)

        return _to_mask_dict(best_mask)


def build_segmenter(
    use_real_sam: bool,
    *,
    max_masks: int = 12,
    min_area_ratio: float = 0.004,
    flood_tol: int = 12,
    checkpoint: str = "models/mobile_sam.pt",
    model_type: str = "vit_t",
    points_per_side: int = 32,
    min_mask_region_area: int = 100,
) -> Segmenter:
    if use_real_sam:
        return SamSegmenter(
            checkpoint=checkpoint,
            model_type=model_type,
            points_per_side=points_per_side,
            min_mask_region_area=min_mask_region_area,
        )
    return MockSegmenter(max_masks, min_area_ratio, flood_tol)
