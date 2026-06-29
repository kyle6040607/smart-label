"""人機協作標記 pipeline。

把四個模組串起來：
  SAM 切割 → DINOv2 取特徵 → few-shot 分類 → 算信心、決定是否送審。

也負責主動學習迴圈：人標了新範例 → 重訓分類器 → 重新預測未審片段。
這層是 API 與各 AI 模組之間的唯一橋樑，方便日後抽換實作。
"""
from __future__ import annotations

import cv2
import numpy as np

from app.config import Config
from app.ml.active_learning import confidence_score, needs_review
from app.ml.classifier import FewShotClassifier
from app.ml.embedding import build_embedder
from app.ml.sam import build_segmenter
from app.models import ImageRecord, LabelExample, Segment
from app.repository import Repository
from app.utils import imread, imwrite


class Pipeline:
    def __init__(self, config: Config, repo: Repository):
        self.config = config
        self.repo = repo
        self.segmenter = build_segmenter(
            config.use_real_sam,
            max_masks=config.sam_max_masks,
            min_area_ratio=config.sam_min_area_ratio,
            flood_tol=config.sam_flood_tol,
            checkpoint=config.sam_checkpoint,
        )
        self.embedder = build_embedder(config.use_real_embedding)
        self.classifier = FewShotClassifier(
            kind=config.classifier_kind, k=config.knn_k, temperature=config.softmax_temperature
        )
        self.refit()

    # ---------- 影像 IO ----------
    @staticmethod
    def _read_rgb(path: str) -> np.ndarray:
        bgr = imread(path, cv2.IMREAD_COLOR)
        if bgr is None:
            raise FileNotFoundError(path)
        return cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)

    def _save_mask(self, image_id: str, seg_id: str, mask: np.ndarray) -> str:
        out = self.config.mask_dir / f"{image_id}_{seg_id}.png"
        imwrite(str(out), (mask > 0).astype(np.uint8) * 255)
        return str(out)

    # ---------- 自動分割整張圖（提案第 3 頁第 2 步）----------
    def segment_image(self, image: ImageRecord) -> list[Segment]:
        img = self._read_rgb(image.path)
        masks = self.segmenter.segment(img)
        segments: list[Segment] = []
        for md in masks:
            seg = Segment(image_id=image.id, bbox=tuple(md["bbox"]), area=md["area"])
            seg.mask_path = self._save_mask(image.id, seg.id, md["mask"])
            self._classify_segment(img, seg, md["mask"])
            self.repo.add_segment(seg)
            segments.append(seg)
        return segments

    # ---------- 互動式：使用者點一下切一塊 ----------
    def segment_point(self, image: ImageRecord, point: tuple[int, int]) -> Segment:
        img = self._read_rgb(image.path)
        md = self.segmenter.segment_at(img, point)
        seg = Segment(image_id=image.id, bbox=tuple(md["bbox"]), area=md["area"])
        seg.mask_path = self._save_mask(image.id, seg.id, md["mask"])
        self._classify_segment(img, seg, md["mask"])
        self.repo.add_segment(seg)
        return seg

    # ---------- 手動描邊：使用者沿物件邊界畫出多邊形 ----------
    def segment_polygon(self, image: ImageRecord, points: list[tuple[int, int]]) -> Segment:
        """把使用者手繪的邊界點轉成精準遮罩。

        用於標種子範例或修正 mock/SAM 切歪的區塊——人決定邊界，最準。
        """
        img = self._read_rgb(image.path)
        h, w = img.shape[:2]
        mask = np.zeros((h, w), np.uint8)
        pts = np.array([points], dtype=np.int32)
        cv2.fillPoly(mask, pts, 255)

        ys, xs = np.where(mask > 0)
        if len(xs) == 0:
            raise ValueError("描邊區域是空的（至少需要 3 個點）")
        bbox = (int(xs.min()), int(ys.min()),
                int(xs.max() - xs.min() + 1), int(ys.max() - ys.min() + 1))
        seg = Segment(image_id=image.id, bbox=bbox, area=int((mask > 0).sum()))
        seg.mask_path = self._save_mask(image.id, seg.id, mask)
        self._classify_segment(img, seg, mask)
        self.repo.add_segment(seg)
        return seg

    # ---------- 對單一片段做分類 + 信心判斷 ----------
    def _classify_segment(self, img: np.ndarray, seg: Segment, mask: np.ndarray) -> None:
        feat = self.embedder.encode(img, mask)
        if self.classifier.ready:
            probs = self.classifier.predict(feat)
            seg.probs = probs
            seg.predicted_label = max(probs, key=probs.get) if probs else None
            seg.confidence = confidence_score(probs, self.config.confidence_strategy)
            seg.needs_review = needs_review(seg.confidence, self.config.confidence_threshold)
        else:
            # 還沒有足夠範例 → 一律送審，請人先標種子
            seg.probs, seg.predicted_label, seg.confidence, seg.needs_review = {}, None, 0.0, True

    # ---------- 把某片段存成 few-shot 種子範例（提案第 3 頁第 1 步）----------
    def add_example_from_segment(self, seg: Segment, label: str) -> LabelExample:
        img = self._read_rgb(self.repo.get_image(seg.image_id).path)
        mask = imread(seg.mask_path, cv2.IMREAD_GRAYSCALE)
        feat = self.embedder.encode(img, mask)
        ex = LabelExample(label=label, feature=feat.tolist(), source_segment_id=seg.id)
        self.repo.add_example(ex)

        # 人也順手把這片段標好
        seg.human_label = label
        seg.reviewed = True
        seg.needs_review = False
        self.repo.update_segment(seg)

        # 主動學習迴圈：回訓 + 重新預測未審片段
        self.refit()
        self.reclassify_pending()
        return ex

    # ---------- 刪掉標錯的類別（連帶回訓）----------
    def delete_label(self, label: str) -> int:
        n = self.repo.delete_label(label)
        self.refit()
        self.reclassify_pending()
        return n

    # ---------- 重建分類器 ----------
    def refit(self) -> None:
        self.classifier.fit(self.repo.list_examples())

    # ---------- 回訓後重新預測尚未人工審核的片段 ----------
    def reclassify_pending(self) -> None:
        cache: dict[str, np.ndarray] = {}
        for seg in self.repo.list_segments():
            if seg.reviewed:
                continue
            if not self.classifier.ready:
                # 範例被刪光、分類器失效 → 清掉舊預測，退回送審（別殘留 stale label）
                seg.probs, seg.predicted_label, seg.confidence, seg.needs_review = {}, None, 0.0, True
                self.repo.update_segment(seg)
                continue
            if seg.image_id not in cache:
                cache[seg.image_id] = self._read_rgb(self.repo.get_image(seg.image_id).path)
            mask = imread(seg.mask_path, cv2.IMREAD_GRAYSCALE)
            self._classify_segment(cache[seg.image_id], seg, mask)
            self.repo.update_segment(seg)
