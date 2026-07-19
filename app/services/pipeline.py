"""人機協作標記 pipeline。

把四個模組串起來：
  SAM 切割 → DINOv2 取特徵 → few-shot 分類 → 算信心、決定是否送審。

也負責主動學習迴圈：人標了新範例 → 重訓分類器 → 重新預測未審片段。
這層是 API 與各 AI 模組之間的唯一橋樑，方便日後抽換實作。
"""
from __future__ import annotations

import cv2
import numpy as np
from typing import Callable

from app.config import Config
from app.ml.active_learning import confidence_score, needs_review
from app.ml.classifier import FewShotClassifier
from app.ml.embedding import build_embedder
from app.ml.sam import build_segmenter
from app.models import ImageRecord, LabelExample, Segment
from app.repository import Repository
from app.utils import imread, imwrite
from app.ml.yolo_world import YoloWorldDetector


class Pipeline:
    def __init__(self, config: Config, repo: Repository):
        self.config = config
        self.repo = repo
        self.yolo_detector = None
        self.segmenter = build_segmenter(
            config.use_real_sam,
            max_masks=config.sam_max_masks,
            min_area_ratio=config.sam_min_area_ratio,
            flood_tol=config.sam_flood_tol,
            checkpoint=config.sam_checkpoint,
            model_type=config.sam_model_type,
            points_per_side=config.sam_points_per_side,
            min_mask_region_area=config.sam_min_mask_region_area,
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

    # ---------- 自動分割整張圖（提案 demo 第 1 步：自動分割）----------
    def segment_image(self, image: ImageRecord, progress_callback: Callable[[dict], None] | None = None) -> list[Segment]:
        if progress_callback:
            progress_callback({"event": "progress", "stage": "segmenting", "progress": 10, "message": "正在執行影像自動切割偵測..."})

        img = self._read_rgb(image.path)
        masks = self.segmenter.segment(img)

        # 取得此圖片目前資料庫中已有的所有片段
        existing_segs = self.repo.list_segments(image.id)

        segments: list[Segment] = []
        total_masks = len(masks)
        for i, md in enumerate(masks):
            if progress_callback:
                progress_val = 75 + int((i / max(total_masks, 1)) * 20)
                progress_callback({
                    "event": "progress",
                    "stage": "classifying",
                    "progress": progress_val,
                    "message": f"正在分類及儲存區塊 ({i + 1}/{total_masks})..."
                })

            bbox = tuple(md["bbox"])

            # 檢查是否已存在相同或極為相近邊界框的區塊（容差 2 像素）
            matched_seg = None
            for ex in existing_segs:
                if (abs(ex.bbox[0] - bbox[0]) <= 2 and
                    abs(ex.bbox[1] - bbox[1]) <= 2 and
                    abs(ex.bbox[2] - bbox[2]) <= 2 and
                    abs(ex.bbox[3] - bbox[3]) <= 2):
                    matched_seg = ex
                    break

            if matched_seg is not None:
                # 已存在相同的區塊，保留既有資料（避免覆蓋已標記成果）
                segments.append(matched_seg)
            else:
                # 缺失的區塊，重新建立、分類並存檔
                seg = Segment(image_id=image.id, bbox=bbox, area=md["area"])
                seg.mask_path = self._save_mask(image.id, seg.id, md["mask"])
                self._classify_segment(img, seg, md["mask"])
                self.repo.add_segment(seg)
                segments.append(seg)
        
        if progress_callback:
            progress_callback({"event": "progress", "stage": "done", "progress": 100, "message": "自動分割完成！"})
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

    # ---------- 自然語言：用文字找物件並切出遮罩（Week 2）----------
    def segment_text(self, image: ImageRecord, prompt: str, progress_callback: Callable[[dict], None] | None = None) -> list[Segment]:
        """依文字提示分割圖片中的物件。

        - prompt 會先移除前後空白，不可為空，最多 200 個字元。
        - 回傳零到多個已完成遮罩存檔、分類與 Repository 寫入的 Segment。
        - 找不到符合物件時回傳空列表，不視為錯誤。
        - prompt 只是搜尋條件，不直接作為 human_label。
        """
        prompt = prompt.strip()
        if not prompt:
            raise ValueError("prompt 不可為空")
        if len(prompt) > 200:
            raise ValueError("prompt 不可超過 200 個字元")

        if progress_callback:
            progress_callback({"event": "progress", "stage": "detecting", "progress": 10, "message": "正在執行物件偵測中..."})

        img = self._read_rgb(image.path)

        # 💡 Mock 測試模式：不需真模型，快速模擬 YOLO-World + SAM 的分割返回，使 pytest 單元測試可秒級通過
        if not self.config.use_real_sam:
            h, w = img.shape[:2]
            # 建立一個 100x100 的模擬遮罩
            mask = np.zeros((h, w), dtype=np.uint8)
            x1, y1 = max(0, w // 2 - 50), max(0, h // 2 - 50)
            x2, y2 = min(w, w // 2 + 50), min(h, h // 2 + 50)
            mask[y1:y2, x1:x2] = 1
            
            seg = Segment(image_id=image.id, bbox=(x1, y1, x2 - x1, y2 - y1), area=int(mask.sum()))
            seg.mask_path = self._save_mask(image.id, seg.id, mask)

            seg.predicted_label = prompt
            seg.probs = {prompt: 1.0}
            seg.confidence = 0.88
            seg.needs_review = needs_review(seg.confidence, self.config.confidence_threshold)

            self.repo.add_segment(seg)
            if progress_callback:
                progress_callback({"event": "progress", "stage": "done", "progress": 100, "message": "文字分割完成！"})
            return [seg]

        # 動態載入 YOLO-World 偵測器 (指向已下載大模型)
        if self.yolo_detector is None:
            model_path = str(self.config.base_dir / "models" / "yolov8x-worldv2.pt")
            self.yolo_detector = YoloWorldDetector(model_path)

        # 1. 呼叫 YOLO-World 找出所有符合文字的 bounding boxes
        boxes = self.yolo_detector.predict_boxes(img, prompt, device=self.segmenter.device)
        
        segments: list[Segment] = []
        total_boxes = len(boxes)

        # 2. 逐一將框餵給 SAM 做分割
        for i, bbox in enumerate(boxes):
            if progress_callback:
                progress_val = 75 + int((i / max(total_boxes, 1)) * 20)
                progress_callback({
                    "event": "progress",
                    "stage": "segmenting",
                    "progress": progress_val,
                    "message": f"正在進行物件分割 ({i + 1}/{total_boxes})..."
                })
            try:
                # 呼叫 SAM 預測遮罩
                md = self.segmenter.segment_by_box(img, bbox)
                
                # 打包 Segment，跑特徵分類並存入資料庫
                seg = Segment(image_id=image.id, bbox=tuple(md["bbox"]), area=md["area"])
                seg.mask_path = self._save_mask(image.id, seg.id, md["mask"])
                self._classify_segment(img, seg, md["mask"])
                self.repo.add_segment(seg)
                segments.append(seg)
            except Exception as e:
                print(f"警告：YOLO Box 進行 SAM 分割失敗: {e}")
                continue

        return segments

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
