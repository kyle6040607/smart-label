"""人機協作標記 pipeline。

把四個模組串起來：
  SAM 切割 → DINOv2 取特徵 → few-shot 分類 → 算信心、決定是否送審。

也負責主動學習迴圈：人標了新範例 → 重訓分類器 → 重新預測未審片段。
這層是 API 與各 AI 模組之間的唯一橋樑，方便日後抽換實作。

分類器是「每個使用者一顆」，彼此獨立不互相共用（見 classifiers）。
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
from app.services.gemini import GeminiService


class Pipeline:
    def __init__(self, config: Config, repo: Repository):
        self.config = config
        self.repo = repo
        self.yolo_detector = None
        self.gemini_service = GeminiService(config.gemini_api_key)
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
        # 每個使用者各自一顆分類器，惰性建立（第一次用到才 fit）
        self.classifiers: dict[str, FewShotClassifier] = {}

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
        classifier = self.get_classifier(image.owner_id)

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
                seg = Segment(image_id=image.id, owner_id=image.owner_id, bbox=bbox, area=md["area"])
                seg.mask_path = self._save_mask(image.id, seg.id, md["mask"])
                self._classify_segment(img, seg, md["mask"], classifier)
                self.repo.add_segment(seg)
                segments.append(seg)
        
        if progress_callback:
            progress_callback({"event": "progress", "stage": "done", "progress": 100, "message": "自動分割完成！"})
        return segments

    # ---------- 互動式：使用者點一下切一塊 ----------
    def segment_point(self, image: ImageRecord, point: tuple[int, int]) -> Segment:
        img = self._read_rgb(image.path)
        md = self.segmenter.segment_at(img, point)
        seg = Segment(image_id=image.id, owner_id=image.owner_id, bbox=tuple(md["bbox"]), area=md["area"])
        seg.mask_path = self._save_mask(image.id, seg.id, md["mask"])
        self._classify_segment(img, seg, md["mask"], self.get_classifier(image.owner_id))
        self.repo.add_segment(seg)
        return seg

    # ---------- 自然語言：用文字找物件並切出遮罩----------
    def segment_text(
        self,
        image: ImageRecord,
        prompt: str,
        progress_callback: Callable[[dict], None] | None = None,
    ) -> list[Segment]:
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
            progress_callback({
                "event": "progress",
                "stage": "detecting",
                "progress": 10,
                "message": "正在執行物件偵測中...",
            })

        img = self._read_rgb(image.path)

        # 中文或超過 3 個英文單字時，交給 Gemini 解析物件類別
        words = prompt.split()
        has_chinese = any("\u4e00" <= c <= "\u9fff" for c in prompt)
        use_gemini = has_chinese or len(words) > 3

        parsed_classes = (
            self.gemini_service.parse_prompt(prompt)
            if use_gemini
            else [prompt]
        )

        # Mock 測試模式：不載入真實模型
        if not self.config.use_real_sam:
            mock_segments: list[Segment] = []
            h, w = img.shape[:2]

            for i, cls in enumerate(parsed_classes):
                # 稍微偏移各個 mock 遮罩，避免完全重疊
                offset = i * 20
                mask = np.zeros((h, w), dtype=np.uint8)

                x1 = max(0, w // 2 - 50 + offset)
                y1 = max(0, h // 2 - 50 + offset)
                x2 = min(w, w // 2 + 50 + offset)
                y2 = min(h, h // 2 + 50 + offset)

                mask[y1:y2, x1:x2] = 1

                seg = Segment(
                    image_id=image.id,
                    owner_id=image.owner_id,
                    bbox=(x1, y1, x2 - x1, y2 - y1),
                    area=int(mask.sum()),
                )
                seg.mask_path = self._save_mask(
                    image.id,
                    seg.id,
                    mask,
                )
                seg.predicted_label = cls
                seg.probs = {cls: 1.0}
                seg.confidence = 0.88
                seg.needs_review = needs_review(
                    seg.confidence,
                    self.config.confidence_threshold,
                )

                self.repo.add_segment(seg)
                mock_segments.append(seg)

            if progress_callback:
                progress_callback({
                    "event": "progress",
                    "stage": "done",
                    "progress": 100,
                    "message": "文字分割完成！",
                })

            return mock_segments

        # 動態載入 YOLO-World
        if self.yolo_detector is None:
            model_path = str(
                self.config.base_dir
                / "models"
                / "yolov8x-worldv2.pt"
            )
            self.yolo_detector = YoloWorldDetector(model_path)

        # 先找出所有類別的 bounding boxes，方便計算整體進度
        detections = []

        for cls_name in parsed_classes:
            boxes = self.yolo_detector.predict_boxes(
                img,
                cls_name,
                device=self.segmenter.device,
            )
            detections.extend(
                (cls_name, bbox)
                for bbox in boxes
            )

        classifier = self.get_classifier(image.owner_id)
        segments: list[Segment] = []
        total_boxes = len(detections)

        # 將每個 bounding box 交給 SAM 分割
        for i, (cls_name, bbox) in enumerate(detections):
            if progress_callback:
                progress_val = 75 + int(
                    (i / max(total_boxes, 1)) * 20
                )
                progress_callback({
                    "event": "progress",
                    "stage": "segmenting",
                    "progress": progress_val,
                    "message": (
                        f"正在進行物件分割 "
                        f"({i + 1}/{total_boxes})..."
                    ),
                })

            try:
                md = self.segmenter.segment_by_box(img, bbox)

                seg = Segment(
                    image_id=image.id,
                    owner_id=image.owner_id,
                    bbox=tuple(md["bbox"]),
                    area=md["area"],
                )
                seg.mask_path = self._save_mask(
                    image.id,
                    seg.id,
                    md["mask"],
                )
                self._classify_segment(
                    img,
                    seg,
                    md["mask"],
                    classifier,
                )

                # 分類器沒有預測結果時，使用 Gemini/文字解析的類別
                if seg.predicted_label is None:
                    seg.predicted_label = cls_name

                self.repo.add_segment(seg)
                segments.append(seg)

            except Exception as exc:
                print(
                    "警告：YOLO Box 進行 SAM 分割失敗："
                    f"{exc}"
                )
                continue

        if progress_callback:
            progress_callback({
                "event": "progress",
                "stage": "done",
                "progress": 100,
                "message": "文字分割完成！",
            })

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
        seg = Segment(image_id=image.id, owner_id=image.owner_id, bbox=bbox, area=int((mask > 0).sum()))
        seg.mask_path = self._save_mask(image.id, seg.id, mask)
        self._classify_segment(img, seg, mask, self.get_classifier(image.owner_id))
        self.repo.add_segment(seg)
        return seg

    # ---------- 對單一片段做分類 + 信心判斷 ----------
    def _classify_segment(self, img: np.ndarray, seg: Segment, mask: np.ndarray, classifier: FewShotClassifier) -> None:
        feat = self.embedder.encode(img, mask)
        if classifier.ready:
            probs = classifier.predict(feat)
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
        # owner 從片段本身帶出來，不是從目前登入者──即使是 admin 幫別人標，範例還是歸屬片段真正的主人
        ex = LabelExample(label=label, feature=feat.tolist(), source_segment_id=seg.id, owner_id=seg.owner_id)
        self.repo.add_example(ex)

        # 人也順手把這片段標好
        seg.human_label = label
        seg.reviewed = True
        seg.needs_review = False
        self.repo.update_segment(seg)

        # 主動學習迴圈：回訓 + 重新預測未審片段（只影響同一個 owner）
        self.refit(seg.owner_id)
        self.reclassify_pending(seg.owner_id)
        return ex

    # ---------- 刪掉標錯的類別（連帶回訓，只影響同一個 owner）----------
    def delete_label(self, label: str, owner_id: str) -> int:
        n = self.repo.delete_label(label, owner_id)
        self.refit(owner_id)
        self.reclassify_pending(owner_id)
        return n

    # ---------- 取得（必要時建立）某使用者的分類器 ----------
    def get_classifier(self, owner_id: str) -> FewShotClassifier:
        if owner_id not in self.classifiers:
            self.refit(owner_id)
        return self.classifiers[owner_id]

    # ---------- 重建某使用者的分類器 ----------
    def refit(self, owner_id: str) -> None:
        clf = FewShotClassifier(
            kind=self.config.classifier_kind, k=self.config.knn_k, temperature=self.config.softmax_temperature
        )
        clf.fit(self.repo.list_examples(owner_id=owner_id))
        self.classifiers[owner_id] = clf

    # ---------- 回訓後重新預測該使用者尚未人工審核的片段 ----------
    def reclassify_pending(self, owner_id: str) -> None:
        classifier = self.get_classifier(owner_id)
        cache: dict[str, np.ndarray] = {}
        for seg in self.repo.list_segments(owner_id=owner_id):
            if seg.reviewed:
                continue
            if not classifier.ready:
                # 範例被刪光、分類器失效 → 清掉舊預測，退回送審（別殘留 stale label）
                seg.probs, seg.predicted_label, seg.confidence, seg.needs_review = {}, None, 0.0, True
                self.repo.update_segment(seg)
                continue
            if seg.image_id not in cache:
                cache[seg.image_id] = self._read_rgb(self.repo.get_image(seg.image_id).path)
            mask = imread(seg.mask_path, cv2.IMREAD_GRAYSCALE)
            self._classify_segment(cache[seg.image_id], seg, mask, classifier)
            self.repo.update_segment(seg)
