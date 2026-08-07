"""人機協作標記 pipeline。

把四個模組串起來：
  SAM 切割 → DINOv2 取特徵 → few-shot 分類 → 算信心、決定是否送審。

也負責主動學習迴圈：人標了新範例 → 重訓分類器 → 重新預測未審片段。
這層是 API 與各 AI 模組之間的唯一橋樑，方便日後抽換實作。
"""
from __future__ import annotations

from contextlib import contextmanager
import logging
import threading
import time

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
from app.services.gemini import GeminiService
from app.storage import (StorageService, require_configured_storage,)


logger = logging.getLogger(__name__)


class InferenceBusyError(RuntimeError):
    """同一個 instance 的推論槽等待逾時。"""


ProgressCallback = Callable[[dict], None]


class Pipeline:
    def __init__(
        self,
        config: Config,
        repo: Repository,
        storage: StorageService | None = None,
    ):
        self.config = config
        self.repo = repo
        self.storage = require_configured_storage(config, storage,)
        self.yolo_detector = None
        self.gemini_service = GeminiService(config.gemini_api_key)
        self._segmenter = None
        self._embedder = None
        self._model_init_lock = threading.Lock()
        self._inference_lock = threading.Lock()
        # 每位使用者各自擁有分類器，避免範例、類別與預測互相污染。
        # classifier 保留為 owner_id="" 的相容別名，供舊資料與既有測試使用。
        self.classifiers: dict[str, FewShotClassifier] = {}
        self.classifier = self._new_classifier()
        self.classifiers[""] = self.classifier
        self.refit()

    @staticmethod
    def _emit_progress(
        callback: ProgressCallback | None,
        *,
        stage: str,
        message: str,
        progress: int,
    ) -> None:
        if callback is not None:
            callback({
                "event": "progress",
                "stage": stage,
                "message": message,
                "progress": progress,
            })

    @contextmanager
    def _inference_slot(self, callback: ProgressCallback | None = None):
        if self._inference_lock.locked():
            self._emit_progress(
                callback,
                stage="waiting_for_inference",
                message="前方有其他推論，正在等待…",
                progress=1,
            )

        acquired = self._inference_lock.acquire(
            timeout=self.config.inference_lock_timeout_seconds
        )
        if not acquired:
            raise InferenceBusyError(
                "model_busy：模型目前忙碌，請稍後再試"
            )
        try:
            yield
        finally:
            self._inference_lock.release()

    def _get_or_load_segmenter(
        self,
        callback: ProgressCallback | None = None,
    ):
        if self._segmenter is not None:
            return self._segmenter

        with self._model_init_lock:
            if self._segmenter is not None:
                return self._segmenter

            self._emit_progress(
                callback,
                stage="loading_model",
                message="模型正在喚醒並載入 MobileSAM 權重…",
                progress=2,
            )
            started = time.perf_counter()
            logger.info("model_load_started model=mobile_sam")
            try:
                segmenter = build_segmenter(
                    self.config.use_real_sam,
                    max_masks=self.config.sam_max_masks,
                    min_area_ratio=self.config.sam_min_area_ratio,
                    flood_tol=self.config.sam_flood_tol,
                    checkpoint=self.config.sam_checkpoint,
                    model_type=self.config.sam_model_type,
                    points_per_side=self.config.sam_points_per_side,
                    min_mask_region_area=self.config.sam_min_mask_region_area,
                )
            except Exception:
                logger.exception("model_load_failed model=mobile_sam")
                raise

            self._segmenter = segmenter
            duration_ms = round((time.perf_counter() - started) * 1000)
            logger.info(
                "model_load_completed model=mobile_sam duration_ms=%s",
                duration_ms,
            )
            self._emit_progress(
                callback,
                stage="model_ready",
                message="MobileSAM 載入完成，開始分割…",
                progress=5,
            )
            return self._segmenter

    def _get_or_load_embedder(self):
        if self._embedder is not None:
            return self._embedder

        with self._model_init_lock:
            if self._embedder is None:
                started = time.perf_counter()
                logger.info("model_load_started model=embedder")
                try:
                    embedder = build_embedder(
                        self.config.use_real_embedding
                    )
                except Exception:
                    logger.exception("model_load_failed model=embedder")
                    raise
                self._embedder = embedder
                logger.info(
                    "model_load_completed model=embedder duration_ms=%s",
                    round((time.perf_counter() - started) * 1000),
                )
        return self._embedder

    def _get_or_load_yolo(self, callback: ProgressCallback | None = None):
        if self.yolo_detector is not None:
            return self.yolo_detector

        with self._model_init_lock:
            if self.yolo_detector is not None:
                return self.yolo_detector

            self._emit_progress(
                callback,
                stage="loading_yolo",
                message="正在載入 YOLO-World 模型…",
                progress=12,
            )
            started = time.perf_counter()
            logger.info("model_load_started model=yolo_world")
            try:
                from app.ml.yolo_world import YoloWorldDetector

                model_path = str(
                    self.config.base_dir
                    / "models"
                    / "yolov8x-worldv2.pt"
                )
                detector = YoloWorldDetector(model_path)
            except Exception:
                logger.exception("model_load_failed model=yolo_world")
                raise
            self.yolo_detector = detector
            logger.info(
                "model_load_completed model=yolo_world duration_ms=%s",
                round((time.perf_counter() - started) * 1000),
            )
            return self.yolo_detector

    @property
    def segmenter(self):
        """相容既有呼叫；真正模型在第一次使用時才建立。"""
        return self._get_or_load_segmenter()

    @property
    def embedder(self):
        """相容既有測試與呼叫；DINO/Mock embedder 延後建立。"""
        return self._get_or_load_embedder()

    def _new_classifier(self) -> FewShotClassifier:
        return FewShotClassifier(
            kind=self.config.classifier_kind,
            k=self.config.knn_k,
            temperature=self.config.softmax_temperature,
        )

    def _classifier_key(self, owner_id: str, project_id: str = "") -> str:
        if project_id:
            return f"{owner_id}:{project_id}"
        return owner_id

    def _classifier_for(self, owner_id: str, project_id: str = "") -> FewShotClassifier:
        key = self._classifier_key(owner_id, project_id)
        classifier = self.classifiers.get(key)
        if classifier is None:
            classifier = self._new_classifier()
            classifier.fit(self.repo.list_examples(owner_id=owner_id, project_id=project_id if project_id else None))
            self.classifiers[key] = classifier
        return classifier

    def _owner_and_project_for_segment(self, segment: Segment) -> tuple[str, str]:
        image = self.repo.get_image(segment.image_id)
        if image is not None:
            return image.owner_id, image.project_id
        return "", ""

    # ---------- 影像 IO ----------
    def _read_rgb(self, reference: str) -> np.ndarray:
        encoded = np.frombuffer(
            self.storage.read_bytes(reference),
            dtype=np.uint8,
        )
        bgr = cv2.imdecode(encoded, cv2.IMREAD_COLOR)
        if bgr is None:
            raise ValueError(f"無法解碼圖片：{reference}")
        return cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)

    def _save_mask(self, image_id: str, seg_id: str, mask: np.ndarray) -> str:
        ok, encoded = cv2.imencode(
            ".png",
            (mask > 0).astype(np.uint8) * 255,
        )
        if not ok:
            raise ValueError("遮罩 PNG 編碼失敗")
        return self.storage.save_bytes(
            f"masks/{image_id}_{seg_id}.png",
            encoded.tobytes(),
            "image/png",
        )

    def _read_mask(self, reference: str) -> np.ndarray:
        encoded = np.frombuffer(
            self.storage.read_bytes(reference),
            dtype=np.uint8,
        )
        mask = cv2.imdecode(encoded, cv2.IMREAD_GRAYSCALE)
        if mask is None:
            raise ValueError(f"無法解碼遮罩：{reference}")
        return mask

    # ---------- 自動分割整張圖（提案 demo 第 1 步：自動分割）----------
    def segment_image(self, image: ImageRecord, progress_callback: Callable[[dict], None] | None = None) -> list[Segment]:
        with self._inference_slot(progress_callback):
            segmenter = self._get_or_load_segmenter(progress_callback)
            return self._segment_image_locked(
                image,
                segmenter,
                progress_callback,
            )

    def _segment_image_locked(
        self,
        image: ImageRecord,
        segmenter,
        progress_callback: ProgressCallback | None = None,
    ) -> list[Segment]:
        if progress_callback:
            progress_callback({"event": "progress", "stage": "segmenting", "progress": 10, "message": "正在執行影像自動切割偵測..."})

        img = self._read_rgb(image.path)
        masks = segmenter.segment(img)

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
        with self._inference_slot():
            segmenter = self._get_or_load_segmenter()
            img = self._read_rgb(image.path)
            md = segmenter.segment_at(img, point)
            seg = Segment(
                image_id=image.id,
                bbox=tuple(md["bbox"]),
                area=md["area"],
            )
            seg.mask_path = self._save_mask(image.id, seg.id, md["mask"])
            self._classify_segment(img, seg, md["mask"])
            self.repo.add_segment(seg)
            return seg

    # ---------- 自然語言：用文字找物件並切出遮罩----------
    def segment_text(
        self,
        image: ImageRecord,
        prompt: str,
        parsed_classes: list[str] | None = None,
        progress_callback: Callable[[dict], None] | None = None,
        *,
        annotation_task_id: str = "",
        task_attempt_token: str = "",
    ) -> list[Segment]:
        """依文字提示分割圖片中的物件。"""
        prompt = prompt.strip()
        if not prompt:
            raise ValueError("prompt 不可為空")
        if len(prompt) > 200:
            raise ValueError("prompt 不可超過 200 個字元")

        with self._inference_slot(progress_callback):
            segmenter = self._get_or_load_segmenter(progress_callback)
            return self._segment_text_locked(
                image,
                prompt,
                segmenter,
                parsed_classes,
                progress_callback,
                annotation_task_id=annotation_task_id,
                task_attempt_token=task_attempt_token,
            )

    def _segment_text_locked(
        self,
        image: ImageRecord,
        prompt: str,
        segmenter,
        parsed_classes: list[str] | None = None,
        progress_callback: ProgressCallback | None = None,
        *,
        annotation_task_id: str = "",
        task_attempt_token: str = "",
    ) -> list[Segment]:

        if progress_callback:
            progress_callback({
                "event": "progress",
                "stage": "detecting",
                "progress": 10,
                "message": "正在執行物件偵測中...",
            })

        img = self._read_rgb(image.path)

        if parsed_classes is None:
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
            print(f"⚠️ [Pipeline Status] 目前為 Mock 模擬模式 (use_real_sam=False)，跳過 YOLO-World 載入。若要接通真實 AI 模型，請確認 .env 中設為 USE_REAL_SAM=1。")
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
                    bbox=(x1, y1, x2 - x1, y2 - y1),
                    area=int(mask.sum()),
                    annotation_task_id=annotation_task_id,
                    task_attempt_token=task_attempt_token,
                )
                seg.mask_path = self._save_mask(
                    image.id,
                    seg.id,
                    mask,
                )
                seg.predicted_label = cls
                seg.probs = {cls: 1.0}
                seg.confidence = 0.88
                seg.detection_confidence = 0.88
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

        print(f"🚀 [Real AI Engine] 正在對圖片使用真實 YOLO-World + MobileSAM 進行 Prompt '{parsed_classes}' 標註分析...")
        # 動態載入 YOLO-World
        yolo_detector = self._get_or_load_yolo(progress_callback)

        # 先找出所有類別的 bounding boxes，方便計算整體進度
        detections = []

        for cls_name in parsed_classes:
            boxes = yolo_detector.predict_boxes(
                img,
                cls_name,
                device=segmenter.device,
                conf=self.config.yolo_world_confidence,
                imgsz=self.config.yolo_imgsz,
                include_confidence=True,
            )
            detections.extend(
                (cls_name, bbox, detection_confidence)
                for bbox, detection_confidence in boxes
            )

        segments: list[Segment] = []
        total_boxes = len(detections)

        # 將每個 bounding box 交給 SAM 分割
        for i, (cls_name, bbox, detection_confidence) in enumerate(detections):
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
                md = segmenter.segment_by_box(img, bbox)

                seg = Segment(
                    image_id=image.id,
                    bbox=tuple(md["bbox"]),
                    area=md["area"],
                    detection_confidence=detection_confidence,
                    annotation_task_id=annotation_task_id,
                    task_attempt_token=task_attempt_token,
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
        with self._inference_slot():
            return self._segment_polygon_locked(image, points)

    def _segment_polygon_locked(
        self,
        image: ImageRecord,
        points: list[tuple[int, int]],
    ) -> Segment:
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
        feat = self._get_or_load_embedder().encode(img, mask)
        owner_id, project_id = self._owner_and_project_for_segment(seg)
        classifier = self._classifier_for(owner_id, project_id)
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
        with self._inference_slot():
            return self._add_example_from_segment_locked(seg, label)

    def _add_example_from_segment_locked(
        self,
        seg: Segment,
        label: str,
    ) -> LabelExample:
        image = self.repo.get_image(seg.image_id)
        if image is None:
            raise ValueError("找不到片段所屬圖片")
        owner_id = image.owner_id
        project_id = image.project_id
        img = self._read_rgb(image.path)
        mask = self._read_mask(seg.mask_path)
        feat = self._get_or_load_embedder().encode(img, mask)

        # 避免重複範例：刪除來自同一個 seg.id 的歷史範例
        existing = [
            ex for ex in self.repo.list_examples(owner_id, project_id)
            if getattr(ex, "source_segment_id", None) == seg.id
        ]
        for ex in existing:
            self.repo.delete_example(ex.id)

        ex = LabelExample(
            owner_id=owner_id,
            project_id=project_id,
            label=label,
            feature=feat.tolist(),
            source_segment_id=seg.id,
        )
        self.repo.add_example(ex)

        # 人也順手把這片段標好
        seg.human_label = label
        seg.reviewed = True
        seg.needs_review = False
        self.repo.update_segment(seg)

        # 主動學習迴圈：回訓 + 重新預測未審片段
        self.refit(owner_id, project_id)
        self._reclassify_pending_locked(owner_id, project_id)
        return ex

    def unreview_segment(self, seg: Segment) -> None:
        """撤銷片段的審核狀態，刪除對應種子範例並重新訓練模型"""
        image = self.repo.get_image(seg.image_id)
        if image is None:
            return
        owner_id = image.owner_id
        project_id = image.project_id

        # 刪除對應的種子範例
        examples = self.repo.list_examples(owner_id, project_id)
        for ex in examples:
            if getattr(ex, "source_segment_id", None) == seg.id:
                self.repo.delete_example(ex.id)

        seg.human_label = None
        seg.final_label = None
        seg.reviewed = False
        seg.needs_review = True
        self.repo.update_segment(seg)

        self.refit(owner_id, project_id)
        self.reclassify_pending(owner_id, project_id)

    # ---------- 刪掉標錯的類別（連帶回訓）----------
    def delete_label(self, label: str, owner_id: str = "", project_id: str = "") -> int:
        with self._inference_slot():
            n, mask_paths = self.repo.delete_label(label, owner_id, project_id)
            if hasattr(self, "storage") and self.storage:
                for p in mask_paths:
                    try:
                        self.storage.delete(p)
                    except Exception:
                        pass
            self.refit(owner_id, project_id)
            self._reclassify_pending_locked(owner_id, project_id)
            return n

    # ---------- 重命名 / 合併類別（連帶回訓）----------
    def rename_label(self, old_label: str, new_label: str, owner_id: str = "", project_id: str = "") -> int:
        n = self.repo.rename_label(old_label, new_label, owner_id, project_id)
        self.refit(owner_id, project_id)
        self.reclassify_pending(owner_id, project_id)
        return n


    # ---------- 重建分類器 ----------
    def refit(self, owner_id: str | None = None, project_id: str | None = None) -> None:
        if owner_id is None:
            all_examples = self.repo.list_examples()
            keys = {self._classifier_key(e.owner_id, e.project_id) for e in all_examples}
            keys.update(self.classifiers.keys())
            keys.add("")
        else:
            keys = {self._classifier_key(owner_id, project_id or "")}

        for key in keys:
            classifier = self.classifiers.get(key)
            if classifier is None:
                classifier = self._new_classifier()
                self.classifiers[key] = classifier

            if ":" in key:
                o_id, p_id = key.split(":", 1)
                examples = self.repo.list_examples(owner_id=o_id, project_id=p_id)
            else:
                examples = self.repo.list_examples(owner_id=key if key else None)
            classifier.fit(examples)

        self.classifier = self.classifiers.get("", self._new_classifier())

    # ---------- 回訓後重新預測尚未人工審核的片段 ----------
    def reclassify_pending(self, owner_id: str | None = None, project_id: str | None = None) -> None:
        with self._inference_slot():
            self._reclassify_pending_locked(owner_id, project_id)

    def _reclassify_pending_locked(
        self,
        owner_id: str | None = None,
        project_id: str | None = None,
    ) -> None:
        cache: dict[str, np.ndarray] = {}
        for seg in self.repo.list_segments():
            if seg.reviewed:
                continue
            image = self.repo.get_image(seg.image_id)
            if image is None:
                continue
            if owner_id and image.owner_id and image.owner_id != owner_id:
                continue
            if project_id and image.project_id and image.project_id != project_id:
                continue
            classifier = self._classifier_for(image.owner_id, image.project_id)
            if not classifier.ready:
                # 範例被刪光、分類器失效 → 清掉舊預測，依當前門檻判斷送審
                seg.probs, seg.predicted_label, seg.confidence = {}, None, 0.0
                conf = getattr(seg, "detection_confidence", 0.0) or 0.0
                if self.config.confidence_threshold <= 0.0:
                    seg.needs_review = False
                else:
                    seg.needs_review = bool(conf < self.config.confidence_threshold)
                self.repo.update_segment(seg)
                continue
            if seg.image_id not in cache:
                cache[seg.image_id] = self._read_rgb(image.path)
            mask = self._read_mask(seg.mask_path)
            self._classify_segment(cache[seg.image_id], seg, mask)
            self.repo.update_segment(seg)

    def update_pending_review_status(
        self,
        owner_id: str | None = None,
        project_id: str | None = None,
    ) -> None:
        """根據目前的 confidence_threshold，僅快速更新當前專案未審核片段之 needs_review 送審狀態。"""
        thresh = self.config.confidence_threshold

        # 若指定了 project_id，先精準過濾屬於該專案的圖片
        if project_id:
            target_images = self.repo.list_images(project_id=project_id)
        else:
            target_images = self.repo.list_images()

        if owner_id:
            target_images = [img for img in target_images if img.owner_id == owner_id or not img.owner_id]

        image_map = {img.id: img for img in target_images}
        if not image_map:
            return

        for img_id in image_map:
            for seg in self.repo.list_segments(img_id):
                if seg.reviewed:
                    continue

                conf = getattr(seg, "confidence", 0.0) or 0.0
                if conf == 0.0 and getattr(seg, "detection_confidence", 0.0):
                    conf = float(seg.detection_confidence)

                if thresh <= 0.0:
                    seg.needs_review = False
                else:
                    seg.needs_review = bool(conf < thresh)

                self.repo.update_segment(seg)

