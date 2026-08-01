"""守住 Pipeline 對 YoloWorldDetector 的呼叫契約。

pipeline.segment_text() 以關鍵字傳入 device / conf 呼叫 predict_boxes()，
簽名少任何一個都只會在真實模式（USE_REAL_SAM=1）下才炸 —— mock 模式在
碰到 YOLO 之前就 return 了，既有測試抓不到。

這裡只讀簽名、不建立實例，因此不需要 GPU 也不需要權重檔。
"""
from __future__ import annotations

import inspect

from app.ml.yolo_world import YoloWorldDetector
from app.services.pipeline import Pipeline


def test_predict_boxes_accepts_pipeline_kwargs():
    """predict_boxes 必須接受 pipeline 實際傳入的關鍵字。"""
    params = inspect.signature(YoloWorldDetector.predict_boxes).parameters

    for name in ("image", "prompt", "device", "conf", "imgsz"):
        assert name in params, f"predict_boxes 缺少 {name} 參數"


def test_pipeline_still_passes_tunables_to_detector():
    """pipeline.segment_text 必須把兩個可調參數都帶給偵測器，面板才有作用。"""
    source = inspect.getsource(Pipeline._segment_text_locked)

    assert "predict_boxes(" in source
    assert "conf=self.config.yolo_world_confidence" in source
    assert "imgsz=self.config.yolo_imgsz" in source
