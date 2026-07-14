import numpy as np
from ultralytics import YOLOWorld

class YoloWorldDetector:
    def __init__(self, model_path: str = "models/yolov8x-worldv2.pt"):
        """初始化 YOLO-World 偵測器，載入指定的本機大模型權重"""
        print(f"--- 載入 YOLO-World 權重檔: {model_path} ---")
        self.model = YOLOWorld(model_path)

    def predict_boxes(self, image: np.ndarray, prompt: str, device: str = "cuda") -> list[list[float]]:
        """給定影像與文字，找出所有符合的 bounding boxes。

        回傳格式為 list[list[float]]，每個元素為 [x1, y1, x2, y2]
        """
        # 1. 先將模型完整轉移到 GPU
        self.model.to("cuda")

        # 2. 繞過 ultralytics 的 NMS bug (GitHub Issue #9321)
        # 傳入多於 1 個類別（如加上一個 placeholder）可以強制它走正確的多類別 GPU 推理路徑。
        # 由於模型此時已在 GPU 上，生成的文字向量也會直接儲存於 GPU。
        self.model.set_classes([prompt, "placeholder_non_exist_class"])

        # 2. 進行預測，強制指定為 GPU (cuda) 運行，並設定最低信心度為 0.6
        results = self.model.predict(image, device="cuda", verbose=False, conf=0.6)

        # 3. 僅提取第一個類別 (即 index == 0 的 prompt 目標) 的預測框
        boxes = []
        for box in results[0].boxes:
            # 確保只拿符合使用者 prompt 的框，過濾掉佔位類別
            if int(box.cls[0].item()) == 0:
                # xyxy[0] 是 [x1, y1, x2, y2] 的 Tensor
                xyxy = box.xyxy[0].cpu().numpy().tolist()
                boxes.append(xyxy)

        return boxes
