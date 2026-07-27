import cv2
import numpy as np
from ultralytics import YOLOWorld

class YoloWorldDetector:
    def __init__(self, model_path: str = "models/yolov8x-worldv2.pt"):
        """初始化 YOLO-World 偵測器，載入指定的本機大模型權重"""
        print(f"--- 載入 YOLO-World 權重檔: {model_path} ---")
        self.model = YOLOWorld(model_path)

    def predict_boxes(self, image: np.ndarray, prompt: str, device: str = "cuda", imgsz: int = 640) -> list[list[float]]:
        """給定影像與文字，找出所有符合的 bounding boxes。

        回傳格式為 list[list[float]]，每個元素為 [x1, y1, x2, y2]
        """
        # 1. 先將模型轉移到指定裝置
        self.model.to(device)

        # 2. 設定預測目標類別（僅使用使用者的 prompt，不添加其他佔位類別）
        self.model.set_classes([prompt])

        # 3. 進行預測
        # 💡 重要：Ultralytics YOLO.predict 傳入 numpy array 時預期格式為 BGR。
        # 由於 Pipeline 中使用的是 RGB 矩陣，在此我們必須將其轉回 BGR，
        # 這能 100% 根除 YOLO-World 內部發生的「紅藍通道對調」隱藏 Bug，讓顏色偵測變得極其精確！
        bgr_image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
        results = self.model.predict(
            bgr_image,
            device=device,
            verbose=True,
            conf=0.4,
            imgsz=imgsz
        )
        # 💡 印出模型實際抓到的所有框數以及名稱
        detected_names = [prompt] * len(results[0].boxes)
        print(f"📊 [YOLO-World] 原始偵測總框數: {len(detected_names)}, 偵測到的物件名稱: {detected_names}")

        # 4. 提取所有預測框並轉為 Python list
        boxes = []
        for box in results[0].boxes:
            # xyxy[0] 是 [x1, y1, x2, y2] 的 Tensor
            xyxy = box.xyxy[0].cpu().numpy().tolist()
            boxes.append(xyxy)

        print(f"🎯 [YOLO-World] 符合 Prompt '{prompt}' 的篩選後框數: {len(boxes)}")
        return boxes
