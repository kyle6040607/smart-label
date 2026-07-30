import cv2
import numpy as np
from ultralytics import YOLOWorld


def letterbox(
    img: np.ndarray,
    new_shape: int | tuple[int, int] = 640,
    color: tuple[int, int, int] = (114, 114, 114),
    scaleup: bool = False,
) -> tuple[np.ndarray, float, tuple[float, float]]:
    """Letterbox 影像預處理：進行等比例縮放與 Padding 補邊。

    參數:
        img: 輸入影像矩陣 (BGR 或 RGB)。
        new_shape: 目標大小 (如 640 或 (640, 640))。
        color: Padding 的填充顏色 (預設為灰色 114)。
        scaleup: 若為 False，當影像尺寸小於設定值時不放大，僅進行補邊。

    回傳:
        (letterboxed_image, scale_ratio, (pad_w, pad_h))
    """
    shape = img.shape[:2]  # 目前影像尺寸 [h, w]
    if isinstance(new_shape, int):
        new_shape = (new_shape, new_shape)

    # 計算縮放比例 r
    r = min(new_shape[0] / shape[0], new_shape[1] / shape[1])
    if not scaleup:  # 如果小於設定值就不放大
        r = min(r, 1.0)

    # 計算未補邊 (Unpadded) 時的縮放後尺寸
    new_unpad = (int(round(shape[1] * r)), int(round(shape[0] * r)))
    dw = new_shape[1] - new_unpad[0]  # 寬度 padding 總和
    dh = new_shape[0] - new_unpad[1]  # 高度 padding 總和

    dw /= 2.0  # 兩邊平分 padding
    dh /= 2.0

    if shape[::-1] != new_unpad:
        img = cv2.resize(img, new_unpad, interpolation=cv2.INTER_LINEAR)

    top, bottom = int(round(dh - 0.1)), int(round(dh + 0.1))
    left, right = int(round(dw - 0.1)), int(round(dw + 0.1))
    img = cv2.copyMakeBorder(
        img, top, bottom, left, right, cv2.BORDER_CONSTANT, value=color
    )

    return img, r, (dw, dh)


class YoloWorldDetector:
    def __init__(self, model_path: str = "models/yolov8x-worldv2.pt"):
        """初始化 YOLO-World 偵測器，載入指定的本機大模型權重"""
        print(f"--- 載入 YOLO-World 權重檔: {model_path} ---")
        self.model = YOLOWorld(model_path)

    def predict_boxes(
        self,
        image: np.ndarray,
        prompt: str,
        device: str = "cuda",
        conf: float | None = None,
        imgsz: int = 640,
        scaleup: bool = False,
    ) -> list[list[float]]:
        """給定影像與文字，找出所有符合的 bounding boxes。

        參數:
            image: 輸入影像 (RGB)
            prompt: 類別標籤字串
            device: 執行設備 ("cuda" 或 "cpu")
            conf: 信心度門檻 (未指定預設 0.4)
            imgsz: Letterbox 目標影像尺寸 (可調參數，預設 640)
            scaleup: 是否放大圖片 (預設 False，若小於 imgsz 則不放大，僅補邊)

        回傳:
            格式為 list[list[float]]，每個元素為原圖空間座標 [x1, y1, x2, y2]
        """
        orig_h, orig_w = image.shape[:2]

        # 1. 先將模型轉移到指定裝置
        self.model.to(device)

        # 2. 設定預測目標類別（僅使用使用者的 prompt，不添加其他佔位類別）
        self.model.set_classes([prompt])

        # 🔍 [Cloud Run / CLIP 驗證] 印出並確認文本特徵向量 txt_feats 讀取狀態
        try:
            txt_feats = getattr(self.model.model, "txt_feats", None)
            if txt_feats is None:
                # 相容處理：部分 Ultralytics 架構下 txt_feats 位於內部底層網路
                inner_model = getattr(self.model.model, "model", None)
                if inner_model is not None:
                    txt_feats = getattr(inner_model, "txt_feats", None)

            if txt_feats is not None:
                print(f"✅ [YOLO-World CLIP 驗證] 成功讀取 txt_feats! 矩陣形狀: {txt_feats.shape}, 設備: {txt_feats.device}")
            else:
                print("⚠️ [YOLO-World CLIP 驗證] 未直接找到 self.model.model.txt_feats 屬性。")
        except Exception as e:
            print(f"❌ [YOLO-World CLIP 驗證] 讀取 txt_feats 異常: {e}")

        # 3. 進行圖像色彩空間轉換與 Letterbox 前處理
        # 💡 注意：Pipeline 中傳入的是 RGB 矩陣，轉換為 BGR
        bgr_image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)

        # 套用 Letterbox 預處理 (scaleup=False: 小於 imgsz 不放大只補邊)
        letterbox_img, ratio, (pad_w, pad_h) = letterbox(
            bgr_image, new_shape=imgsz, scaleup=scaleup
        )

        # 4. 進行模型推論
        results = self.model.predict(
            letterbox_img,
            device=device,
            verbose=False,
            conf=conf if conf is not None else 0.4,
            imgsz=imgsz,
        )

        # 印出模型實際抓到的所有框數以及名稱
        detected_names = [prompt] * len(results[0].boxes)
        print(
            f"📊 [YOLO-World] 原始偵測總框數: {len(detected_names)}, 偵測到的物件名稱: {detected_names}"
        )

        # 5. 提取所有預測框，並將 Letterbox 空間座標精確還原回原圖空間座標
        boxes = []
        for box in results[0].boxes:
            # xyxy 是在 Letterbox 影像上的 [x1, y1, x2, y2]
            lbox = box.xyxy[0].cpu().numpy().tolist()

            # 減去 Padding 並除以縮放比例 ratio 還原回原圖
            x1 = (lbox[0] - pad_w) / ratio
            y1 = (lbox[1] - pad_h) / ratio
            x2 = (lbox[2] - pad_w) / ratio
            y2 = (lbox[3] - pad_h) / ratio

            # 防呆邊界裁切，防止座標溢出原圖範圍
            x1 = max(0.0, min(float(orig_w), float(x1)))
            y1 = max(0.0, min(float(orig_h), float(y1)))
            x2 = max(0.0, min(float(orig_w), float(x2)))
            y2 = max(0.0, min(float(orig_h), float(y2)))

            boxes.append([x1, y1, x2, y2])

        print(
            f"🎯 [YOLO-World] (Letterbox imgsz={imgsz}, scaleup={scaleup}) 符合 Prompt '{prompt}' 的原圖座標框數: {len(boxes)}"
        )
        return boxes
