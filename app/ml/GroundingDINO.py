import os
import cv2
import torch
import numpy as np
import torchvision.transforms as T

# 💡 針對新版 transformers 移除 get_head_mask 導致 Grounding DINO 崩潰的相容性熱補丁 (Monkey Patch)
try:
    from transformers.models.bert.modeling_bert import BertModel
    if not hasattr(BertModel, "get_head_mask"):
        def get_head_mask(self, head_mask, num_hidden_layers, is_attention_chunked=False):
            return [None] * num_hidden_layers
        BertModel.get_head_mask = get_head_mask
        print("💡 [Grounding DINO] 成功套用 BertModel.get_head_mask 相容性熱補丁！")
except Exception as e:
    print(f"⚠️ [Grounding DINO] 套用 BertModel 熱補丁失敗: {e}")

try:
    from groundingdino.util.inference import load_model, predict
    from groundingdino.util import box_ops
    HAS_GROUNDINGDINO = True
except ImportError as e:
    HAS_GROUNDINGDINO = False
    print(f"⚠️ [Grounding DINO] 載入底層套件失敗（這在 Mock 測試模式下是正常的）: {e}")

class GroundingDinoDetector:
    def __init__(
        self, 
        config_path: str = "app/ml/GroundingDINO/groundingdino/config/GroundingDINO_SwinT_OGC.py",
        checkpoint_path: str = "models/groundingdino_swint_ogc.pth"
    ):
        """初始化 Grounding DINO 偵測器"""
        if not HAS_GROUNDINGDINO:
            raise ImportError(
                "偵測到未安裝 Grounding DINO 套件！"
                "請確保在非 Mock 模式下執行前，已在專案根目錄下使用 `uv pip install --no-build-isolation -e app/ml/GroundingDINO` 安裝套件。"
            )
        print(f"--- 載入 Grounding DINO 權重檔: {checkpoint_path} ---")
        if not os.path.exists(config_path):
            raise FileNotFoundError(f"找不到 Grounding DINO 配置文件: {config_path}")
        if not os.path.exists(checkpoint_path):
            raise FileNotFoundError(f"找不到 Grounding DINO 權重文件: {checkpoint_path}")
            
        # 暫時先加載到 cpu，隨後推理時會依據傳入的 device 進行 .to(device=device) 切換
        self.model = load_model(config_path, checkpoint_path, device="cpu")
        self.model.eval()

    def predict_boxes(
        self, 
        image: np.ndarray, 
        prompt: str, 
        device: str = "cuda", 
        conf: float = 0.25
    ) -> list[list[float]]:
        """給定 RGB 影像與文字，找出所有符合 Graves 的 bounding boxes。

        回傳格式為 list[list[float]]，每個元素為 [x1, y1, x2, y2]
        """
        # 💡 針對 Python 3.13 下 PyTorch to(device_object) 重載解析 Bug 的防禦性保護：
        # 將 device 強制轉為純字串格式（如 "cuda" 或 "cpu"），避免 PyTorch C++ bindings 將 torch.device 物件錯誤配對到 dtype 參數而崩潰。
        device_str = "cuda" if "cuda" in str(device) else "cpu"

        # 1. 確保模型在指定設備上運行
        self.model = self.model.to(device=device_str)

        # 2. 影像預處理：採用 Grounding DINO 官方標準 transforms (包含 RandomResize 800px)
        # 這能保證輸入模型後產生的特徵圖大小遠大於 900 queries，完全消除 selected index k out of range 問題
        from PIL import Image
        import groundingdino.datasets.transforms as GD_T

        image_pil = Image.fromarray(image)
        transform = GD_T.Compose([
            GD_T.RandomResize([800], max_size=1333),
            GD_T.ToTensor(),
            GD_T.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ])
        image_tensor, _ = transform(image_pil, None)

        # 3. 處理文字提示：Grounding DINO 要求文字提示必須以句號結尾才能有最佳語意對齊效果
        clean_prompt = prompt.strip()
        if not clean_prompt.endswith("."):
            clean_prompt += "."

        # 4. 進行推理預測
        # box_threshold 決定目標檢測的靈敏度，text_threshold 決定文字與影像特徵匹配的靈敏度
        try:
            with torch.no_grad():
                boxes, logits, phrases = predict(
                    model=self.model,
                    image=image_tensor,
                    caption=clean_prompt,
                    box_threshold=0.3,
                    text_threshold=0.25,
                    device=device_str,
                    remove_combined=True
                )
        except IndexError as e:
            # 防禦邊界條件：當該張圖片沒有足夠特徵時，PyTorch topk 可能拋出 selected index k out of range
            print(f"🎯 [Grounding DINO] 該張圖片未偵測到物體 (IndexError: {e})")
            return []
        except Exception as e:
            print(f"⚠️ [Grounding DINO] 預測異常: {e}")
            return []

        # 5. 後處理：將歸一化的 [cx, cy, w, h] 轉換為絕對的 [x1, y1, x2, y2] 像素座標
        h, w = image.shape[:2]
        if boxes.shape[0] == 0:
            print(f"🎯 [Grounding DINO] 符合 Prompt '{prompt}' 的篩選後框數: 0")
            return []

        boxes_xyxy = box_ops.box_cxcywh_to_xyxy(boxes)
        scale_factor = torch.tensor([w, h, w, h], device=boxes_xyxy.device)
        boxes_abs = boxes_xyxy * scale_factor
        boxes_list = boxes_abs.cpu().numpy().tolist()

        print(f"📊 [Grounding DINO] 偵測到物件: {phrases}, 信心值為: {logits.cpu().numpy().tolist()}")
        print(f"🎯 [Grounding DINO] 符合 Prompt '{prompt}' 的篩選後框數: {len(boxes_list)}")
        return boxes_list
