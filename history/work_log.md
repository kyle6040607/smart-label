# Work Log

## 2026-07-18

- **app/ml/GroundingDINO**
  - **環境變數與 C++ 編譯繞過**：由於本機 Windows CUDA Toolkit 11.8 與現有 PyTorch 13.2 存在 CUDA 版本不相容（CUDA Mismatch）衝突，且 `setup.py` 中有強制透過 subprocess 呼叫 pip 安裝 torch 的 Bug，我們將 `setup.py` 中的 `get_extensions()` 函數修改為直接 `return None`。這成功強制腳本進入純 Python 構建模式，完美避開了 MSVC 編譯與 CUDA 衝突，並在 `uv` 虛擬環境中以 `uv pip install --no-build-isolation -e .` 在 5 秒內極速安裝成功。
  - **修復底層所有 positional to() Bug**：直接修改本機克隆的 `groundingdino/util/inference.py`（第 64-65 行）、`groundingdino/util/utils.py`（第 162 行、第 595 行）、`groundingdino/util/misc.py`（第 427 行、第 431 行）、以及模型內部的 `groundingdino/models/GroundingDINO/utils.py`（第 132 行），將所有以位置參數傳遞的 `.to(device)` 全數修改為具名引數 `.to(device=device)`。這徹底解決了模型內部 Forward 傳播時拿取 Tensor 實體 `torch.device` 物件在 Python 3.13 / PyTorch bindings 下產生的 `TypeError` 重載解析錯誤。
  - **新舊 Transformers 版本相容性修復**：修改 `bertwarper.py` 的 `get_extended_attention_mask` 呼叫端。因新版 `transformers` 徹底拿掉了 `device` 參數，故採用 `try-except` 包裝：首選嘗試無 `device` 的新版 API，拋出 `TypeError` 時則安全回退至有 `device` 的舊版 API，徹底解決了跨版本引數順序不一致的崩潰。
  - **解決 C++ 算子缺失強制 GPU 呼叫 Bug**：修改 `ms_deform_attn.py` 中的 `_C` 導入防護（導入失敗時將其設為 `None`），並將 CUDA 執行條件限制在 `_C is not None` 時才執行自定義 CUDA C++ 算子。當 `_C` 缺失時，將安全降級回退到純 PyTorch 實作的 `multi_scale_deformable_attn_pytorch` 運算，其同樣支持 GPU (CUDA) 的矩陣計算。這解鎖了 Windows 無編譯環境下透過 GPU 執行 Deformable Attention 的能力。
- **app/ml/GroundingDINO.py**
  - **新建 GroundingDinoDetector 封裝類別**：比照 `YoloWorldDetector`，封裝了 Grounding DINO 影像定位邏輯。包含自動常態化影像 Tensor（ToTensor + Normalize）、處理 Prompt 尾端句點對齊、執行模型預測，並實作 `box_cxcywh_to_xyxy` 轉換，將歸一化的相對比例座標乘以實體寬高還原為與 SAM 相容的絕對 `[x1, y1, x2, y2]` 像素座標。
  - **相容性熱補丁 (Monkey Patch)**：針對新版 `transformers` 移除了 `BertModel.get_head_mask` 方法導致老舊 `groundingdino` 執行期崩潰的 Bug，於檔案最頂部動態注入 `get_head_mask` 回傳 `[None] * num_hidden_layers`，在不降級任何套件的情況下，完美解決了 AttributeError 相容性問題。
  - **防範 Python 3.13 專屬的 PyTorch `to()` Bug**：針對 Python 3.13 下 PyTorch C++ bindings 在呼叫 `tensor.to(device_object)` 時會錯誤將 `torch.device` 物件配對至 `dtype` 參數而拋出 `TypeError` 的 Bug，加固 `predict_boxes`：強制將傳入的 `device` 轉為純字串格式（`"cuda"` 或 `"cpu"`），徹底根除了此 C 語言層面的重載配對崩潰。
  - **優化 predict 調用參數**：在呼叫 `predict(...)` 時，將 `remove_combined` 顯式設為 `True`，避免一個候選框匹配到多個分拆詞彙時導致不必要的重疊框合併問題。
  - **導入 try-except 導入防護**：針對 Cloud Run 等不需安裝 AI 模型套件的輕量 Mock 測試環境，在頂部導入 `groundingdino` 模組時加上 `try...except ImportError` 保護，並在 `__init__` 加入判定。這可防止 Python 啟動時因為找不到相關 AI 套件而直接崩潰，保證系統能以純通訊模式順利執行。
- **app/templates/index.html**
  - 在自然語言分割控制區中加入 `<select id="modelEngineSelect">` 下拉選單元件，提供 `YOLO-World` 與 `Grounding DINO` 選項。
- **app/static/js/app.js**
  - 修改 `$("textSegBtn").onclick`，讀取下拉選單的值並將其作為 `engine` 參數傳入 POST `/segment_text` API 的 body JSON。
  - 於 `setSegmentationLoading` 與 `selectImage` 中將下拉選單控制鈕的啟用與禁用與其他文字分割控制鈕同步。
- **app/routes/segment.py**
  - 在 `/images/<image_id>/segment_text` 路由中接收 `engine` 參數，並轉發給 Pipeline。
- **app/services/pipeline.py**
  - 導入並初始化 `GroundingDinoDetector`。
  - 修改 `segment_text` 接收 `engine` 參數，實現當引擎為 `grounding_dino` 時動態載入 Grounding DINO 並在偵測時對其與 YOLO-World 進行分流呼叫。
