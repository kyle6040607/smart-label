# 📝 專案工作與變更紀錄 (Work Log)

---

## 📅 [2026-07-31] 匯出器 Docstring 與文件註解 YOLOv26x-seg 全面更名對齊

### 📄 文件與模組註解對齊 (Documentation & Alignment)
- **更名全專案 Exporter 模組與文件中的 YOLOv8-seg 為 YOLOv26x-seg**：
  - 檔案：`app/services/exporter.py`
  - 內容：將 `exporter.py` 模組頂層 docstring、註解及規格說明中的 `YOLOv8-seg` 全數更新對齊為 **`YOLOv26x-seg`**。說明 Ultralytics 下游分割格式（`images/` + 歸一化多邊形 `labels/*.txt` + `data.yaml`）完全相容最新旗艦款 `yolo26x-seg.pt` 之訓練。

---

## 📅 [2026-07-31] Gemini LLM 重複呼叫效能極致優化 (One-time Gemini Prompt Parsing)

### 🚀 後端效能與 API 配額最佳化 (Performance Optimization)
- **多圖批次標註重構 Gemini 提前解析機制**：
  - 檔案：`app/routes/segment.py`, `app/services/pipeline.py`
