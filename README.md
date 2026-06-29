# 智慧分割標記助手

SAM 驅動的 AI 影像標記網頁 — 人機協作。使用者標少量範例，系統自動分割其餘照片，人只審核被標紅的低信心樣本。

> 這是**基本架構（骨架）**：整條流程已能端到端跑通，重模型（SAM、DINOv2）先用可運作的 mock 實作 + 乾淨介面，之後直接抽換真模型即可，上層 API / 前端不用改。

## 核心流程（對應提案）

```
上傳照片 → SAM 切割（零樣本遮罩）→ DINOv2/CLIP 取特徵（凍結）
        → few-shot 小分類器（我們訓練）→ 算信心分數
        → 高信心自動接受 / 低信心標紅送人審 → 人修正 → 回訓（主動學習迴圈）
```

## 目錄結構

```
app/
├── __init__.py            Flask 應用工廠 create_app()
├── config.py             設定（路徑、信心門檻、模型開關）
├── api/                  REST API（Flask blueprints）
│   ├── images.py           上傳 / 瀏覽影像
│   ├── segment.py          自動分割 / 單點分割 / 取遮罩
│   ├── labels.py           標種子範例
│   ├── review.py           審核佇列 / 統計
│   └── export.py           匯出資料集（COCO / YOLO / mask）
├── core/                 AI pipeline（專案核心）
│   ├── sam.py              SAM 切割：MockSegmenter ↔ SamSegmenter
│   ├── embedding.py        特徵抽取：MockEmbedder ↔ DinoEmbedder
│   ├── classifier.py       few-shot 分類器（kNN / softmax）★ 我們訓練這個
│   ├── active_learning.py  信心分數（max_prob / margin / entropy）
│   ├── pipeline.py         串接四個模組 + 主動學習迴圈
│   └── export.py           打包成可訓練資料集（最終產出）
├── storage/repository.py  資料存取（記憶體+JSON，可抽換成 MySQL/MongoDB）
├── models/schemas.py      資料結構（ImageRecord / Segment / LabelExample）
├── templates/index.html   標記網頁
└── static/                前端 JS / CSS
main.py                   啟動進入點
tests/test_pipeline.py    端到端冒煙測試
```

## 快速開始

```bash
uv sync                 # 安裝依賴
uv run python main.py   # 啟動 → http://127.0.0.1:5000
uv run pytest           # 跑測試
```

操作：上傳照片 → 點縮圖選圖 → 「自動分割整張」或直接點物件 → 標幾個種子類別
→ 右側審核紅色低信心片段 → 看「已省下 ○○% 工時」統計即時更新。

## 抽換成真模型

骨架用環境變數切換 mock / 真實作，預設全 mock：

| 變數 | 預設 | 說明 |
|------|------|------|
| `USE_REAL_SAM` | `0` | `1` 啟用真 SAM（需 `uv add segment-anything torch torchvision` + 下載 checkpoint） |
| `USE_REAL_EMBEDDING` | `0` | `1` 啟用真 DINOv2/CLIP（需 `uv add torch transformers`） |
| `CLASSIFIER` | `knn` | few-shot 分類器：`knn` 或 `softmax` |
| `CONFIDENCE_STRATEGY` | `max_prob` | 信心策略：`max_prob` / `margin` / `entropy` |
| `CONFIDENCE_THRESHOLD` | `0.6` | 低於此值標紅送審（可調旋鈕） |

真實作的接點都已在 `core/sam.py` 與 `core/embedding.py` 留好 `TODO` 與安裝指引。

## API 一覽

| 方法 | 路徑 | 用途 |
|------|------|------|
| POST | `/api/images` | 上傳照片（多檔） |
| GET  | `/api/images` | 列出照片 |
| GET  | `/api/images/<id>/file` | 取原圖 |
| POST | `/api/images/<id>/segment` | 自動分割整張 |
| POST | `/api/images/<id>/segment_point` | 單點分割 `{x, y}` |
| GET  | `/api/segments/<id>/mask` | 取遮罩 PNG |
| POST | `/api/segments/<id>/label` | 標種子範例 `{label}` |
| POST | `/api/segments/<id>/review` | 審核修正 `{label}` |
| GET  | `/api/review/queue` | 低信心待審佇列 |
| GET  | `/api/stats` | 統計（自動接受比例 ≈ 省下工時） |
| GET  | `/api/export?format=` | 匯出資料集 zip：`coco`（預設）/ `yolo` / `mask` |

## 對應提案的開發週期

- **第 1 週**：跑通 SAM；上傳→點選→出遮罩 → 把 `SamSegmenter` 接上（目前 Mock 已通）
- **第 2 週**：MVP + 自動分類；embedding + kNN → 把 `DinoEmbedder` 接上
- **第 3 週**：主動學習迴圈；信心分流、審核介面、準確率曲線（架構已備）
- **第 4 週**：Docker 上線、跑出成果數據、做 demo

## 技術對應（提案第 9 頁）

Flask 網頁 · Scikit-Learn few-shot · OpenCV 影像處理 · MySQL/MongoDB（Repository 介面已留）· Docker+GCP 部署 · Tableau 呈現
