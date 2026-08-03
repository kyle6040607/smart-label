# Seer

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
├── config.py              設定（路徑、信心門檻、模型開關）
├── models.py              資料結構（ImageRecord / Segment / LabelExample）
├── repository.py          資料存取（記憶體+JSON，可抽換成 MySQL/MongoDB）
├── utils.py               共用工具
├── ml/                    AI 模型模組
│   ├── sam.py               SAM 切割：MockSegmenter ↔ SamSegmenter
│   ├── embedding.py         特徵抽取：MockEmbedder ↔ DinoEmbedder
│   ├── classifier.py        few-shot 分類器（kNN / softmax）★ 我們訓練這個
│   └── active_learning.py   信心分數（max_prob / margin / entropy）
├── services/              業務邏輯
│   ├── pipeline.py          串接四個模組 + 主動學習迴圈
│   ├── exporter.py          打包成可訓練資料集（COCO / YOLO / mask）
│   └── line_login.py        LINE 登入
├── routes/                REST API（Flask blueprints）
│   ├── images.py            上傳 / 瀏覽影像
│   ├── segment.py           自動分割 / 單點分割 / 取遮罩
│   ├── labels.py            標種子範例
│   ├── review.py            審核佇列 / 統計
│   ├── export.py            匯出資料集（COCO / YOLO / mask）
│   ├── auth.py              帳號登入 / session
│   └── line_bot.py          LINE Bot
├── templates/             標記網頁（index.html / login.html）
└── static/                前端 JS / CSS
main.py                   啟動進入點
tests/                    測試（test_pipeline / test_export / test_dinov2）
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
| `USE_REAL_SAM` | `0` | `1` 啟用真 MobileSAM（相依已列在 `pyproject.toml`，`uv sync` 即安裝；另需下載 checkpoint，見下方「下載 MobileSAM checkpoint」） |
| `USE_REAL_EMBEDDING` | `0` | `1` 啟用真 DINOv2（相依已列在 `pyproject.toml`，`uv sync` 即安裝；模型由 `torch.hub` 首次執行時自動下載） |
| `CLASSIFIER` | `knn` | few-shot 分類器：`knn` 或 `softmax` |
| `CONFIDENCE_STRATEGY` | `max_prob` | 信心策略：`max_prob` / `margin` / `entropy` |
| `CONFIDENCE_THRESHOLD` | `0.6` | 低於此值標紅送審（可調旋鈕） |

真實作已完成，程式碼在 `app/ml/sam.py`（`SamSegmenter`，真 MobileSAM）與 `app/ml/embedding.py`（`DinoEmbedder`，真 DINOv2）。

### 下載 MobileSAM checkpoint

真 SAM 需要權重檔 `mobile_sam.pt`（約 40MB）。此檔已被 `.gitignore` 排除、**不進 git**，需自行下載放到 `models/`：

```bash
# 從 MobileSAM 官方 repo 下載權重到 models/
curl -L -o models/mobile_sam.pt \
  https://github.com/ChaoningZhang/MobileSAM/raw/master/weights/mobile_sam.pt
```

> Windows PowerShell：`curl.exe -L -o models\mobile_sam.pt https://github.com/ChaoningZhang/MobileSAM/raw/master/weights/mobile_sam.pt`
> 或手動到 <https://github.com/ChaoningZhang/MobileSAM/tree/master/weights> 下載 `mobile_sam.pt` 放進 `models/`。

預設路徑為 `models/mobile_sam.pt`（由環境變數 `SAM_CHECKPOINT` 控制，放別處就覆蓋它），權重架構由 `SAM_MODEL_TYPE` 明確指定，MobileSAM 應設為 `vit_t`。放好後啟用真 SAM：

```bash
USE_REAL_SAM=1 uv run python main.py
```

## 部署 / 安全設定

登入 session 與帳號相關的環境變數，正式部署務必覆蓋預設值：

| 變數 | 預設 | 說明 |
|------|------|------|
| `SMART_LABEL_ENV` | `dev` | 設 `prod` 進入正式模式；此時若未提供 `SECRET_KEY` 會**拒絕啟動** |
| `SECRET_KEY` | dev 固定值 | session cookie 簽章金鑰。開發用固定 dev key；正式**必須**設不可預測的隨機值，否則有人可偽造 cookie 冒充登入 |
| `DEFAULT_ADMIN_USER` | `sa` | 首次啟動種入的管理者帳號 |
| `DEFAULT_ADMIN_PASSWORD` | `sa` | 預設管理者密碼（雜湊後儲存）；正式部署請改掉 |

> 正式環境產生金鑰範例：`python -c "import os; print(os.urandom(32).hex())"`，把結果設成 `SECRET_KEY`。同一部署（多 worker / 多容器）要用**同一把** key，session 才會一致。

## 圖片與遮罩儲存

預設存於本機 `data/uploads`、`data/masks`、`data/tasks`。若要改存
Google Cloud Storage，在 `.env` 設定：

```dotenv
USE_GCS=1
GCS_PROJECT_ID=smart-label-501610
GCS_BUCKET=smart_label_bucket
```

GCS 物件分別存於 `images/`、`masks/`、`datasets/`。`USE_GCS=1`
時若 GCS storage 未正確建立，應用程式會直接失敗，不會靜默改存本機。

啟用 GCS 前，必須在仍可讀取舊本機檔案的機器上搬移既有資料：

```bash
uv run python scripts/migrate_files_to_gcs.py --dry-run
uv run python scripts/migrate_files_to_gcs.py
```

腳本會上傳並改寫 `images.path`、`segments.mask_path` 及
`annotation_tasks.dataset_zip_path`。它不會刪除本機來源檔案，且可以安全重跑。
若 DB 記錄的舊路徑已搬到其他位置，可用 `--source-root` 指定舊 `data` 目錄。
MySQL 資料量較大時可用 `--batch-size` 調整每批讀取筆數；正式遷移會先完成
全量 preflight，任何來源缺檔時都不會開始上傳。也可使用
`--preflight-only` 只做檢查。

LIFF 任務 ZIP 會直接串流寫入目前 storage，不會先把整包 ZIP 放進記憶體。
下載 GCS ZIP 時會優先產生 10 分鐘有效的 signed URL；Cloud Run service
account 需具備 `iam.serviceAccounts.signBlob` 權限。若只有簽署權限缺失，
系統會記錄 warning 並退回由應用程式分塊串流。

## LIFF 背景任務 Worker

獨立 Worker 使用短交易領取一個 `pending` 任務並立即提交 `processing`
狀態，模型推論與 ZIP 打包不會持有資料庫 row lock。無任務時的輪詢間隔由
`TASK_WORKER_POLL_SECONDS` 控制。

正式環境的每次領取都會產生 fencing token，並每 60 秒更新 heartbeat、延長
15 分鐘 lease。Worker 中斷後，下一次 drain 啟動會用
`FOR UPDATE SKIP LOCKED` 排他回收逾時任務。模型失敗會在 1 分鐘、5 分鐘後
重試，最多 3 次；只有最後一次失敗才標成 `failed` 並發送 LINE 通知。
Cloud Run Job 本身應關閉 platform retry，重試狀態以 Cloud SQL 為唯一來源。
逾時 attempt 的 Segment、mask、縮圖與未發布 ZIP 會依 fencing token 清除；
若 GCS 清理暫時失敗，任務會保留清理 token，下一次 scanner 會繼續清理，
成功前不允許重新領取。Web 與 Job 同時啟動時，schema migration 會先取得
MySQL advisory lock，避免兩個執行個體同時執行相同的 `ALTER TABLE`。

LIFF 任務建立時會保存推論設定快照。候選偵測預設使用 0.15，匯出條件為
`detection_confidence >= 0.5`；低於門檻的 Segment 與 mask 會保留，但不放入
ZIP，並建立最大 512×512、JPEG quality 80 的 private 縮圖供 LIFF 任務頁查看。
全部結果都低信心時任務仍會完成，但不建立空 ZIP，並發送一次 LINE 通知
引導使用者從 Rich Menu 查看未通過縮圖。

```bash
# 常駐輪詢（適合 Cloud Run Worker Pool）
python scripts/task_worker.py --mode loop

# 清空目前佇列後離開（適合 Cloud Run Job）
python scripts/task_worker.py --mode drain
```

`loop` 模式必須使用 MySQL。JSON Repository 無法由 Web 與獨立 Worker
跨程序安全共用，因此 Worker 會拒絕以 JSON 後端啟動。Web 與 Worker 必須
連到同一個 MySQL 服務。

Cloud Run Job 請使用與 Web 相同的 image、service account、Cloud SQL 與 GCS
設定，command 設為 `python scripts/task_worker.py --mode drain`，並固定
`tasks=1`、`parallelism=1`、`max-retries=0`。建議由 Cloud Scheduler 每 5 分鐘
執行一次；空佇列時 Pipeline 採延遲初始化，不會載入 SAM/embedding 模型。

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
