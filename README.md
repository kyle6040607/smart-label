# Seer

**AI 影像標記平台 — 人機協作，讓系統做初稿，人只處理關鍵判斷。**

用一句話說出你要找什麼（例如「找出戴安全帽的工人」），系統自動定位物件並產生輪廓；
只有低信心的結果會標紅送人審核。修正會回饋成訓練樣本，越標越準。
支援從網頁工作台操作，也支援直接用 LINE 傳照片。

---

## 為什麼需要它

傳統影像標記的成本不在「難」，而在「重複」：每張圖從零框選、修邊、命名，
換一個需求就得重做一輪，而且今天改過的錯，明天可能再犯一次。

Seer 把流程反過來——系統先做完大量初稿，人只確認系統不確定的部分：

```
說出目標  →  系統偵測 + 分割  →  信心分流  →  人審低信心  →  匯出資料集
                                      ↓                ↑
                              高信心自動接受      修正回饋成訓練樣本
```

## 主要功能

| 功能 | 說明 |
|---|---|
| 開放詞彙偵測 | YOLO-World 直接用文字提示找物件，不必事先訓練類別 |
| 中文提示詞理解 | Gemini 把「戴安全帽的工人」翻成模型看得懂的詞彙清單 |
| 互動式分割 | MobileSAM 單點分割、整張自動分割、多邊形手繪 |
| 少樣本分類 | DINOv2 特徵 + kNN / softmax 分類器，標幾張就能推廣 |
| 主動學習 | 三種信心策略（max_prob / margin / entropy）分流，低信心才送審 |
| 模型訓練 | 標好的資料可直接訓練 YOLOv26x-seg，支援自訂輪數與中途停止 |
| LINE 整合 | LINE Bot + LIFF 上傳照片，背景處理完推播通知並下載結果 ZIP |
| 資料集匯出 | COCO / YOLO segmentation / 語意分割 mask 三種格式 |
| 多專案隔離 | 每個使用者可建立多個專案，影像與標註互不干擾 |

## 快速開始

```bash
uv sync                        # 安裝依賴
uv run python main.py          # 啟動 → http://127.0.0.1:8080
uv run pytest                  # 跑測試
```

預設以 JSON 檔案儲存、模型走 mock 實作，不需任何外部服務即可跑通整條流程。
要啟用真模型見下方「模型設定」。

**操作流程**：註冊登入 → 上傳照片 → 輸入要找的物件（或直接點圖）→
系統自動標記 → 右側審核紅色低信心片段 → 匯出資料集或直接訓練模型。

## 設定

所有設定都走環境變數，放在專案根目錄的 `.env`（見 `.env.example`）。

### 模型設定

| 變數 | 預設 | 說明 |
|---|---|---|
| `USE_REAL_SAM` | `0` | `1` 啟用真 MobileSAM，需先下載 checkpoint（見下） |
| `USE_REAL_EMBEDDING` | `0` | `1` 啟用真 DINOv2，模型由 `torch.hub` 首次執行時自動下載 |
| `CLASSIFIER` | `knn` | few-shot 分類器：`knn` 或 `softmax` |
| `CONFIDENCE_STRATEGY` | `max_prob` | 信心策略：`max_prob` / `margin` / `entropy` |
| `CONFIDENCE_THRESHOLD` | `0.6` | 低於此值標紅送審 |
| `YOLO_WORLD_CONFIDENCE` | `0.4` | YOLO-World 偵測門檻 |
| `YOLO_IMGSZ` | `640` | 推論解析度，須為 32 倍數；越大越抓得到小物件 |
| `GEMINI_API_KEY` | 空 | 提示詞語意解析；未設定時直接把原文送給模型 |

<details>
<summary><b>下載 MobileSAM checkpoint</b></summary>

真 SAM 需要權重檔 `mobile_sam.pt`（約 40MB），已被 `.gitignore` 排除、不進 git：

```bash
curl -L -o models/mobile_sam.pt \
  https://github.com/ChaoningZhang/MobileSAM/raw/master/weights/mobile_sam.pt
```

> Windows PowerShell 用 `curl.exe`，或手動從
> <https://github.com/ChaoningZhang/MobileSAM/tree/master/weights> 下載放進 `models/`。

路徑由 `SAM_CHECKPOINT` 控制，架構由 `SAM_MODEL_TYPE` 指定（MobileSAM 為 `vit_t`）。
放好後 `USE_REAL_SAM=1 uv run python main.py` 即啟用。

</details>

### 安全設定

| 變數 | 預設 | 說明 |
|---|---|---|
| `SMART_LABEL_ENV` | `dev` | 設 `prod` 進入正式模式；此時未提供 `SECRET_KEY` 會**拒絕啟動** |
| `SECRET_KEY` | dev 固定值 | session cookie 簽章金鑰，正式部署**必須**設隨機值 |
| `DEFAULT_ADMIN_USER` / `DEFAULT_ADMIN_PASSWORD` | `sa` / `sa` | 首次啟動種入的管理者帳號，正式部署請改掉 |

> 產生金鑰：`python -c "import os; print(os.urandom(32).hex())"`。
> 同一部署的多 worker / 多容器要用**同一把** key，session 才會一致。

### 資料儲存

| 變數 | 預設 | 說明 |
|---|---|---|
| `DB_BACKEND` | `auto` | `auto`（有 MySQL 位址就用）/ `json` / `mysql` |
| `MYSQL_HOST` · `MYSQL_USER` · `MYSQL_PASSWORD` · `MYSQL_DATABASE` | 空 | MySQL 連線資訊 |
| `MYSQL_UNIX_SOCKET` | 空 | Cloud SQL 走 socket：`/cloudsql/<PROJECT>:<REGION>:<INSTANCE>` |
| `USE_GCS` | `0` | `1` 改存 Google Cloud Storage，需搭配 `GCS_PROJECT_ID`、`GCS_BUCKET` |

檔案預設存於本機 `data/uploads`、`data/masks`、`data/tasks`；
啟用 GCS 後分別對應 `images/`、`masks/`、`datasets/`。

<details>
<summary><b>既有資料搬到 GCS</b></summary>

啟用 GCS 前，必須在仍可讀取舊本機檔案的機器上搬移既有資料：

```bash
uv run python scripts/migrate_files_to_gcs.py --dry-run
uv run python scripts/migrate_files_to_gcs.py
```

腳本會上傳並改寫 `images.path`、`segments.mask_path` 及 `annotation_tasks.dataset_zip_path`，
不刪除本機來源檔案，可以安全重跑。舊路徑已移動時用 `--source-root` 指定；
資料量大時用 `--batch-size` 調整批次；正式遷移會先完成全量 preflight，
任何來源缺檔都不會開始上傳，也可用 `--preflight-only` 只做檢查。

`USE_GCS=1` 時若 GCS 未正確建立，應用程式會直接失敗，不會靜默改存本機。

</details>

### LINE 整合

| 變數 | 說明 |
|---|---|
| `LINE_CHANNEL_SECRET` · `LINE_CHANNEL_ACCESS_TOKEN` | Messaging API channel（Bot 推播） |
| `LINE_LOGIN_CHANNEL_ID` · `LINE_LOGIN_CHANNEL_SECRET` | LINE Login channel（登入，與上面是不同 channel） |
| `LIFF_ID` · `PUBLIC_BASE_URL` | LIFF 應用與對外網址 |
| `SMTP_HOST` · `SMTP_PORT` · `SMTP_USER` · `SMTP_PASSWORD` | 註冊驗證碼寄送；未設定時驗證碼直接印在 log |

## 專案結構

```
app/
├── __init__.py            Flask 應用工廠 create_app()
├── config.py              設定（路徑、門檻、模型開關、雲端服務）
├── models.py              資料結構（ImageRecord / Segment / Project / AnnotationTask）
├── repository.py          資料存取（JSON / MySQL 可抽換）
├── storage.py             檔案儲存（本機 / GCS 可抽換）
├── ml/                    模型模組
│   ├── sam.py               互動式分割：MockSegmenter ↔ SamSegmenter（MobileSAM）
│   ├── yolo_world.py        開放詞彙偵測（YOLO-World）
│   ├── yolov26x_seg.py      實例分割模型訓練（YOLOv26x-seg）
│   ├── embedding.py         特徵抽取：MockEmbedder ↔ DinoEmbedder（DINOv2）
│   ├── classifier.py        few-shot 分類器（kNN / softmax）
│   └── active_learning.py   信心分數（max_prob / margin / entropy）
├── services/              業務邏輯
│   ├── pipeline.py          串接偵測、分割、分類與主動學習迴圈
│   ├── exporter.py          匯出資料集（COCO / YOLO / mask）
│   ├── gemini.py            中文提示詞 → 模型詞彙清單
│   ├── task_runner.py       背景任務領取、lease、重試
│   ├── task_processor.py    LIFF 任務推論與 ZIP 打包
│   ├── task_notifier.py     LINE 完成／失敗通知
│   ├── cloud_run_jobs.py    觸發 Cloud Run Job
│   ├── line_login.py        LINE Login OAuth
│   └── mailer.py            SMTP 驗證碼寄送
├── routes/                REST API（Flask blueprints）
│   ├── auth.py              註冊 / 登入 / Email 驗證 / LINE Login
│   ├── projects.py          專案建立、切換、刪除
│   ├── images.py            上傳 / 瀏覽 / 刪除影像
│   ├── segment.py           文字提示、單點、多邊形分割與批次操作
│   ├── labels.py            標籤管理與種子範例
│   ├── review.py            審核佇列 / 統計
│   ├── training.py          YOLOv26x-seg 訓練啟動、狀態、停止、下載
│   ├── export.py            匯出資料集
│   ├── line_bot.py          LINE Bot webhook
│   └── liff.py              LIFF 分批上傳與任務頁
├── templates/             網頁（index / login / register / verify / liff）
└── static/                前端 JS / CSS

scripts/                 CLI 工具（背景 worker、資料遷移）
tests/                   測試（25+ 檔，含模型 benchmark）
MySQL/init.sql           資料庫 schema
docs/                    架構圖、週報、簡報素材
main.py                  啟動進入點
```

## API 一覽

所有 API 以 `/api` 為前綴，需登入 session。

**專案** `/api/projects`
| 方法 | 路徑 | 用途 |
|---|---|---|
| GET / POST | `/api/projects` | 列出 / 建立專案 |
| PATCH / DELETE | `/api/projects/<id>` | 更名 / 刪除 |
| POST | `/api/projects/<id>/select` | 切換目前專案 |

**影像** `/api/images`
| 方法 | 路徑 | 用途 |
|---|---|---|
| GET / POST | `/api/images` | 列出 / 上傳照片（多檔、支援壓縮檔） |
| GET | `/api/images/<id>/file` | 取原圖 |
| DELETE | `/api/images/<id>` · `/api/images/delete_batch` | 刪除 / 批次刪除 |

**分割與標記**
| 方法 | 路徑 | 用途 |
|---|---|---|
| POST | `/api/images/<id>/segment_text` | 文字提示偵測 + 分割 |
| POST | `/api/images/batch_segment_text` | 多張批次文字提示分割 |
| POST | `/api/images/<id>/segment` · `/segment_point` · `/segment_polygon` | 整張自動 / 單點 / 手繪分割 |
| GET | `/api/segments/tasks` · POST `/api/segments/tasks/<id>/cancel` | 背景分割任務狀態 / 取消 |
| GET | `/api/segments/<id>/mask` | 取遮罩 PNG |
| POST | `/api/segments/<id>/label` | 標種子範例 `{label}` |
| GET / POST / PATCH / DELETE | `/api/labels` · `/api/labels/rename` | 標籤管理 |

**審核與統計**
| 方法 | 路徑 | 用途 |
|---|---|---|
| GET | `/api/review/queue` | 低信心待審佇列 |
| POST | `/api/segments/<id>/review` · `/unreview` | 審核修正 / 取消審核 |
| POST | `/api/segments/batch_confirm_high_confidence` | 批次接受高信心結果 |
| POST | `/api/segments/batch_action_by_threshold` | 依門檻批次處理 |
| GET | `/api/stats` | 統計（自動接受比例 ≈ 省下工時） |
| GET / POST | `/api/parameters` | 讀取 / 調整推論參數 |

**訓練與匯出**
| 方法 | 路徑 | 用途 |
|---|---|---|
| POST | `/api/projects/<id>/train` | 啟動 YOLOv26x-seg 訓練（可指定 epochs / patience） |
| GET | `/api/projects/<id>/train/status` | 訓練進度 |
| POST | `/api/projects/<id>/train/stop` | 中途停止 |
| GET | `/api/projects/<id>/train/download` | 下載訓練好的權重 |
| GET | `/api/export?format=` | 匯出資料集 zip：`coco`（預設）/ `yolo` / `mask` |
| GET | `/api/export/preview` | 匯出前預覽內容 |

**LINE**：`/callback`（Bot webhook）、`/liff/*`（分批上傳、任務狀態、結果下載）。

## 部署

以 Docker 打包，部署於 Cloud Run + Cloud SQL + Cloud Storage：

```bash
docker compose up --build          # 本機
gunicorn "app:create_app()"        # 正式（Cloud Run 由 Dockerfile 帶起）
```

<details>
<summary><b>LIFF 背景任務 Worker</b></summary>

LINE 上傳的照片由獨立 Worker 處理，避免長時間推論佔住 web request：

```bash
python scripts/task_worker.py --mode loop    # 常駐輪詢（Cloud Run Worker Pool）
python scripts/task_worker.py --mode drain   # 清空佇列後離開（Cloud Run Job）
```

Worker 用短交易領取一個 `pending` 任務並立即提交 `processing`，
模型推論與 ZIP 打包不持有資料庫 row lock。無任務時輪詢間隔由
`TASK_WORKER_POLL_SECONDS` 控制。

正式環境每次領取都會產生 fencing token，每 60 秒更新 heartbeat、延長 15 分鐘 lease。
Worker 中斷後，下一次 drain 啟動會用 `FOR UPDATE SKIP LOCKED` 排他回收逾時任務。
模型失敗會在 1 分鐘、5 分鐘後重試，最多 3 次；只有最後一次失敗才標成 `failed`
並發送 LINE 通知。Cloud Run Job 本身應關閉 platform retry，重試狀態以 Cloud SQL 為唯一來源。
逾時 attempt 的 Segment、mask、縮圖與未發布 ZIP 會依 fencing token 清除；
GCS 清理暫時失敗時任務會保留清理 token，成功前不允許重新領取。
Web 與 Job 同時啟動時，schema migration 會先取得 MySQL advisory lock。

`loop` 模式必須使用 MySQL——JSON Repository 無法跨程序安全共用，
Worker 會拒絕以 JSON 後端啟動。Web 與 Worker 必須連到同一個 MySQL 服務。

Cloud Run Job 請使用與 Web 相同的 image、service account、Cloud SQL 與 GCS 設定，
command 設為 `python scripts/task_worker.py --mode drain`，並固定
`tasks=1`、`parallelism=1`、`max-retries=0`。建議由 Cloud Scheduler 每 5 分鐘執行一次；
空佇列時 Pipeline 採延遲初始化，不會載入 SAM / embedding 模型。

</details>

<details>
<summary><b>LIFF 推論與匯出門檻</b></summary>

LIFF 任務建立時會保存推論設定快照，不受 Web 端動態參數影響。
候選偵測預設 `0.15`，匯出條件為 `detection_confidence >= 0.5`；
低於門檻的 Segment 與 mask 會保留但不放入 ZIP，並建立最大 512×512、
JPEG quality 80 的 private 縮圖供 LIFF 任務頁查看。
全部結果都低信心時任務仍會完成，但不建立空 ZIP，並發送一次 LINE 通知
引導使用者從 Rich Menu 查看未通過縮圖。

ZIP 會直接串流寫入儲存層，不先放進記憶體。下載 GCS ZIP 時優先產生
10 分鐘有效的 signed URL，Cloud Run service account 需具備
`iam.serviceAccounts.signBlob` 權限；只缺簽署權限時會記錄 warning
並退回由應用程式分塊串流。

</details>

## 技術棧

Python 3.13 · Flask · PyTorch · Ultralytics（YOLO-World / YOLOv26x-seg）·
MobileSAM · DINOv2 · scikit-learn · OpenCV · MySQL / Cloud SQL ·
Google Cloud Storage · LINE Messaging API + LIFF · Gemini API · Docker · Cloud Run
