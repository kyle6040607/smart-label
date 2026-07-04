# 自建切割模型 × GCP 化：可行性評估與範圍規劃

> 目標：評估「在 SAM 這一段，改用自建/自訓的切割模型」以及「把整個專案搬上 GCP、用自己的 GCP 資源做訓練」的可行性，並盤點加進現有專案後，整個系統的範圍會擴張成什麼樣子。
>
> 對象：`smart-label`（智慧分割標記助手，SAM 驅動的人機協作影像標記工具）。

---

## 0. 一句話結論

**兩件事都可行，而且現有骨架已經替你把縫留好了。**

- 「自建切割模型對抗 SAM」——不要理解成「打敗 SAM 的全能」（那不切實際），而是「在你的**特定領域**做得比 SAM 更準、更輕、更快」。這條路完全可行，而且你的工具本身就會產出訓練資料，形成天然的**資料飛輪**。
- 「變成 app + 用 GCP 訓練」——把**服務**（Flask 網站）和**訓練**（吃 GPU 的批次工作）分成兩個 workload 分開部署，就是標準做法。現有的 `Segmenter` / `Repository` 介面讓你「換模型、換儲存後端」時上層完全不用改。

代價是：專案範圍會從「單機 Flask 骨架」擴張成「serving + 儲存 + 訓練 + MLOps + 基礎設施」五塊的小型系統。下面逐項說明。

---

## 1. 現況盤點（縫在哪裡）

| 模組 | 現況 | 抽換縫 |
|------|------|--------|
| 切割 `app/ml/sam.py` | `MockSegmenter`（OpenCV 分水嶺）跑通流程；`SamSegmenter` 只有骨架 | **`Segmenter` Protocol**：`segment()` / `segment_at()` |
| 特徵 `app/ml/embedding.py` | `MockEmbedder`（顏色直方圖）；`DinoEmbedder` 骨架 | `Embedder` Protocol：`encode()` |
| 分類 `app/ml/classifier.py` | few-shot kNN / softmax（**唯一「我們訓練」的部分**） | 已可用 |
| 儲存 `app/repository.py` | 記憶體 + `store.json`；檔案存本機磁碟 | `Repository` 類別（方法可換後端） |
| 服務 `app/__init__.py` | Flask `create_app()` 工廠，單一進入點 | 已容器化友善 |

**關鍵觀察**：要「換掉 SAM 的實作」，你只需要在 `sam.py` 新增第三個實作 `CustomSegmenter`，讓它符合同一個 `Segmenter` 介面，再用環境變數切換即可。`pipeline.py` / 各 route / 前端 **一行都不用動**。這就是這個專案設計最值錢的地方。

---

## 2. 問題一：能不能自建切割模型「對抗」SAM？

### 2.1 先校正期待

SAM 用了 SA-1B（**11 億個 mask**）訓練，追求的是「任何影像、任何物件」的通用零樣本切割。想在**通用能力**上贏它，個人/課程專案不可能複製那個資料規模——這條路直接放棄。

但你的專案根本不需要通用。它是**特定領域**的標記工具（你只標你關心的那幾類物件）。在窄領域上，一個小模型只要看過幾百張你的資料，就可以又快又準，甚至在「你的物件」上切得比 SAM 原始輸出更貼邊。**這才是「對抗」的正確定義：不是全能，而是在你的地盤上贏。**

### 2.2 五種可選路線（由淺到深）

| 方案 | 做法 | 訓練成本 | 領域內品質 | 定位 |
|------|------|----------|-----------|------|
| **A. 直接用 SAM（凍結）** | 接 `SamSegmenter` | 0 | 通用強、貼邊普通 | baseline + 自動標註機 |
| **B. 輕量 SAM 變體** | MobileSAM / FastSAM / SAM2 | 0~低 | 接近 A、快很多 | 省資源、CPU 可跑 |
| **C. 微調 SAM decoder** | 凍結 image encoder，只訓 mask decoder（或加 LoRA） | 中（單張 GPU 可） | 領域內↑ | 「改良」SAM |
| **D. 自訓專用分割模型** | YOLOv8/11-seg、Mask R-CNN、U-Net，用你的資料微調 | 中高（需 GPU + 標註） | 領域內可超越、通用差 | 真正「取代」SAM ⭐ |
| **E. 蒸餾（distillation）** | 用 SAM 自動標大量資料 → 訓一個小 student | 中 | 快、專用 | 資料飛輪的產物 |

### 2.3 推薦主線：A → D（資料飛輪）

你的工具**已經內建資料集匯出**（`app/services/exporter.py`，支援 COCO / YOLO / mask）。這讓下面這個迴圈天然成立：

```
接真 SAM 當 baseline ─► 人審少量低信心樣本 ─► 匯出 COCO/YOLO 資料集
        ▲                                              │
        │                                              ▼
  換上你的模型（越來越準）◄── GCP 上訓練 YOLOv8-seg ◄── 上傳資料集到 GCS
```

- 前期：SAM + 人工當「自動標註機」，快速攢出一份乾淨的領域資料集。
- 中期：拿這份資料在 GCP 上訓練 `CustomSegmenter`（建議 **YOLOv8/11-seg**：訓練簡單、GCP 上一張 T4 幾小時搞定、直接吐 COCO/YOLO 格式，和你的 exporter 對齊）。
- 後期：把訓練好的 checkpoint 接回 `Segmenter` 介面，量測「自訓模型 vs SAM」的 mIoU 與省下的人工工時——這正好是報告裡最有力的成果數據。

> 「D 當主線、A 當對照組」是最有題目性的敘事：你不是重造 SAM，而是證明「在特定任務上，一個小模型＋人機協作的資料飛輪，可以逼近甚至超越通用大模型」。

### 2.4 落地時新增的介面（不影響上層）

```python
# app/ml/sam.py 新增第三個實作
class CustomSegmenter:
    """載入你在 GCP 訓練出來的專用分割模型（YOLOv8-seg / Mask R-CNN）。"""
    def __init__(self, checkpoint: str, device: str = "cpu"):
        # 從 GCS 或本地載入 checkpoint
        ...
    def segment(self, image): ...        # 符合 Segmenter Protocol
    def segment_at(self, image, point): ...

# build_segmenter() 增加一個分支：USE_CUSTOM_SEG=1 → CustomSegmenter
```

---

## 3. 問題二：變成 app + 用 GCP 訓練，怎麼做？

### 3.1 最重要的觀念：服務 ≠ 訓練，要拆成兩個 workload

| Workload | 特性 | 該放哪 |
|----------|------|--------|
| **Serving（Flask 網站）** | 長時間在線、輕量、多人存取、要快回應 | **Cloud Run**（容器、自動擴縮） |
| **Training（訓練分割模型）** | 吃 GPU、批次跑幾小時、跑完就關 | **Vertex AI Custom Training Job** 或 GCE GPU VM |

**絕對不要**把訓練塞進 web request（會 timeout、擋住服務），也不要為了偶爾訓練養一台 24 小時開著的 GPU 機器（燒錢）。訓練是「按下按鈕 → 起一個 GPU job → 跑完存 checkpoint 到 GCS → 關機」。

### 3.2 GCP 元件對應表

| 需求 | GCP 服務 | 備註 |
|------|----------|------|
| 跑 Flask 網站 | **Cloud Run** | 容器化、自動擴縮；推論若要 GPU 可開 Cloud Run GPU |
| 存原圖 / 遮罩 / 資料集 / checkpoint | **Cloud Storage (GCS)** | 取代目前的本機 `data/uploads`、`data/masks` |
| 存 metadata（取代 `store.json`） | **Firestore**（簡單）或 **Cloud SQL / MySQL**（提案原訂） | `Repository` 介面已備好可換 |
| 訓練分割模型（GPU） | **Vertex AI Custom Training** | 一張 T4/L4 即可；跑完自動釋放 |
| Docker image 倉庫 | **Artifact Registry** | Cloud Run 從這裡拉 image |
| 觸發 / 排程回訓 | **Cloud Run Jobs** + **Cloud Scheduler** 或 **Vertex Pipelines** | 例如「累積 N 筆新標註就回訓」 |
| 訓練指標視覺化 | **Vertex AI TensorBoard**（或提案的 Tableau 接 export 資料） | mIoU 曲線、模型比較 |

### 3.3 現況會擋路、必須一起改的點

1. **儲存無狀態化**：Cloud Run 是無狀態、多實例、重啟即失憶。目前把圖存本機磁碟、metadata 存 `store.json`，上雲後會遺失/不同步。
   - → 影像/遮罩 IO 改走 **GCS**（`app/utils.py` 的 `imread`/`imwrite` 加雲端路徑支援）。
   - → `Repository` 換 **Firestore/Cloud SQL** 後端（介面已抽象，上層不動）。
2. **缺 Dockerfile**：要新增 `Dockerfile` + `gunicorn`（正式 WSGI server，別用 Flask 內建 dev server 上線）。
3. **SAM 推論慢**：SAM ViT-H 在 CPU 上一張圖數秒~十幾秒。選項：用 **MobileSAM/FastSAM**（CPU 可接受）、開 **Cloud Run GPU**、或把「自動分割整張」做成**非同步工作**（丟 job、前端輪詢）。
4. **公開前建議加驗證**：目前無 auth。上公網前用 **Identity-Aware Proxy** 或簡單登入保護。

### 3.4 成本感（課程/ demo 規模，量級參考）

- Cloud Run serving：demo 流量基本落在免費額度內。
- GCS：幾 GB 影像，月費個位數美金。
- Vertex 訓練：T4 GPU 約 **$0.35–0.5/hr**，訓一次 YOLOv8-seg（幾百張圖）約 1–3 小時 → **一次訓練約 1~2 美金**。
- 結論：對學生專案，**訓練花費幾乎可忽略**，主要成本是「記得把 GPU job 關掉」（用 Vertex job 就會自動關）。

---

## 4. 加進專案後，整個範圍變成什麼

現在是「單一 Flask 骨架、本機、mock 模型、JSON 存檔」。整合後會長成下面**五層**，其中兩層是全新的。

```
┌─ 4) 部署 / 基礎設施層（全新）────────────────────────────┐
│  Artifact Registry · Cloud Run · GCS buckets ·           │
│  Firestore/Cloud SQL · IAM · (可選) Cloud Build CI/CD    │
├─ 3) 訓練 / MLOps 層（全新）──────────────────────────────┤
│  匯出資料集→GCS · Vertex 訓練 job · checkpoint 版本管理 · │
│  回訓觸發(手動/排程) · mIoU 指標與「模型 vs SAM」比較     │
├─ 2) 模型層（核心新增）──────────────────────────────────┤
│  SamSegmenter(接真SAM) · DinoEmbedder(接真DINOv2) ·      │
│  CustomSegmenter(自訓模型 ⭐) · training/ 訓練腳本        │
├─ 1) 應用層（已有，小改）────────────────────────────────┤
│  Flask app + Dockerfile/gunicorn · Repository 換雲端後端 │
│  · utils IO 走 GCS                                       │
└─ 0) 前端 / API（幾乎不動）──────────────────────────────┘
   index.html · app.js · 各 route  ← 靠介面隔離，受益於不變
```

### 逐層要動/新增的檔案

| 層 | 動作 | 具體檔案 / 元件 |
|----|------|----------------|
| 應用 | 改 | `Dockerfile`（新）、`app/utils.py`（GCS IO）、`app/repository.py`（新增 Firestore/SQL 實作） |
| 應用 | 改 | `app/config.py` 加雲端設定（bucket 名、DB 連線、`USE_CUSTOM_SEG`） |
| 模型 | 補 | `app/ml/sam.py`（`SamSegmenter` 補完 + 新增 `CustomSegmenter`）、`app/ml/embedding.py`（`DinoEmbedder` 補完） |
| 模型 | 新 | `training/`（獨立於 web app）：資料集載入、YOLOv8-seg 訓練/評估腳本 |
| MLOps | 新 | 匯出→GCS 腳本、Vertex 訓練 job 設定、checkpoint registry、回訓觸發 |
| 基礎設施 | 新 | Artifact Registry、Cloud Run 服務、GCS buckets、DB、IAM 角色、（可選）`cloudbuild.yaml` |

---

## 5. 建議的分階段路線圖（疊在提案原本的 4 週上）

| 階段 | 目標 | 產出 |
|------|------|------|
| **P0 現況** | 骨架端到端可跑（已完成） | mock 全通 |
| **P1 接真模型（本機）** | `SamSegmenter` + `DinoEmbedder` 補完，本機驗證品質 | 真 SAM 切割 + DINOv2 特徵 |
| **P2 上雲 serving** | 加 Dockerfile → 部署 Cloud Run；儲存搬 GCS + Firestore/SQL | 一個公開網址、無狀態可擴縮 |
| **P3 GCP 訓練** | exporter 匯出資料集→GCS；Vertex 訓練 YOLOv8-seg | 你自己的 `CustomSegmenter` checkpoint |
| **P4 對抗實驗 + 飛輪** | 接回 `CustomSegmenter`；量測「自訓 vs SAM」mIoU / 省工時；排程回訓 | 報告的核心成果數據 |

> P1、P3 可平行：一邊本機接 SAM 攢資料，一邊搭訓練管線。

---

## 6. 風險與取捨（誠實版）

- **想贏 SAM 的通用能力 → 放棄**。守住「窄領域贏」的敘事，才守得住。
- **範圍膨脹風險**：五層全做是一個小型 MLOps 系統，時間有限就砍 P4 的「自動排程回訓」，保留「手動觸發訓練」即可交出完整故事。
- **推論延遲**：真 SAM 在 CPU 上慢，demo 前先決定走 MobileSAM/FastSAM 還是 Cloud Run GPU，別到 demo 才發現卡。
- **資料量**：自訓模型至少要幾百張標好的領域圖才有意義；P1 的「SAM+人工當自動標註機」就是為了快速湊到這個量。
- **成本失控唯一來源**：忘了關 GPU。用 Vertex Custom Job（跑完自動釋放）而非長開的 GCE VM，即可規避。

---

## 7. 下一步可以立刻做的三件事

1. **P1 起手**：把 `SamSegmenter` / `DinoEmbedder` 的 `TODO` 補完，本機用 `USE_REAL_SAM=1 USE_REAL_EMBEDDING=1` 驗證真模型品質。
2. **補 `Dockerfile` + gunicorn**：先讓專案能容器化本機跑，為 P2 上 Cloud Run 鋪路。
3. **抽 `Repository` 雲端後端介面**：先加一個 Firestore 或 GCS 版實作，驗證「換儲存後端、上層不動」這條假設。

需要的話，我可以直接動手做上面任何一項（例如先產出 `Dockerfile` 與 `CustomSegmenter` 骨架，或把 `SamSegmenter` 接上真 SAM）。
