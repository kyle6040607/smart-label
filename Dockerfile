# ==============================================================================
# Seer - Google Cloud Run Optimized Dockerfile
# ==============================================================================
FROM python:3.13-slim

# 防止 Python 寫入 .pyc 檔案以及將輸出直接發送至 Cloud Logging
ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PORT=8080 \
    VIRTUAL_ENV=/app/.venv \
    PATH="/app/.venv/bin:$PATH"

WORKDIR /app

# 安裝系統相依套件：
# - git: mobile-sam 依賴 Git 來源
# - libgl1, libglib2.0-0: OpenCV (cv2) 執行期必要 C 函式庫
# - curl, ca-certificates: 網路連線與憑證驗證
RUN apt-get update && apt-get install -y --no-install-recommends \
        git \
        libgl1 \
        libglib2.0-0 \
        curl \
        ca-certificates \
    && rm -rf /var/lib/apt/lists/*

# 從官方鏡像複製最新版 uv 套件管理器
COPY --from=ghcr.io/astral-sh/uv:latest /uv /uvx /bin/

# 複製專案相依性定義
COPY pyproject.toml ./

# 1. 先獨立安裝 CPU 版的 PyTorch / torchvision（僅 ~180MB，完全避免下載 2.5GB CUDA 套件）
# 2. 再安裝專案其他相依套件（自動重用已安裝的 CPU 版 PyTorch）
RUN uv venv $VIRTUAL_ENV && \
    uv pip install torch torchvision --index-url https://download.pytorch.org/whl/cpu && \
    uv pip install . --find-links https://download.pytorch.org/whl/cpu

# YOLO-World 的 set_classes() 需要 CLIP，但 pyproject 沒宣告，Ultralytics 會在
# 執行期臨時 pip install 並警告「要重啟才生效」。在這裡先裝好，避免執行期安裝。
# 不寫進 pyproject 是為了不動 uv.lock，避免影響本機開發環境。
RUN uv pip install "clip @ git+https://github.com/ultralytics/CLIP.git"

# 建立執行期需要的資料與暫存目錄
RUN mkdir -p data uploads masks models

# 權重都在建置階段取得，不依賴 Git：yolov8x 根本沒被追蹤，mobile_sam 雖然
# 有追蹤但走 Git LFS，Cloud Build 的 checkout 只會拿到指標檔而非真檔案。
# 執行期下載則會佔用 instance memory 並拖慢每次冷啟動。
# 放在 COPY . . 之前，改程式碼時這層可重用。
RUN curl -fsSL -o models/mobile_sam.pt \
      https://raw.githubusercontent.com/ChaoningZhang/MobileSAM/master/weights/mobile_sam.pt && \
    echo "6dbb90523a35330fedd7f1d3dfc66f995213d81b29a5ca8108dbcdd4e37d6c2f  models/mobile_sam.pt" \
      | sha256sum -c -

RUN curl -fsSL -o models/yolov8x-worldv2.pt \
      https://github.com/ultralytics/assets/releases/download/v8.4.0/yolov8x-worldv2.pt && \
    echo "41e771bfbbb8894dd857f3fef7cac3b3578dffd49fd3547101efa6a606a02a0e  models/yolov8x-worldv2.pt" \
      | sha256sum -c -

RUN curl -fsSL -o models/yolo26x-seg.pt \
      https://github.com/ultralytics/assets/releases/download/v8.4.0/yolo26x-seg.pt && \
    echo "92b3de0065766a17180d6219858717dc9d03cdce8a3ca9576c97fd75aabb64f3  models/yolo26x-seg.pt" \
      | sha256sum -c -

RUN curl -fsSL -o models/yolo26n.pt \
      https://github.com/ultralytics/assets/releases/download/v8.4.0/yolo26n.pt && \
    echo "9b09cc8bf347f0fc8a5f7657480587f25db09b34bf33b0652110fb03a8ad4fef  models/yolo26n.pt" \
      | sha256sum -c -


# CLIP 權重（ViT-B/32，約 338MB）預先抓進 image 的快取目錄，否則第一次
# set_classes() 會在執行期下載。build 與執行期同為 root，快取路徑一致。
RUN python -c "import clip; clip.load('ViT-B/32', device='cpu')"

# 複製應用程式程式碼
COPY . .

# 暴露預設連接埠
EXPOSE 8080

# 使用 Gunicorn 啟動（符合 Cloud Run 最佳實踐：1 worker + 8 threads，併發效能最佳）
CMD exec gunicorn \
    --bind 0.0.0.0:${PORT:-8080} \
    --workers 1 \
    --threads 8 \
    --timeout 0 \
    main:app