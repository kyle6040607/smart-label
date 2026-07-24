# ==============================================================================
# Smart Label - Google Cloud Run Optimized Dockerfile
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

# 建立執行期需要的資料與暫存目錄
RUN mkdir -p data uploads masks models

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