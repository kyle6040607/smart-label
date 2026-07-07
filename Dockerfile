FROM python:3.13-slim

# Python 設定
ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1

# 安裝 uv
RUN pip install --no-cache-dir uv

# 工作目錄
WORKDIR /app

# 先複製相依設定
COPY pyproject.toml ./
COPY uv.lock ./

# 安裝套件
RUN uv sync --frozen

# 複製專案
COPY . .

# 對外開放 Flask Port
EXPOSE 5000

# 啟動 Flask
CMD ["uv", "run", "python", "main.py"]