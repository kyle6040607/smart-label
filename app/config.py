"""應用設定。

集中管理路徑、門檻、模型開關等可調參數。
之後要接 MySQL / MongoDB 就在這裡加連線設定。
"""
from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = BASE_DIR / "data"
UPLOAD_DIR = DATA_DIR / "uploads"
MASK_DIR = DATA_DIR / "masks"
DB_FILE = DATA_DIR / "store.json"


@dataclass
class Config:
    # --- 檔案存放 ---
    base_dir: Path = BASE_DIR
    data_dir: Path = DATA_DIR
    upload_dir: Path = UPLOAD_DIR
    mask_dir: Path = MASK_DIR
    db_file: Path = DB_FILE

    # --- 主動學習 / 信心門檻（提案第 8 頁的「可調旋鈕」）---
    # strategy: "max_prob" | "margin" | "entropy"
    confidence_strategy: str = os.getenv("CONFIDENCE_STRATEGY", "max_prob")
    confidence_threshold: float = float(os.getenv("CONFIDENCE_THRESHOLD", "0.6"))

    # --- few-shot 分類器（提案第 6、7 頁）---
    # classifier: "knn" | "softmax"
    classifier_kind: str = os.getenv("CLASSIFIER", "knn")
    knn_k: int = int(os.getenv("KNN_K", "5"))
    softmax_temperature: float = float(os.getenv("SOFTMAX_TEMPERATURE", "0.1"))

    # --- mock 分割器的可調旋鈕（換上真 SAM 後失效）---
    sam_max_masks: int = int(os.getenv("SAM_MAX_MASKS", "12"))
    sam_min_area_ratio: float = float(os.getenv("SAM_MIN_AREA_RATIO", "0.004"))
    sam_flood_tol: int = int(os.getenv("SAM_FLOOD_TOL", "12"))  # 單點分割的容差，越大圈越多
    sam_checkpoint: str = os.getenv("SAM_CHECKPOINT", "models/mobile_sam.pt")

    # --- 後端模型開關：mock 先跑通流程，之後抽換真模型 ---
    use_real_sam: bool = os.getenv("USE_REAL_SAM", "0") == "1"
    use_real_embedding: bool = os.getenv("USE_REAL_EMBEDDING", "0") == "1"

    # --- 登入 / session ---
    # 正式部署請用環境變數覆蓋，勿沿用預設值。
    secret_key: str = os.getenv("SECRET_KEY", "dev-smart-label-change-me")
    default_admin_user: str = os.getenv("DEFAULT_ADMIN_USER", "sa")
    default_admin_password: str = os.getenv("DEFAULT_ADMIN_PASSWORD", "sa")

    # --- 上傳限制 ---
    max_content_length: int = 32 * 1024 * 1024  # 32 MB
    allowed_ext: tuple[str, ...] = field(
        default_factory=lambda: ("png", "jpg", "jpeg", "bmp", "webp")
    )

    def ensure_dirs(self) -> None:
        for d in (self.data_dir, self.upload_dir, self.mask_dir):
            d.mkdir(parents=True, exist_ok=True)


config = Config()
