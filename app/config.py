"""應用設定。

集中管理路徑、門檻、模型開關等可調參數。
之後要接 MySQL / MongoDB 就在這裡加連線設定。
"""
from __future__ import annotations

import os
from dotenv import load_dotenv
from dataclasses import dataclass, field
from pathlib import Path



BASE_DIR = Path(__file__).resolve().parent.parent
load_dotenv(dotenv_path=BASE_DIR / ".env")
DATA_DIR = BASE_DIR / "data"
UPLOAD_DIR = DATA_DIR / "uploads"
MASK_DIR = DATA_DIR / "masks"
DB_FILE = DATA_DIR / "store.json"


def _env_bool(name: str, default: str = "0") -> bool:
    return os.getenv(name, default).strip().lower() in {
        "1",
        "true",
        "yes",
        "on",
    }



def _resolve_secret_key() -> str:
    """決定 session 簽章用的 SECRET_KEY。

    正式環境（SMART_LABEL_ENV=prod）沒給環境變數就直接拒絕啟動，
    避免沿用可預測的 dev 預設值而讓攻擊者偽造 session cookie。
    開發環境才 fallback 到固定 dev key（本機方便、多 worker 也一致）。
    """
    key = os.getenv("SECRET_KEY")
    if key:
        return key
    if os.getenv("SMART_LABEL_ENV", "dev").lower() in {"prod", "production"}:
        raise RuntimeError(
            "SECRET_KEY 未設定：正式環境必須以環境變數提供 SECRET_KEY"
        )
    return "dev-smart-label-change-me"



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
    yolo_world_confidence: float = float(os.getenv("YOLO_WORLD_CONFIDENCE", "0.4"))
    # YOLO-World 推論解析度：越大越抓得到小物件，但速度與記憶體成本上升。
    # Ultralytics 要求為 32 的倍數，可用範圍見 routes/segment.py 的驗證。
    yolo_imgsz: int = int(os.getenv("YOLO_IMGSZ", "640"))

    # --- LIFF 固定推論/匯出設定（不讀取 Web 動態 parameters）---
    # 先以較寬鬆門檻取得候選，再以 detection confidence 決定是否匯出。
    liff_yolo_world_confidence: float = float(
        os.getenv("LIFF_YOLO_WORLD_CONFIDENCE", "0.15")
    )
    liff_export_confidence_threshold: float = float(
        os.getenv("LIFF_EXPORT_CONFIDENCE_THRESHOLD", "0.5")
    )
    liff_yolo_imgsz: int = int(os.getenv("LIFF_YOLO_IMGSZ", "640"))

    # --- few-shot 分類器（提案第 6、7 頁）---
    # classifier: "knn" | "softmax"
    classifier_kind: str = os.getenv("CLASSIFIER", "knn")
    knn_k: int = int(os.getenv("KNN_K", "5"))
    softmax_temperature: float = float(os.getenv("SOFTMAX_TEMPERATURE", "0.1"))

    # --- 真正的 SAM 自動分割可調旋鈕 ---
    sam_points_per_side: int = int(os.getenv("SAM_POINTS_PER_SIDE", "32"))
    sam_min_mask_region_area: int = int(os.getenv("SAM_MIN_MASK_REGION_AREA", "100"))

    # --- mock 分割器的可調旋鈕（換上真 SAM 後失效）---
    sam_max_masks: int = int(os.getenv("SAM_MAX_MASKS", "12"))
    sam_min_area_ratio: float = float(os.getenv("SAM_MIN_AREA_RATIO", "0.004"))
    sam_flood_tol: int = int(os.getenv("SAM_FLOOD_TOL", "12"))  # 單點分割的容差，越大圈越多
    sam_checkpoint: str = os.getenv("SAM_CHECKPOINT", "models/mobile_sam.pt")
    # 權重架構必須明確指定，不從 checkpoint 檔名推測。
    sam_model_type: str = os.getenv("SAM_MODEL_TYPE", "vit_t")

    # --- 後端模型開關：mock 先跑通流程，之後抽換真模型 ---
    use_real_sam: bool = os.getenv("USE_REAL_SAM", "0") == "1"
    use_real_embedding: bool = os.getenv("USE_REAL_EMBEDDING", "0") == "1"

    # --- 檔案儲存（本機 / Google Cloud Storage）---
    use_gcs: bool = _env_bool("USE_GCS")
    gcs_project_id: str = (
        os.getenv("GCS_PROJECT_ID", "")
        or os.getenv("GCP_PROJECT_ID", "")
        or os.getenv("GCP_PROJECT", "")
        or os.getenv("GOOGLE_CLOUD_PROJECT", "")
        or os.getenv("GCLOUD_PROJECT", "")
    ).strip()
    gcs_bucket_name: str = (
        os.getenv("GCS_BUCKET_NAME", "")
        or os.getenv("GCS_BUCKET", "")
    ).strip()

    # --- LINE Bot（Messaging API channel）---
    line_channel_secret: str = os.getenv("LINE_CHANNEL_SECRET", "")
    line_channel_access_token: str = os.getenv("LINE_CHANNEL_ACCESS_TOKEN", "")

    # --- LIFF ---
    liff_id: str = os.getenv("LIFF_ID", "").strip()
    public_base_url: str = os.getenv("PUBLIC_BASE_URL", "http://127.0.0.1:8080",).strip().rstrip("/")

    # --- LINE Login（登入用，是另一個獨立的 LINE Login channel）---
    line_login_channel_id: str = os.getenv("LINE_LOGIN_CHANNEL_ID", "")
    line_login_channel_secret: str = os.getenv("LINE_LOGIN_CHANNEL_SECRET", "")
    # 留空則自動用 url_for 依當前請求組出 callback；反向代理 / https 不一致時再用環境變數覆蓋
    line_login_redirect_uri: str = os.getenv("LINE_LOGIN_REDIRECT_URI", "")
    # --- 登入 / session ---
    # 正式部署請用環境變數覆蓋，勿沿用預設值（見 _resolve_secret_key）。
    secret_key: str = field(default_factory=_resolve_secret_key)
    default_admin_user: str = os.getenv("DEFAULT_ADMIN_USER", "sa")
    default_admin_password: str = os.getenv("DEFAULT_ADMIN_PASSWORD", "sa")

    # --- Gemini API Key ---
    gemini_api_key: str = os.getenv("GEMINI_API_KEY", "")

    # --- Email 驗證碼（SMTP）---
    # 未設定 SMTP_HOST 時走開發模式：驗證碼直接印在伺服器 log，不寄信。
    # Gmail 範例：SMTP_HOST=smtp.gmail.com、SMTP_PORT=587、
    #            SMTP_USER=你的 Gmail、SMTP_PASSWORD=應用程式密碼（16 碼）
    smtp_host: str = os.getenv("SMTP_HOST", "")
    smtp_port: int = int(os.getenv("SMTP_PORT", "587"))
    smtp_user: str = os.getenv("SMTP_USER", "")
    smtp_password: str = os.getenv("SMTP_PASSWORD", "")
    mail_from: str = os.getenv("MAIL_FROM", "") or os.getenv("SMTP_USER", "")
    otp_ttl_seconds: int = int(os.getenv("OTP_TTL_SECONDS", "600"))     # 驗證碼有效 10 分鐘
    otp_max_attempts: int = int(os.getenv("OTP_MAX_ATTEMPTS", "5"))    # 最多嘗試 5 次

    # --- 資料庫後端（MySQL / Cloud SQL）---
    # DB_BACKEND: "auto"（預設，有給 MySQL 位址就用 MySQL，否則 JSON）| "json" | "mysql"
    db_backend: str = os.getenv("DB_BACKEND", "auto")
    mysql_host: str = os.getenv("MYSQL_HOST", "")
    mysql_port: int = int(os.getenv("MYSQL_PORT", "3306"))
    mysql_user: str = os.getenv("MYSQL_USER", "")
    mysql_password: str = os.getenv("MYSQL_PASSWORD", "")
    mysql_database: str = os.getenv("MYSQL_DATABASE", "")
    # Cloud Run / App Engine 掛 Cloud SQL 時走 unix socket：
    # /cloudsql/<PROJECT>:<REGION>:<INSTANCE>；設了這個就不用 mysql_host
    mysql_unix_socket: str = os.getenv("MYSQL_UNIX_SOCKET", "")

    # --- LIFF 背景任務 lease / retry ---
    task_heartbeat_seconds: float = float(
        os.getenv("TASK_HEARTBEAT_SECONDS", "60")
    )
    task_lease_seconds: float = float(
        os.getenv("TASK_LEASE_SECONDS", "900")
    )
    task_recovery_scan_seconds: float = float(
        os.getenv("TASK_RECOVERY_SCAN_SECONDS", "60")
    )
    task_recovery_batch_size: int = int(
        os.getenv("TASK_RECOVERY_BATCH_SIZE", "10")
    )
    task_max_attempts: int = int(os.getenv("TASK_MAX_ATTEMPTS", "3"))
    task_retry_base_seconds: float = float(
        os.getenv("TASK_RETRY_BASE_SECONDS", "60")
    )

    # --- LIFF 建立任務後觸發 Cloud Run Job ---
    # Job 名稱留空時停用，確保地端與測試環境不會誤觸正式 Job。
    cloud_run_task_job_name: str = os.getenv(
        "CLOUD_RUN_TASK_JOB_NAME", ""
    ).strip()
    cloud_run_task_job_region: str = os.getenv(
        "CLOUD_RUN_TASK_JOB_REGION", "asia-east1"
    ).strip()
    cloud_run_task_job_project_id: str = (
        os.getenv("CLOUD_RUN_TASK_JOB_PROJECT_ID", "")
        or os.getenv("GCS_PROJECT_ID", "")
        or os.getenv("GCP_PROJECT_ID", "")
        or os.getenv("GCP_PROJECT", "")
        or os.getenv("GOOGLE_CLOUD_PROJECT", "")
    ).strip()
    cloud_run_task_job_trigger_timeout_seconds: float = float(
        os.getenv("CLOUD_RUN_TASK_JOB_TRIGGER_TIMEOUT_SECONDS", "5")
    )

    @property
    def use_mysql(self) -> bool:
        if self.db_backend == "mysql":
            return True
        if self.db_backend == "json":
            return False
        return bool(self.mysql_host or self.mysql_unix_socket)

    # --- 上傳限制 ---
    max_content_length: int = int(os.getenv("MAX_CONTENT_LENGTH", str(8 * 1024 * 1024 * 1024)))
    max_image_size: int = int(os.getenv("MAX_IMAGE_SIZE", str(25 * 1024 * 1024)))
    allowed_ext: tuple[str, ...] = field(
        default_factory=lambda: ("png", "jpg", "jpeg", "bmp", "webp", "tiff", "tif", "avif", "heic")
    )
    allowed_archive_ext: tuple[str, ...] = field(
        default_factory=lambda: ("zip", "7z", "tar", "gz", "tgz", "bz2", "tbz2", "xz", "txz")
    )

    def ensure_dirs(self) -> None:
        for d in (self.data_dir, self.upload_dir, self.mask_dir):
            d.mkdir(parents=True, exist_ok=True)


config = Config()
