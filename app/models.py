"""資料結構定義。

用 dataclass 當作整個系統流通的資料模型，之後要落地到
MySQL / MongoDB 時，這些就是對應的資料表 / 文件結構。
"""
from __future__ import annotations

import time
import uuid
from dataclasses import dataclass, field, asdict
from typing import Any


def _new_id() -> str:
    return uuid.uuid4().hex[:12]


@dataclass
class Project:
    """使用者建立的專案/標註會話。"""

    id: str = field(default_factory=_new_id)
    owner_id: str = ""
    name: str = "未命名專案"
    mode: str = "novice"  # 'novice' 或 'engineer'
    created_at: float = field(default_factory=time.time)
    updated_at: float = field(default_factory=time.time)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass
class ImageRecord:
    """一張上傳的待標記照片。"""

    id: str = field(default_factory=_new_id)
    owner_id: str = ""          # Web 使用者 ID；舊資料 / 尚未綁定的 LIFF 圖片可暫時留空
    project_id: str = ""        # 所屬專案 ID
    filename: str = ""
    path: str = ""
    width: int = 0
    height: int = 0
    file_hash: str = ""
    created_at: float = field(default_factory=time.time)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass
class AnnotationTask:
    """由 LIFF 建立的圖片標註與訓練任務。"""

    id: str = field(default_factory=_new_id)

    # 任務輸入
    user_id: str = ""
    project_id: str = ""
    line_user_id: str = ""   # LIFF 驗證取得的 LINE 使用者 ID
    prompt: str = ""
    image_ids: list[str] = field(default_factory=list)
    # 已經完成標註的圖片，追加照片時不會重複處理
    processed_image_ids: list[str] = field(default_factory=list)

    # 最新完成的資料集版本；0 表示尚未產生 ZIP
    dataset_version: int = 0

    # 最後一次成功發送 LINE 通知的版本
    notified_dataset_version: int = 0

    # pending / processing / completed / failed
    status: str = "pending"

    # 任務完成後產生的檔案
    dataset_zip_path: str = ""
    best_model_path: str = ""

    # 下載網址使用的隨機憑證
    download_token: str = field(
        default_factory=lambda: uuid.uuid4().hex
    )

    error_message: str = ""
    created_at: float = field(default_factory=time.time)
    updated_at: float = field(default_factory=time.time)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

@dataclass
class Segment:
    """SAM 切出來的一塊遮罩 + few-shot 分類結果。

    對應提案流程：SAM 切割 → 取特徵 → 小分類器給類別與信心。
    """

    id: str = field(default_factory=_new_id)
    image_id: str = ""
    mask_path: str = ""           # 遮罩檔（PNG，0/255）
    bbox: tuple[int, int, int, int] = (0, 0, 0, 0)  # x, y, w, h
    area: int = 0

    # few-shot 分類輸出
    predicted_label: str | None = None
    probs: dict[str, float] = field(default_factory=dict)  # 各類別機率
    confidence: float = 0.0       # 依 strategy 算出的信心
    needs_review: bool = False    # 低信心 → 標紅送審（提案第 8 頁）

    # 人工審核 / 標記結果
    human_label: str | None = None
    reviewed: bool = False

    @property
    def final_label(self) -> str | None:
        """人工標的優先，其次才是模型預測。"""
        return self.human_label or self.predicted_label

    def to_dict(self) -> dict[str, Any]:
        d = asdict(self)
        d["final_label"] = self.final_label
        return d


@dataclass
class User:
    """系統使用者（登入帳號）。

    密碼永遠只存雜湊值（password_hash），不落地明文。
    之後接 MySQL / MongoDB 時，這就是對應的 users 資料表 / 集合。
    """

    id: str = field(default_factory=_new_id)
    username: str = ""
    password_hash: str = ""       # werkzeug scrypt/pbkdf2 雜湊，非明文；LINE-only 帳號留空
    email: str = ""               # 註冊信箱；LINE-only 帳號可為空
    email_verified: bool = False  # Email 是否已通過驗證碼驗證
    otp_hash: str = ""            # 驗證碼雜湊（sha256），不存明碼
    otp_expires: float = 0.0      # 驗證碼到期時間（epoch 秒）
    otp_attempts: int = 0         # 已嘗試次數，達上限須重寄
    role: str = "user"            # 預留：user / admin
    created_at: float = field(default_factory=time.time)

    # --- LINE Login 綁定（提案：LINE 登入 / 帳號綁定）---
    line_user_id: str | None = None  # LINE 的使用者唯一識別（id_token 的 sub）
    display_name: str = ""           # LINE 顯示名稱
    avatar_url: str = ""             # LINE 大頭貼

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass
class LabelExample:
    """使用者標的種子範例（few-shot 的錨點，提案第 7 頁）。

    存的是特徵向量 + 類別，分類器靠這些範例做最近鄰 / softmax。
    """

    id: str = field(default_factory=_new_id)
    owner_id: str = ""          # 與來源圖片相同的 Web 使用者 ID
    project_id: str = ""        # 所屬專案 ID
    label: str = ""
    feature: list[float] = field(default_factory=list)
    source_segment_id: str | None = None
    created_at: float = field(default_factory=time.time)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)
