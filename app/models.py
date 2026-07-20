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
class ImageRecord:
    """一張上傳的待標記照片。"""

    id: str = field(default_factory=_new_id)
    filename: str = ""
    path: str = ""
    width: int = 0
    height: int = 0
    file_hash: str = ""
    created_at: float = field(default_factory=time.time)

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
    label: str = ""
    feature: list[float] = field(default_factory=list)
    source_segment_id: str | None = None
    created_at: float = field(default_factory=time.time)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass
class SegmentJob:
    """批量分割工作：一次送多張圖，背景逐張處理，前端輪詢進度。

    status: queued（排隊中）→ running（處理中）→ done（結束）。
    伺服器重啟時未完成的 job 會被標成 interrupted（見 Repository._load）。
    failed 記錄單張失敗（不中斷整批），結束後前端可拿去重試。
    """

    id: str = field(default_factory=_new_id)
    image_ids: list[str] = field(default_factory=list)
    prompt: str | None = None     # None = 自動分割整張；有值 = 逐張文字分割
    status: str = "queued"        # queued / running / done / interrupted
    done: int = 0                 # 已處理張數（含失敗的）
    failed: list[dict] = field(default_factory=list)  # [{"image_id":..., "error":...}]
    created_at: float = field(default_factory=time.time)

    def to_dict(self) -> dict[str, Any]:
        d = asdict(self)
        d["total"] = len(self.image_ids)
        return d


@dataclass
class LineSession:
    """LINE 使用者目前這一輪的圖片 + 提示詞暫存。

    使用者可傳多張圖片累加進 image_ids，輸入「傳完了」後 images_done
    設為 True，才能接受 prompt；圖文都到齊才觸發 pipeline 處理。
    圖片本身走既有的 ImageRecord 流程存檔，這裡只存 id（外鍵）。
    """

    line_user_id: str = ""
    image_ids: list[str] = field(default_factory=list)
    images_done: bool = False     # 使用者輸入「傳完了」
    confirmed: bool = False       # 使用者輸入「確認」
    prompt: str | None = None
    updated_at: float = field(default_factory=time.time)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)