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
