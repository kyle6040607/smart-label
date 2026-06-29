"""Unicode 安全的影像讀寫。

cv2.imread / imwrite 在 Windows 不支援非 ASCII 路徑（例如裝在「桌面」底下），
會直接讀失敗回 None。改走 np.fromfile / tofile + imdecode / imencode 繞過——
這兩個吃 bytes，不碰作業系統的路徑編碼。
"""
from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np


def imread(path: str, flags: int = cv2.IMREAD_COLOR) -> np.ndarray | None:
    data = np.fromfile(path, dtype=np.uint8)  # 吃 unicode 路徑
    if data.size == 0:
        return None
    return cv2.imdecode(data, flags)


def imwrite(path: str, img: np.ndarray) -> bool:
    ext = Path(path).suffix or ".png"
    ok, buf = cv2.imencode(ext, img)
    if ok:
        buf.tofile(path)
    return bool(ok)
