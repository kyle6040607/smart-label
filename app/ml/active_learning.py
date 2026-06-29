"""信心分數與送審判斷

小分類器輸出各類別機率後，用三種策略之一算「信心」，
低於門檻就標紅送人審。門檻是可調旋鈕：
  調高 → 品質穩、送審多；調低 → 更省人力。
"""
from __future__ import annotations

import numpy as np


def confidence_score(probs: dict[str, float], strategy: str = "max_prob") -> float:
    """依策略把機率分佈換算成單一信心值（越高越有把握）。

    strategy:
      - "max_prob": 最高機率值
      - "margin":   前兩名差距（差距大代表有把握）
      - "entropy":  1 - 正規化亂度（越集中越有把握）
    """
    if not probs:
        return 0.0
    p = np.array(sorted(probs.values(), reverse=True), dtype=np.float64)

    if strategy == "max_prob":
        return float(p[0])

    if strategy == "margin":
        top2 = p[1] if len(p) > 1 else 0.0
        return float(p[0] - top2)

    if strategy == "entropy":
        p = p[p > 0]
        ent = -np.sum(p * np.log(p))
        max_ent = np.log(len(p)) if len(p) > 1 else 1.0
        return float(1.0 - ent / max_ent) if max_ent > 0 else 1.0

    raise ValueError(f"未知的 confidence strategy: {strategy}")


def needs_review(confidence: float, threshold: float) -> bool:
    """信心低於門檻 → 標紅送人工審核。"""
    return confidence < threshold
