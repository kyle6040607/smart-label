"""few-shot 小分類器（提案第 6、7 頁）。

吃「特徵向量 → 類別」，只用使用者標的少量種子範例（錨點）學習。
這是整個專案唯一「我們訓練」的部分，其餘（SAM、DINOv2）都是現成凍結的。

提供兩種：
  - knn：最近 K 個鄰居投票（提案第 7 頁），免訓練、加範例即時生效。
  - softmax：以類別原型做 softmax 機率。
兩者都吐出「各類別機率」，交給 active_learning 算信心。
"""
from __future__ import annotations

import numpy as np

from app.models import LabelExample


class FewShotClassifier:
    def __init__(self, kind: str = "knn", k: int = 5, temperature: float = 0.1):
        self.kind = kind
        self.k = k
        self.temperature = temperature
        self._labels: list[str] = []
        self._X: np.ndarray = np.empty((0, 0))
        self._y: list[str] = []

    @property
    def labels(self) -> list[str]:
        return self._labels

    @property
    def ready(self) -> bool:
        """至少要有 2 個類別才有分類意義。"""
        return len(self._labels) >= 2

    def fit(self, examples: list[LabelExample]) -> None:
        """用目前所有種子範例重建分類器（主動學習回訓就是再呼叫一次）。"""
        if not examples:
            self._labels, self._X, self._y = [], np.empty((0, 0)), []
            return
        self._X = np.array([ex.feature for ex in examples], dtype=np.float64)
        self._y = [ex.label for ex in examples]
        self._labels = sorted(set(self._y))

    def predict(self, feature: np.ndarray) -> dict[str, float]:
        """回傳每個類別的機率（加總為 1）。"""
        if not self.ready:
            return {}
        if self.kind == "knn":
            return self._predict_knn(feature)
        if self.kind == "softmax":
            return self._predict_softmax(feature)
        raise ValueError(f"未知的 classifier kind: {self.kind}")

    # ---- kNN：看最近 K 個鄰居的類別比例（提案第 7 頁）----
    def _predict_knn(self, feature: np.ndarray) -> dict[str, float]:
        dists = np.linalg.norm(self._X - feature, axis=1)
        k = min(self.k, len(dists))
        idx = np.argsort(dists)[:k]
        probs = {lab: 0.0 for lab in self._labels}
        # 距離倒數加權投票，越近的鄰居影響越大
        for i in idx:
            w = 1.0 / (dists[i] + 1e-6)
            probs[self._y[i]] += w
        total = sum(probs.values()) or 1.0
        return {lab: round(v / total, 4) for lab, v in probs.items()}

    # ---- softmax：以每類原型（平均特徵）做相似度 softmax ----
    def _predict_softmax(self, feature: np.ndarray) -> dict[str, float]:
        protos = {
            lab: self._X[[i for i, y in enumerate(self._y) if y == lab]].mean(axis=0)
            for lab in self._labels
        }
        labs = self._labels
        sims = np.array([feature @ protos[lab] for lab in labs])  # 特徵已 L2 normalize → cosine
        z = sims / self.temperature
        z -= z.max()
        e = np.exp(z)
        p = e / e.sum()
        return {lab: round(float(v), 4) for lab, v in zip(labs, p)}
