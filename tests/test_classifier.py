"""FewShotClassifier 測試：單一類別的相似度信心 + 多類別機率行為。"""
from __future__ import annotations

import numpy as np
import pytest

from app.ml.classifier import FewShotClassifier
from app.models import LabelExample


def _unit(v) -> np.ndarray:
    v = np.asarray(v, dtype=np.float64)
    return v / np.linalg.norm(v)


def _ex(label: str, vec) -> LabelExample:
    return LabelExample(label=label, feature=_unit(vec).tolist())


def test_not_ready_without_examples():
    clf = FewShotClassifier()
    assert not clf.ready
    assert clf.predict(_unit([1, 0, 0])) == {}


@pytest.mark.parametrize("kind", ["knn", "softmax"])
def test_single_class_uses_similarity_as_confidence(kind):
    """單一類別就能運作：像種子 → 高信心；不像 → 低信心（不會恆為 1.0）。"""
    clf = FewShotClassifier(kind=kind)
    clf.fit([_ex("cat", [1, 0, 0]), _ex("cat", [0.9, 0.1, 0])])
    assert clf.ready

    like = clf.predict(_unit([1, 0.05, 0]))["cat"]
    unlike = clf.predict(_unit([0, 0, 1]))["cat"]
    assert like > 0.9
    assert unlike < 0.2
    assert 0.0 <= unlike < like <= 1.0


def test_multiclass_probs_unchanged():
    """多類別維持原本的相對機率行為：加總為 1、較近的類別機率較高。"""
    clf = FewShotClassifier()
    clf.fit([_ex("cat", [1, 0]), _ex("dog", [0, 1])])
    probs = clf.predict(_unit([1, 0.1]))
    assert set(probs) == {"cat", "dog"}
    assert abs(sum(probs.values()) - 1.0) < 1e-6
    assert probs["cat"] > probs["dog"]
