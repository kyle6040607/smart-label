"""SamSegmenter 的建構測試，不需真實權重。"""
from __future__ import annotations

import sys
import types

import pytest

from app.ml.sam import SamSegmenter, build_segmenter


class _FakeSam:
    def __init__(self, checkpoint: str):
        self.checkpoint = checkpoint

    def to(self, *, device: str):
        self.device = device
        return self

    def eval(self):
        return self


class _FakeAutomaticMaskGenerator:
    def __init__(self, sam, **kwargs):
        self.sam = sam
        self.kwargs = kwargs


class _FakePredictor:
    def __init__(self, sam):
        self.sam = sam


@pytest.fixture
def fake_mobile_sam(monkeypatch):
    calls: list[tuple[str, str]] = []

    def factory(model_type: str):
        def build(*, checkpoint: str):
            calls.append((model_type, checkpoint))
            return _FakeSam(checkpoint)
        return build

    module = types.SimpleNamespace(
        sam_model_registry={"vit_t": factory("vit_t"), "vit_h": factory("vit_h")},
        SamAutomaticMaskGenerator=_FakeAutomaticMaskGenerator,
        SamPredictor=_FakePredictor,
    )
    monkeypatch.setitem(sys.modules, "mobile_sam", module)

    import torch
    monkeypatch.setattr(torch.cuda, "is_available", lambda: False)
    return calls


def test_model_type_is_explicit_and_independent_of_checkpoint_name(fake_mobile_sam):
    segmenter = build_segmenter(
        use_real_sam=True,
        checkpoint="models/model.pt",
        model_type="vit_t",
    )

    assert isinstance(segmenter, SamSegmenter)
    assert fake_mobile_sam == [("vit_t", "models/model.pt")]


def test_unknown_model_type_fails_before_loading_checkpoint(fake_mobile_sam):
    with pytest.raises(ValueError, match="SAM_MODEL_TYPE.*vit_x.*vit_h, vit_t"):
        SamSegmenter(checkpoint="models/model.pt", model_type="vit_x")

    assert fake_mobile_sam == []
