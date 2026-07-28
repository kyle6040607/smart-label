import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

import time
import numpy as np
import torch
import cv2

from app.ml.yolo_world import YoloWorldDetector
from app.ml.GroundingDINO import GroundingDinoDetector

def run_accuracy_benchmark():
    print("=" * 90)
    print("  Cat Dataset Detection Accuracy & Confidence Benchmark")
    print("=" * 90)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"[Device]: {device.upper()}")

    cats_dir = PROJECT_ROOT / "data" / "benchmark_cats"
    image_paths = sorted(
        p for p in cats_dir.iterdir()
        if p.is_file() and p.suffix.lower() == ".jpg"
    )

    print(f"✅ 載入 {len(image_paths)} 張實體貓咪照片進行準確度評估。")

if __name__ == "__main__":
    run_accuracy_benchmark()
