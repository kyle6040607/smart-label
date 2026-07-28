import sys
from pathlib import Path

# 將專案根目錄加入搜尋路徑，避免 ModuleNotFoundError
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

import time
import numpy as np
import torch
import cv2
from PIL import Image

from app.config import Config
from app.ml.yolo_world import YoloWorldDetector
from app.ml.GroundingDINO import GroundingDinoDetector
import groundingdino.datasets.transforms as GD_T
from groundingdino.util.inference import predict as dino_predict

def get_cpu_ram_usage() -> float:
    """取得當前進程佔用的實體記憶體 (RAM)，單位為 MB。"""
    try:
        import psutil
        return psutil.Process().memory_info().rss / (1024 * 1024)
    except ImportError:
        try:
            import ctypes
            class PROCESS_MEMORY_COUNTERS(ctypes.Structure):
                _fields_ = [
                    ("cb", ctypes.c_ulong),
                    ("PageFaultCount", ctypes.c_ulong),
                    ("PeakWorkingSetSize", ctypes.c_size_t),
                    ("WorkingSetSize", ctypes.c_size_t),
                    ("QuotaPeakPagedPoolUsage", ctypes.c_size_t),
                    ("QuotaPagedPoolUsage", ctypes.c_size_t),
                    ("QuotaPeakNonPagedPoolUsage", ctypes.c_size_t),
                    ("QuotaNonPagedPoolUsage", ctypes.c_size_t),
                    ("PagefileUsage", ctypes.c_size_t),
                    ("PeakPagefileUsage", ctypes.c_size_t),
                ]
            GetProcessMemoryInfo = ctypes.windll.psapi.GetProcessMemoryInfo
            GetCurrentProcess = ctypes.windll.kernel32.GetCurrentProcess
            counters = PROCESS_MEMORY_COUNTERS()
            counters.cb = ctypes.sizeof(counters)
            if GetProcessMemoryInfo(GetCurrentProcess(), ctypes.byref(counters), counters.cb):
                return counters.WorkingSetSize / (1024 * 1024)
            return 0.0
        except Exception:
            return 0.0

def run_benchmark():
    print("=" * 100)
    print("  AI Models GPU Benchmark (Multi-Prompt: Cat & Dog Detection & Misclassification Analysis)")
    print("=" * 100)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"[Running Device]: {device.upper()}")
    print("-" * 100)

    cats_dir = PROJECT_ROOT / "data" / "benchmark_cats"
    dogs_dir = PROJECT_ROOT / "data" / "benchmark_dogs"
    cats_dir.mkdir(parents=True, exist_ok=True)
    dogs_dir.mkdir(parents=True, exist_ok=True)

    allowed_ext = {".jpg"}
    cat_image_paths = sorted([p for p in cats_dir.iterdir() if p.is_file() and p.suffix.lower() in allowed_ext])
    dog_image_paths = sorted([p for p in dogs_dir.iterdir() if p.is_file() and p.suffix.lower() in allowed_ext])

    loaded_images = []
    for p in cat_image_paths:
        img_bgr = cv2.imread(str(p))
        if img_bgr is not None:
            loaded_images.append(("cat", p.name, cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)))

    for p in dog_image_paths:
        img_bgr = cv2.imread(str(p))
        if img_bgr is not None:
            loaded_images.append(("dog", p.name, cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)))

    if not loaded_images:
        print("💡 在 `data/benchmark_cats/` 與 `data/benchmark_dogs/` 中未檢測到照片，自動產生模擬圖片...")
        for i in range(3):
            img = np.full((1024, 1024, 3), 240, dtype=np.uint8)
            cv2.circle(img, (400, 500), 200, (100, 150, 50), -1)
            loaded_images.append(("cat", f"simulated_cat_{i+1}.jpg", img))

    print(f"✅ 成功載入共 {len(loaded_images)} 張測試影像 (貓咪: {len(cat_image_paths)} 張, 狗狗: {len(dog_image_paths)} 張)。")

    target_classes = ["cat", "dog"]

    # Phase 1: YOLO-World
    print(f"\n[Phase 1] Benchmarking YOLO-World (v8x-worldv2) on {device.upper()}...")
    yolo_detector = YoloWorldDetector()
    yolo_detector.model.to(device)
    yolo_detector.model.set_classes(target_classes)

    # Phase 2: Grounding DINO
    print(f"\n[Phase 2] Benchmarking Grounding DINO (SwinT_OGC) on {device.upper()}...")
    dino_detector = GroundingDinoDetector()

if __name__ == "__main__":
    run_benchmark()
