import sys
from pathlib import Path
# 將專案根目錄加入搜尋路徑，避免 ModuleNotFoundError
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

import time
import numpy as np
import torch
import cv2

from app.config import Config
from app.ml.sam import build_segmenter

def get_cpu_ram_usage() -> float:
    """取得當前進程佔用的實體記憶體 (RAM)，單位為 MB。"""
    try:
        import psutil
        return psutil.Process().memory_info().rss / (1024 * 1024)
    except ImportError:
        # Windows Fallback without psutil dependency
        try:
            import os
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

def run_sam_comparison():
    print("=" * 70)
    print("  SAM Benchmark: Mock vs MobileSAM (Cat Images)")
    print("=" * 70)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"[Running Device]: {device.upper()}")
    print("-" * 70)

    # 1. 準備貓咪圖片目錄
    cats_dir = Path("data/benchmark_cats")
    cats_dir.mkdir(parents=True, exist_ok=True)

    # 尋找資料夾下的貓圖
    image_extensions = ("*.png", "*.jpg", "*.jpeg", "*.BMP", "*.PNG", "*.JPG")
    image_paths = []
    for ext in image_extensions:
        image_paths.extend(cats_dir.glob(ext))
    # 去除 Windows 系統下因副檔名大小寫不分（Case-insensitive）造成的重複匹配並排序
    image_paths = sorted(list(set(image_paths)))

    # 如果資料夾為空，自動產生 3 張模擬貓圖（畫有貓咪大小輪廓的測試圖）
    if not image_paths:
        print("💡 在 `data/benchmark_cats/` 中未檢測到照片。")
        print("   自動產生 3 張模擬貓圖（1024x1024 包含幾何色塊）進行基準測試...")
        for i in range(3):
            # 建立背景
            img = np.full((1024, 1024, 3), 240, dtype=np.uint8)
            # 畫上圓形和方形模擬貓咪與背景物品
            cv2.circle(img, (400, 500), 200, (100, 150, 50), -1)  # 模擬貓臉
            cv2.circle(img, (250, 400), 60, (50, 50, 50), -1)     # 模擬貓耳
            cv2.circle(img, (550, 400), 60, (50, 50, 50), -1)     # 模擬貓耳
            cv2.rectangle(img, (700, 700), (950, 950), (120, 80, 200), -1) # 模擬背景紙箱
            
            path = cats_dir / f"simulated_cat_{i+1}.png"
            cv2.imwrite(str(path), img)
            image_paths.append(path)
    else:
        print(f"✅ 成功載入 `data/benchmark_cats/` 下的 {len(image_paths)} 張真實貓圖。")

    # 讀取影像至記憶體（排除磁碟讀取 I/O 速度對 AI 運算時間的干擾）
    loaded_images = []
    for p in image_paths:
        img_bgr = cv2.imread(str(p))
        if img_bgr is not None:
            loaded_images.append(cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB))

    if not loaded_images:
        print("❌ 錯誤：無法讀取 any 測試影像！")
        return

    

    # ----------------- 階段一：測試 Mock 模式 -----------------
    print("\n[Phase 1] Benchmarking Mock Mode...")
    mock_segmenter = build_segmenter(use_real_sam=False)
    
    # 暖機 (Warm-up)
    _ = mock_segmenter.segment(loaded_images[0])

    t0 = time.perf_counter()
    mock_mask_counts = []
    mock_ram_peak = 0.0
    for img in loaded_images:
        masks = mock_segmenter.segment(img)
        mock_mask_counts.append(len(masks))
        # 紀錄運行過程中的最大 RAM 佔用
        mock_ram_peak = max(mock_ram_peak, get_cpu_ram_usage())
    mock_total_time = (time.perf_counter() - t0) * 1000
    mock_latency = mock_total_time / len(loaded_images)
    mock_fps = 1000.0 / mock_latency if mock_latency > 0 else 0.0
    mock_avg_masks = np.mean(mock_mask_counts)

    # ----------------- 階段二：測試 MobileSAM (Real) -----------------
    print("\n[Phase 2] Benchmarking MobileSAM (Real Model) Mode...")
    
    real_success = False
    real_latency = 0.0
    real_fps = 0.0
    real_avg_masks = 0.0
    real_ram_peak = 0.0
    vram_peak = "N/A"
    
    try:
        # 讀取專案預設設定
        cfg = Config()
        real_segmenter = build_segmenter(
            use_real_sam=True,
            checkpoint=cfg.sam_checkpoint,
            points_per_side=cfg.sam_points_per_side,
            min_mask_region_area=cfg.sam_min_mask_region_area
        )
        
        # 使用 inference_mode 加速並減少記憶體消耗
        with torch.inference_mode():
            # 暖機 (Warm-up)
            print("Warming up MobileSAM on GPU/CPU...")
            _ = real_segmenter.segment(loaded_images[0])

            if device == "cuda":
                torch.cuda.synchronize() # 先同步，確保之前的暖機或初始化運算已結束
                torch.cuda.reset_peak_memory_stats() # 再重置記憶體統計

            t0 = time.perf_counter()
            real_mask_counts = []
            real_ram_peak = 0.0
            for img in loaded_images:
                masks = real_segmenter.segment(img)
                real_mask_counts.append(len(masks))
                # 紀錄運行過程中的最大 RAM 佔用
                real_ram_peak = max(real_ram_peak, get_cpu_ram_usage())
                
            if device == "cuda":
                torch.cuda.synchronize() # 同步 GPU，確保所有異步運算均完成，以量測真實時間
                
            real_total_time = (time.perf_counter() - t0) * 1000
            real_latency = real_total_time / len(loaded_images)
            real_fps = 1000.0 / real_latency if real_latency > 0 else 0.0
            real_avg_masks = np.mean(real_mask_counts)
            
            if device == "cuda":
                # 使用 max_memory_reserved() 統計最大預留顯存
                vram_bytes = torch.cuda.max_memory_reserved()
                vram_peak = f"{vram_bytes / (1024 * 1024):.1f} MB"
            else:
                vram_peak = "N/A (CPU)"
                
            real_success = True
    except Exception as e:
        print(f"\n❌ 無法執行 MobileSAM 測試：{e}")
        print("   請確認 `models/` 下已有 SAM 權重檔，或是 Python 已安裝 `segment_anything` 套件。\n")

    # ----------------- 階段三：印出與儲存對比表 -----------------
    print("\n" + "=" * 100)
    print("  COMPARISON BENCHMARK TABLE (Mock vs MobileSAM)")
    print("=" * 100)
    
    
    table_lines = [
        "| 模型模式 (Mode) | 測試圖像數量 | 平均推論時間 (Latency) | 每秒處理幀數 (FPS) | 平均產出遮罩數 (Avg Masks) | 系統記憶體佔用 (CPU RAM) | 顯示卡記憶體峰值 (VRAM Peak) |",
        "| --- | --- | --- | --- | --- | --- | --- |",
        f"| **Mock (模擬)** | {len(loaded_images)} 張 | {mock_latency:.2f} ms ({mock_latency/1000:.4f} s) | {mock_fps:.2f} | {mock_avg_masks:.1f} 塊 | {mock_ram_peak:.1f} MB | 0 MB (無佔用) |"
    ]
    
    if real_success:
        table_lines.append(
            f"| **MobileSAM (真實)** | {len(loaded_images)} 張 | {real_latency:.2f} ms ({real_latency/1000:.2f} s) | {real_fps:.2f} | {real_avg_masks:.1f} 塊 | {real_ram_peak:.1f} MB | {vram_peak} |"
        )
    else:
        table_lines.append(
            f"| **MobileSAM (真實)** | {len(loaded_images)} 張 | 測試失敗 | 測試失敗 | 測試失敗 | 測試失敗 | 測試失敗 |"
        )
        
    print("\n".join(table_lines))
    print("=" * 100)
    
    # 儲存報告
    report_path = Path("data/benchmark_cats/sam_comparison_report.md")
    with open(report_path, "w", encoding="utf-8") as f:
        f.write("# SAM Model Comparison Report (Mock vs MobileSAM)\n\n")
        f.write(f"- **測試設備:** {device.upper()}\n")
        f.write(f"- **圖像來源:** `data/benchmark_cats/` 內的影像 (尺寸為原圖或等比例縮小至 1024x1024)\n\n")
        f.write("\n".join(table_lines) + "\n\n")
        f.write("### ⏱️ 詳細單張與總時間摘要\n")
        f.write(f"- **Mock (模擬)**：\n")
        f.write(f"  - 單張平均推論時間：`{mock_latency:.2f} ms` ({mock_latency/1000:.4f} 秒)\n")
        f.write(f"  - 總運算時間 ({len(loaded_images)}張)：`{mock_latency * len(loaded_images) / 1000:.2f} 秒`\n")
        if real_success:
            f.write(f"- **MobileSAM (真實)**：\n")
            f.write(f"  - 單張平均推論時間：`{real_latency:.2f} ms` ({real_latency/1000:.2f} 秒)\n")
            f.write(f"  - 總運算時間 ({len(loaded_images)}張)：`{real_latency * len(loaded_images) / 1000:.2f} 秒` ({real_latency * len(loaded_images) / 60000:.1f} 分鐘)\n")
    print(f"\n報告已儲存至: {report_path.absolute()}")

if __name__ == "__main__":
    run_sam_comparison()
