"""
YOLO 實例分割 (Segmentation) 雙模型比對與測試腳本
對同一張圖片進行推論，並匯出左右並排對比圖 (result_comparison.jpg)。
"""
import os
import glob
import cv2
import numpy as np
from pathlib import Path
from ultralytics import YOLO


def find_test_image() -> str:
    """尋找一張測試照片（優先測試 benchmark_dogs，其次為 uploads 目錄）。"""
    candidates = [
        Path("data/benchmark_dogs/dog.155.jpg"),
        Path("data/benchmark_dogs/dog.0.jpg"),
    ]
    for c in candidates:
        if c.is_file():
            return str(c.resolve())
            
    imgs = list(Path("data/uploads").glob("*.jpg")) + list(Path("data/uploads").glob("*.png"))
    if imgs:
        return str(imgs[0].resolve())
    return ""


def run_single_inference(image_path: str, model_path: str) -> tuple[np.ndarray, str, int]:
    """對單一圖片與模型執行推論，並使用 Ultralytics 官方原生 result.plot() 繪製標準結果。"""
    m_path = Path(model_path)
    display_name = m_path.name

    actual_path = str(m_path.resolve())
    if not m_path.is_file():
        print(f"⚠️ 指定模型 {display_name} 檔案不存在，自動改用基礎模型 models/yolo26x-seg.pt 代替。")
        actual_path = str(Path("models/yolo26x-seg.pt").resolve())
        display_name = "yolo26x-seg.pt"

    print(f"🚀 正在載入模型 [{display_name}]: {actual_path}")
    model = YOLO(actual_path)
    results = model(image_path)
    result = results[0]

    # 採用 Ultralytics 官方標準原生繪圖 (絕不使用自訂增艷疊圖)
    plotted_bgr = result.plot()

    num_boxes = len(result.boxes) if result.boxes is not None else 0
    num_masks = len(result.masks) if result.masks is not None else 0

    if num_masks > 0:
        summary = f"[{display_name}] 偵測到 {num_boxes} 個物件 (含 {num_masks} 個實例分割遮罩)"
    else:
        summary = f"[{display_name}] 偵測到 {num_boxes} 個物件 (⚠️ 模型未輸出分割遮罩，僅輸出物件框)"

    return plotted_bgr, summary, num_boxes


def compare_models(
    image_path: str = None,
    model1_path: str = "models/yolo26x_seg_20260804_011711.pt",
    model2_path: str = "models/yolo26x-seg.pt",
    output_path: str = "result_comparison.jpg"
):

    if not image_path or not os.path.exists(image_path):
        image_path = find_test_image()
        if not image_path:
            print("❌ 找不到測試圖片！")
            return

    print("==================================================")
    print("🔍 開始進行 YOLO 雙模型推論與效果比對...")
    print(f"🖼️ 測試圖片: {image_path}")
    print(f"🤖 模型 A : {model1_path}")
    print(f"🤖 模型 B : {model2_path}")
    print("==================================================")

    # 分別推論
    img1, sum1, count1 = run_single_inference(image_path, model1_path)
    img2, sum2, count2 = run_single_inference(image_path, model2_path)

    # 確保兩張圖尺寸一致
    h1, w1 = img1.shape[:2]
    h2, w2 = img2.shape[:2]
    if (h1, w1) != (h2, w2):
        img2 = cv2.resize(img2, (w1, h1), interpolation=cv2.INTER_AREA)

    # 上方加上標題條 (Header bar)
    header_h = 50
    header1 = np.zeros((header_h, w1, 3), dtype=np.uint8)
    header2 = np.zeros((header_h, w1, 3), dtype=np.uint8)

    title1 = f"Model A: {Path(model1_path).name} ({count1} objs)"
    title2 = f"Model B: {Path(model2_path).name} ({count2} objs)"

    cv2.putText(header1, title1, (10, 32), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 2)
    cv2.putText(header2, title2, (10, 32), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 2)

    col1 = np.vstack([header1, img1])
    col2 = np.vstack([header2, img2])

    # 左右拼貼左右對比圖
    comparison_img = np.hstack([col1, col2])

    # 存檔
    cv2.imwrite(output_path, comparison_img)
    print("==================================================")
    print(f"🎉 比對完成！")
    print(f"📊 {sum1}")
    print(f"📊 {sum2}")
    print(f"🖼️ 左右對比圖已成功匯出至: {os.path.abspath(output_path)}")
    print("==================================================")


if __name__ == "__main__":
    import sys
    kwargs = {}
    if len(sys.argv) > 1 and sys.argv[1]:
        kwargs["image_path"] = sys.argv[1]
    if len(sys.argv) > 2 and sys.argv[2]:
        kwargs["model1_path"] = sys.argv[2]
    if len(sys.argv) > 3 and sys.argv[3]:
        kwargs["model2_path"] = sys.argv[3]

    compare_models(**kwargs)
