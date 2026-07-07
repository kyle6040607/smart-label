from pathlib import Path

import cv2

from app.models import ImageRecord, Segment
from app.repository import Repository

repo = Repository(Path("data/db.json"))

# 讀圖片取得尺寸
img_path = Path("data/uploads/cat1.jpg")
image = cv2.imread(str(img_path))

if image is None:
    raise FileNotFoundError("找不到 data/uploads/cat.jpg")

h, w = image.shape[:2]

# 建立 ImageRecord
img = ImageRecord(
    filename="cat1.jpg",
    path=str(img_path),
    width=w,
    height=h,
)

repo.add_image(img)

# 建立 Segment
seg = Segment(
    image_id=img.id,
    mask_path="data/masks/cat1_mask.png",

    # 先給一個簡單 bbox
    bbox=(0, 0, w, h),

    # 面積先填一個值即可
    area=w * h,

    # 表示人工已經標成 cat
    human_label="cat",
    reviewed=True,
)

repo.add_segment(seg)

print("建立測試資料完成")
print("Image ID:", img.id)
print("Segment ID:", seg.id)