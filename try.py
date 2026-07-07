import cv2

img = cv2.imread("data/uploads/cat1.jpg")
mask = cv2.imread("data/masks/cat1_mask.png",0)

print("image:", img.shape)
print("mask :", mask.shape)