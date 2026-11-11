import cv2

img = cv2.imread("../intro_to_images/DP.png")

#[y1:y2,x1:x2]
cv2.imshow("crop", img[100:200,50:100])
# cv2.imshow("crop", img)

cv2.waitKey(0)
