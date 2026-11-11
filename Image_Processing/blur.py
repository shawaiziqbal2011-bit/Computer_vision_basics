import cv2

img = cv2.imread("../intro_to_images/DP.png")
blur = cv2.GaussianBlur(img,(15,15),0)
# cv2.imshow("image", img)
cv2.imshow("blur",blur)
cv2.waitKey(0)
