import cv2

img = cv2.imread("../intro_to_images/DP.png")
grayScale = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
# cv2.imshow("image", img)
cv2.imshow("grayScale",grayScale)
cv2.waitKey(0)
