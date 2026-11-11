import cv2

img = cv2.imread("DP.png")
# img = cv2.resize(img, (300, 400))
cv2.imshow("Image", img)
cv2.waitKey(0)
# cv2.destroyAllWindows()
