import cv2
import cvzone as cvz
import numpy as np

img = cv2.imread("shapes.jpg")

grayScale = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

#Detects edges
imgCanny = cv2.Canny(grayScale, 100, 200)
#Make Edges thicker
imageDilated =cv2.dilate(imgCanny,np.ones((4,4),np.uint8),iterations=1)
imgContours,conFound =cvz.Utils.findContours(img,imageDilated,filter=None,drawCon=True)

imgStack = cvz.stackImages([img, grayScale, imgCanny,imageDilated,imgContours], 3, 1)

cv2.imshow("Image", imgStack)

print(imgContours)
cv2.waitKey(0)
