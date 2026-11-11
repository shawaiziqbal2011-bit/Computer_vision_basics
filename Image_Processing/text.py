import cv2

img = cv2.imread("../intro_to_images/DP.png")

putText = cv2.putText(img,"Hello from user",(400,100),cv2.FONT_HERSHEY_DUPLEX,1,(0,200,0),2)

cv2.imshow("text", putText)

cv2.waitKey(0)
