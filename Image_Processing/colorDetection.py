import cv2
import cvzone
from cvzone.ColorModule import  ColorFinder

cap = cv2.VideoCapture(0)

myColorFinder =ColorFinder(trackBar=False)
hsvVal = {'hmin': 30, 'smin': 0, 'vmin': 0, 'hmax': 51, 'smax': 131, 'vmax': 255}
while True:
    success, frame = cap.read()
    if not success:
        break
    imgColor , mask = myColorFinder.update(frame,hsvVal)

    imgStack = cvzone.stackImages([frame,imgColor,mask],3,0.5)

    cv2.imshow("color detection", imgStack)
    if cv2.waitKey(1) & 0xFF==ord('q'):
        break
