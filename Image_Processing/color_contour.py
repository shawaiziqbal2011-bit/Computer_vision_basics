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

    # Color detection
    imgColor, mask = myColorFinder.update(frame, hsvVal)

    # Create a copy for contour detection
    imgContours = frame.copy()

    # Find contours from the mask
    contours, hierarchy = cv2.findContours(
        mask,
        cv2.RETR_EXTERNAL,
        cv2.CHAIN_APPROX_SIMPLE
    )

    # Draw detected contours
    cv2.drawContours(imgContours, contours, -1, (0, 255, 0), 3)

    # Stack all four images
    imgStack = cvzone.stackImages(
        [frame, imgColor, mask, imgContours], 2, 0.5
    )

    cv2.imshow("Color Detection", imgStack)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break