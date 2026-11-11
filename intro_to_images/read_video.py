import cv2

cap = cv2.VideoCapture("test_video.mp4")
while True:
    success, frame = cap.read()
    if not success:
        break
    cv2.imshow("frame", frame)
    cv2.waitKey(1)
