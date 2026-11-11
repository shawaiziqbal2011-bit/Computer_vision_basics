import cv2
from cvzone.PoseModule import PoseDetector

detector = PoseDetector()

cam = cv2.VideoCapture(0)

while True:
    success, frame = cam.read()

    if not success:
        break

    # Detect pose
    frame = detector.findPose(frame)

    # Get landmark list and bounding box
    lmlist, bboxInfo = detector.findPosition(frame)

    if bboxInfo:
        center = bboxInfo['center']
        print("Center:", center)

    cv2.imshow('Pose Detection', frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break