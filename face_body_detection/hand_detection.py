import cv2
from cvzone.HandTrackingModule import HandDetector

detector =HandDetector(detectionCon=0.8, maxHands=2)


cam = cv2.VideoCapture(0)
while True:
    success , frame = cam.read()

    #framer returns the detected face
    #bboxes returns the list containing information about the face we have detected
    #bounding boxes contain information like face id ,position , score and center(face center x , y coordinates)
    hands ,frame = detector.findHands(frame)
    if hands:
        hand1 =hands[0]
        #landmarklist have 21 points on a hand
        lmlist = hand1['lmList']
        #left or right hand
        handType = hand1['type']

        print("This is landmark list ",lmlist)
        print("Hand Type ",handType)


    cv2.imshow('hand detection',frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break