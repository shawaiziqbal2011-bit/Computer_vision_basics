import cv2
from cvzone.FaceDetectionModule import FaceDetector

detector =FaceDetector()


cam = cv2.VideoCapture(0)
while True:
    success , frame = cam.read()

    #framer retuens the detected face
    #bboxes returns the list containing information about the face we have detected
    #bounding boxes contain information like face id ,position , score and center(face center x , y coordinates)
    frame ,bboxes = detector.findFaces(frame)
    if bboxes:
        center = bboxes[0]['center']
    cv2.imshow('frame',frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break