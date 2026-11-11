from ultralytics import YOLO
import cv2
import cvzone

cap = cv2.VideoCapture(0)
model = YOLO('yolo26s.pt')

while True:
    success, frame = cap.read()
    results = model(frame, stream=True)

    # Extracts the bounding box coordinates (x1, y1, x2, y2) of
    # each detected object and calculates its width and height.

    for r in results:
        for box in r.boxes:
            x1, y1, x2, y2 = map(int, box.xyxy[0])
            # height and width
            w, h = x2 - x1, y2 - y1
            # extract confidence (Prediction score how much accurate it is )
            conf = int(box.conf[0] * 100)
            # Get the class ID and retrieve the corresponding object name from YOLO's class labels.
            cls = int(box.cls[0])
            name = model.names[cls]
            # Draw corner-style bounding box around the detected object.
            cvzone.cornerRect(frame, (x1, y1, w, h))
            # Display the detected object's name and confidence score above the bounding box.
            cvzone.putTextRect(frame, f"{name} {conf}", (x1, y1 - 10), scale=1)

        cv2.imshow('My Camera', frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
