import cv2
import numpy as np
from tracker import Tracker

cap = cv2.VideoCapture(0)

tracker = Tracker()

entry_line = 250

count_in = 0
count_out = 0

detected_ids = set()

while True:
    ret, frame = cap.read()
    if not ret:
        break

    frame = cv2.resize(frame, (640, 480))
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    blur = cv2.GaussianBlur(gray, (5,5), 0)
    _, thresh = cv2.threshold(blur, 200, 255, cv2.THRESH_BINARY)

    contours, _ = cv2.findContours(thresh, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

    detections = []
    for cnt in contours:
        area = cv2.contourArea(cnt)
        if area > 1500:
            x, y, w, h = cv2.boundingRect(cnt)
            detections.append([x, y, w, h])

    boxes_ids = tracker.update(detections)

    for box_id in boxes_ids:
        x, y, w, h, id = box_id
        cx = (x + x + w) // 2
        cy = (y + y + h) // 2

        if id not in detected_ids:
            if cy < entry_line:
                count_in += 1
            else:
                count_out += 1
            detected_ids.add(id)

        cv2.rectangle(frame, (x, y), (x+w, y+h), (0,255,0), 2)
        cv2.putText(frame, str(id), (x, y-10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0,255,0), 2)

    cv2.line(frame, (0, entry_line), (640, entry_line), (255,0,0), 2)

    cv2.putText(frame, "IN: " + str(count_in), (10,40), cv2.FONT_HERSHEY_SIMPLEX, 1, (0,255,0), 2)
    cv2.putText(frame, "OUT: " + str(count_out), (10,80), cv2.FONT_HERSHEY_SIMPLEX, 1, (0,0,255), 2)

    cv2.imshow("Frame", frame)

    if cv2.waitKey(1) == 27:
        break

cap.release()
cv2.destroyAllWindows()