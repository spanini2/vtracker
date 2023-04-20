import cv2
import numpy as np
import cvzone

videoCapture =  cv2.VideoCapture("../videos/USA_Canada_Highlights.mp4")

mser = cv2.MSER_create()

while True:
    ret, frame = videoCapture.read()
    if not ret:
        break

    regions, _ = mser.detectRegions(frame)

    # for region in regions:
    #     x, y, w, h = cv2.boundingRect(region)
    #     cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
    
    hulls = [cv2.convexHull(p.reshape(-1, 1, 2)) for p in regions]
    cv2.polylines(frame, hulls, 1, (0, 255, 0))

    cv2.imshow("frame", frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

videoCapture.release()
cv2.destroyAllWindows