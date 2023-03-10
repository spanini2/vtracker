import cv2

videoCapture =  cv2.VideoCapture("../videos/USA_Canada_Highlights.mp4")

while True:
    ret, frame = videoCapture.read()
    if not ret:
        break

    cv2.imshow("frame", frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

videoCapture.release()
cv2.destroyAllWindows