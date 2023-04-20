import cv2
import numpy as np

videoCapture =  cv2.VideoCapture("../videos/USA_Canada_Highlights.mp4")

while True:
    ret, frame = videoCapture.read()
    if not ret:
        break

    Z = np.float32(frame.reshape((-1,3)))

    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 10, 1.0)
    K = 4
    _,labels,centers = cv2.kmeans(Z, K, None, criteria, 10, cv2.KMEANS_RANDOM_CENTERS)
    labels = labels.reshape((frame.shape[:-1]))
    reduced = np.uint8(centers)[labels]

    result = [np.hstack([frame, reduced])]
    for i, c in enumerate(centers):
        mask = cv2.inRange(labels, i, i)
        mask = np.dstack([mask]*3) # Make it 3 channel
        # ex_img = cv2.bitwise_and(frame, mask)
        # ex_reduced = cv2.bitwise_and(reduced, mask)
        # result.append(np.hstack([ex_img, ex_reduced]))

    cv2.imshow("frame", np.vstack(result))

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

videoCapture.release()
cv2.destroyAllWindows