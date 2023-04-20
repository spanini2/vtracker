import cv2
import numpy as np
import cvzone


# Setup SimpleBlobDetector parameters.
params = cv2.SimpleBlobDetector_Params()
# Change thresholds
params.minThreshold = 10
params.maxThreshold = 200
# Filter by Area.
params.filterByArea = True
params.minArea = 1500
# Filter by Circularity
params.filterByCircularity = True
params.minCircularity = 0.1
# Filter by Convexity
params.filterByConvexity = True
params.minConvexity = 0.87
# Filter by Inertia
params.filterByInertia = True
params.minInertiaRatio = 0.01

videoCapture =  cv2.VideoCapture("../videos/USA_Canada_Highlights.mp4")
backSub = cv2.createBackgroundSubtractorKNN()
detector = cv2.SimpleBlobDetector_create(params)



while True:
    ret, frame = videoCapture.read()
    if not ret:
        break

    keypoints = detector.detect(frame)
    im_with_keypoints = cv2.drawKeypoints(frame, keypoints, np.array([]), (20, 255, 57), cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)

    # mask = backSub.apply(frame)     
    # dilate = cv2.dilate(mask, None)     
    # blur = cv2.GaussianBlur(dilate, (15, 15),0)     
    # ret, thresh = cv2.threshold(blur,0,255,cv2.THRESH_BINARY | cv2.THRESH_OTSU)

    stackImages = cvzone.stackImages([frame, im_with_keypoints], 2, 0.7)

    cv2.imshow("frame", stackImages)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

videoCapture.release()
cv2.destroyAllWindows