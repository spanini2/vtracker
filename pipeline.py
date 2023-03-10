import cv2
import numpy as np
import cvzone
from cvzone.ColorModule import ColorFinder

videoCapture =  cv2.VideoCapture("../videos/USA_Canada_Highlights.mp4")

hsvVals = {'hmin': 27, 'smin': 83, 'vmin': 50, 'hmax': 36, 'smax': 255, 'vmax': 202}
ballColorFinder = ColorFinder(True)

while True:
    ret, frame = videoCapture.read()
    if not ret:
        break

    black = np.zeros((frame.shape[0], frame.shape[1], 3), dtype=np.uint8)
    contour_black = np.zeros((frame.shape[0], frame.shape[1], 3), dtype=np.uint8)
    imageColor, mask = ballColorFinder.update(frame, hsvVals)
    blurFrame = cv2.GaussianBlur(mask, (17, 17), 0)
    contours, _ = cv2.findContours(blurFrame, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    cv2.drawContours(frame, contours, -1, (0,255,0), 3)
    cv2.drawContours(contour_black, contours, -1, (255,255,255), 3)
    blur_contour = cv2.GaussianBlur(contour_black, (17, 17), 0)

    # rows = contour_black.shape[0]
    # circles = cv2.HoughCircles(contour_black, cv2.HOUGH_GRADIENT, 1, rows / 8,
    #                            param1=100, param2=30,
    #                            minRadius=1, maxRadius=300)
    
    # if circles is not None:
    #     circles = np.uint16(np.around(circles))
    #     for i in circles[0, :]:
    #         center = (i[0], i[1])
    #         print(center)
    #         cv2.circle(black, center, 1, (0, 100, 100), 3)
    #         # circle outline
    #         radius = i[2]
    #         cv2.circle(black, center, radius, (255, 0, 255), 3)

    stackImages = cvzone.stackImages([frame, mask, contour_black, blur_contour, imageColor], 3, 0.5)

    cv2.imshow("frame", stackImages)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

videoCapture.release()
cv2.destroyAllWindows