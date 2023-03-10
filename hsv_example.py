import cv2
import numpy as np
import cvzone
from cvzone.ColorModule import ColorFinder


img = cv2.imread("../videos/court_picture.png")
hsvVals = {'hmin': 27, 'smin': 83, 'vmin': 50, 'hmax': 36, 'smax': 255, 'vmax': 202}

ballColorFinder = ColorFinder(False)


while True:
    black = np.zeros((img.shape[0], img.shape[1], 3), dtype=np.uint8)
    imageColor, mask = ballColorFinder.update(img, hsvVals)
    blurFrame = cv2.GaussianBlur(mask, (17, 17), 0)
    contours, _ = cv2.findContours(blurFrame, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    cv2.drawContours(black, contours, -1, (0,255,0), 3)
    stackImages = cvzone.stackImages([img, imageColor, black], 3, 0.5)

    cv2.imshow("img", stackImages)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cv2.destroyAllWindows