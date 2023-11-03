import cv2
import numpy as np

cap = cv2.VideoCapture(0)

while True:
    _, frame = cap.read()

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    blur = cv2. GaussianBlur(gray, (25, 25), 1)

    sobX = cv2.Sobel(blur, cv2.CV_8U, 1, 0)

    sobY = cv2.Sobel(blur, cv2.CV_8U, 0, 1)

    cv2.imshow("Frame", frame)
    #cv2.imshow("Gray", gray)
    #cv2.imshow("Blurred Frame", blur)
    cv2.imshow("Sobel H", sobX)
    #cv2.imshow("Sobel Y", sobX)

    key = cv2.waitKey(1)
    if key == 27:
        break
cap.release()
cv2.destroyAllWindows()