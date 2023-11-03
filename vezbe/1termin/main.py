import numpy as np
import cv2 as cv
cap = cv.VideoCapture(0)
while True:
     retVal, img = cap.read()
     #img = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
     img = cv.medianBlur(img, 7)
     cv.imshow('Camera', img)
     if cv.waitKey(1) & 0xFF == ord('q'):
        break
cap.release()
cv.destroyAllWindows()