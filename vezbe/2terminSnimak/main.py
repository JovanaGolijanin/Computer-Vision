import cv2
import numpy as np

img = cv2.imread("dots.png", cv2.IMREAD_GRAYSCALE)

blur = cv2.GaussianBlur (img, (11, 11), 0)

img = blur

sobx = cv2.Sobel(img, cv2.CV_8U, 1, 0)
soby = cv2.Sobel (img, cv2.CV_8U, 0, 1)

lapl = cv2.Laplacian (img, cv2.CV_8U, ksize=3)

cv2.imshow("Image", img)
cv2.imshow("Sobelx", sobx)
cv2.imshow("Sobely", soby)
cv2.imshow("Laplacian", lapl)
cv2.waitKey (0)
cv2.destroyAllWindows ()