import cv2 as cv
import numpy as np

#other test images are in jpg format
image_path = 'slika4.png'

# Loading the image
original_image = cv.imread(image_path)

# Converting the image to grayscale
gray_image = cv.cvtColor(original_image, cv.COLOR_BGR2GRAY)

# Thresholding the image
threshold_value = 40
_, binary_image = cv.threshold(gray_image, threshold_value, 255, cv.THRESH_BINARY)

# Creating marker image
marker = binary_image.copy()
marker[1:-1, 1:-1] = 0

# Performing morphological reconstruction
kernel = np.ones((3, 3), np.uint8)
while True:
    tmp = marker.copy()
    marker = cv.dilate(marker, kernel)
    marker = cv.min(binary_image, marker)
    difference = cv.subtract(marker, tmp)
    if cv.countNonZero(difference) == 0:
        break

# Creating mask and apply it to the original image
mask = cv.bitwise_not(marker)
mask_color = cv.cvtColor(mask, cv.COLOR_GRAY2BGR)
result_image = cv.bitwise_and(original_image, mask_color)

# Display the final image
cv.imshow('Modified Final Image', result_image)
cv.waitKey(0)
cv.destroyAllWindows()