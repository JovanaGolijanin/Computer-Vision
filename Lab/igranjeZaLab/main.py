import cv2

# Učitajte svoju sliku
img = cv2.imread("slika_3.png", cv2.IMREAD_GRAYSCALE)

# Parametri za tensor pravca najmanje promene
length_threshold = 50
distance_threshold = 1.414
canny_low_threshold = 50
canny_high_threshold = 50
canny_aperture_size = 3

# Primena tensora pravca najmanje promene
filtered_img = cv2.ximgproc.createFastLineDetector(length_threshold, distance_threshold, canny_low_threshold, canny_high_threshold, canny_aperture_size).detect(img)

# Prikazivanje rezultirajuće slike
cv2.imshow("Slika bez Pruga", filtered_img)

cv2.waitKey(0)
cv2.destroyAllWindows()
