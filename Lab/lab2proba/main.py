import cv2
import numpy as np

# Load the input image
image = cv2.imread('coins.png')

# Convert the image to grayscale
# Grayscale slika se koristi jer će se obično segmentacija
# i analiza objekata vršiti na ovakvim jednobojnim slikama
gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

# Apply a threshold to the grayscale image to convert it into a binary image
# Ovde se postavlja prag na 150, što znači da će sve piksele sa intenzitetom
# većim od 150 biti postavljeni na 0 (crno), dok će svi pikseli ispod praga biti
# postavljeni na 255 (bela). Upotrebom cv2.THRESH_BINARY_INV se postiže
# inverzni threshold, što znači da se crni i beli pikseli zamjenjuju.
thresh = cv2.threshold(gray, 150, 255, cv2.THRESH_BINARY_INV)[1]

# Define a kernel for morphological operations
kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (2,2))

# Perform morphological dilation, erosion, closing, and opening operations on the binary image
# Ovo pomaže u uklanjanju šuma i povezivanju delova novčića.
closing = cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, kernel)
opening = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, kernel)

# Find contours (boundaries of connected regions) in the opened binary image
# Pronađene konture predstavljaju granice povezanih oblasti na slici.
cnts = cv2.findContours(closing, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
cnts = cnts[0] if len(cnts) == 2 else cnts[1]

# Create a mask from the filtered binary image to keep only the regions of interest
mask = np.zeros_like(closing)

# Iterate over the detected contours and filter out objects with the same color
# Ako je razlika u boji između objekta i pozadine veća od 50,
# objekat se zadržava u maski.
# Na ovaj način se filtriraju objekti koji su značajno različiti u boji od pozadine.
for c in cnts:
    # Compute the average color of the object
    mask.fill(0)
    cv2.drawContours(mask, [c], -1, 255, -1)
    avg_color = cv2.mean(image, mask=mask)[:3]

    # Check if the color of the object is significantly different from the background color
    bg_color = [128, 128, 128]
    color_diff = np.linalg.norm(np.array(avg_color) - np.array(bg_color))
    if color_diff > 50:
        cv2.drawContours(mask, [c], -1, 255, -1)

# Show the mask
cv2.imshow('coin_mask.png', mask)

# Apply the mask to the original image to obtain the masked image
# Originalna slika se osenčava na masku kako bi se dobio rezultat
# koji sadrži samo prepoznate objekte.
result = cv2.bitwise_and(image, image, mask=mask)

# Show the masked image
cv2.imshow('masked_coins.png', result)

cv2.waitKey(0)
cv2.destroyAllWindows()
