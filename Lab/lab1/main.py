import cv2
import numpy as np
import matplotlib.pyplot as plt
from typing import Any

def BeliKvadrati(matrix, x, y):
    for i in range(-2, 2):
        for j in range(-2, 2):
            matrix[x + i, y + j] = 1


# Load gray image
import os

img_dir = ''
img_name = 'slika_3.png'
img_path = os.path.join(img_dir, img_name)
img = cv2.imread("slika_3.png", cv2.IMREAD_GRAYSCALE)

# Perform 2D Fourier Transform,prebacio je u frekventni domen
f = np.fft.fft2(img)

# Shift the zero-frequency component to the center of the spectrum, i iskoristio fshift
fshift = np.fft.fftshift(f)

# Compute the magnitude spectrum (amplitude)
magnitude_spectrum = 20 * np.log(np.abs(fshift))

# Display the magnitude spectrum
plt.imshow(magnitude_spectrum, cmap='gray')
plt.title('Magnituda spektra pre uklanjanja šuma')
plt.savefig('fft_mag.png')
plt.show()

# ovo rucno gledam samo gde su beli pixeli (4 ih ima)
# ovo su koordinate tih pixela
x1, y1 = 230, 230
x2, y2 = 155, 355
x3, y3 = 355, 155
x4, y4 = 280, 280

BeliKvadrati(fshift, x1, y1)
BeliKvadrati(fshift, x2, y2)
BeliKvadrati(fshift, x3, y3)
BeliKvadrati(fshift, x4, y4)

# Compute the magnitude spectrum (amplitude)
magnitude_spectrum = 20 * np.log(np.abs(fshift))

# Display the magnitude spectrum
plt.imshow(magnitude_spectrum, cmap='gray')
plt.title('Magnituda spektra nakon uklanjanja šuma')
plt.savefig('fft_mag_filtered.png')
plt.show()

# Shift the zero-frequency component back to the top-left corner
f_ishift = np.fft.ifftshift(fshift)

# Compute the inverse Fourier Transform
img_filtered = np.fft.ifft2(f_ishift).real

# Display the filtered image
cv2.imshow('Final image', img_filtered.astype(np.uint8))
cv2.imwrite('output.png', img_filtered)
cv2.waitKey(0)
cv2.destroyAllWindows()