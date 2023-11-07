import cv2
import numpy as np

# Učitavanje slike
slika = cv2.imread("coins.png")

# Pretvaranje slike u Grayscale (nijanse sive)
siva_slika = cv2.cvtColor(slika, cv2.COLOR_BGR2GRAY)

# Ručno postavljanje praga za segmentaciju svih novčića
prag = 150  # Prilagodite vrednost prema potrebi

# Thresholding za izdvajanje svih novčića
_, maska_srebrni = cv2.threshold(siva_slika, prag, 255, cv2.THRESH_BINARY)

# Morfološka operacija za popunjavanje rupa unutar novčića
kernel = np.ones((5, 5), np.uint8)
maska_srebrni = cv2.morphologyEx(maska_srebrni, cv2.MORPH_CLOSE, kernel)

# Pretvaranje originalne slike u HSV prostor boja
hsv_slika = cv2.cvtColor(slika, cv2.COLOR_BGR2HSV)

# Izdvajanje Saturation kanala
saturation_kanal = hsv_slika[:, :, 1]

# Ručno postavljanje praga za segmentaciju markera (bakarnog novčića)
prag_bakar = 50  # Prilagodite vrednost prema potrebi

# Thresholding za izdvajanje markera (bakarnog novčića)
_, marker = cv2.threshold(saturation_kanal, prag_bakar, 255, cv2.THRESH_BINARY)

# Morfološka operacija za filtriranje nepotrebnih piksela
maska_bakar = cv2.morphologyEx(marker, cv2.MORPH_CLOSE, kernel)

# Izdvajanje bakarnog novčića koristeći morfološku rekonstrukciju
maska_bakar = cv2.bitwise_not(maska_bakar)
marker = cv2.bitwise_and(maska_srebrni, maska_bakar)
bakarni_novcic = cv2.bitwise_and(slika, slika, mask=marker)

# Prikazivanje rezultata
cv2.imshow("Izlazna maska bakarnog novcica", marker)
cv2.imshow("Bakarni novcic", bakarni_novcic)
cv2.waitKey(0)
cv2.destroyAllWindows()




"""
import cv2
import numpy as np

# Učitavanje slike
slika = cv2.imread("coins.png")

# Pretvaranje slike u HSV prostor boja
hsv_slika = cv2.cvtColor(slika, cv2.COLOR_BGR2HSV)

# Izdvajanje Saturation kanala
saturation_kanal = hsv_slika[:, :, 1]

# Ručno postavljanje praga za segmentaciju bakarnog novčića
prag = 100  # Prilagodite vrednost prema potrebi

# Thresholding za izdvajanje bakarnog novčića
ret, maska = cv2.threshold(saturation_kanal, prag, 255, cv2.THRESH_BINARY)

# Morfološka operacija za filtriranje nepotrebnih piksela
kernel = np.ones((5, 5), np.uint8)
maska = cv2.morphologyEx(maska, cv2.MORPH_OPEN, kernel)

# Morfološka rekonstrukcija za izdvajanje bakarnog novčića
marker = maska.copy()
cv2.dilate(marker, kernel, iterations=2)
cv2.erode(marker, kernel, iterations=2)

# Morfološka rekonstrukcija
maska = cv2.bitwise_not(marker)
bakarni_novcic = cv2.bitwise_and(saturation_kanal, saturation_kanal, mask=maska)

# Rezultujuća maska sa bakarnim novčićem
cv2.imshow("Maska bakarnog novcica", bakarni_novcic)
cv2.waitKey(0)
cv2.destroyAllWindows()

"""
