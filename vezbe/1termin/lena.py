import numpy as np
import cv2 as cv

#ocitava sliku koja je u fajlu Lenna.png, BGR
img = cv.imread('Lenna.png')

#svi ukaciju na isto, pa zato pravimo kopiju
img_r = img.copy()
img_g = img.copy()
img_b = img.copy()

#hight, width, kanali
img_r[:, :, 0:2] *= 0 #R
img_g[:, :, 0] *= 0 #G
img_g[:, :, 2] *= 0 #G
img_b[:, :, 1:3] *= 0 #B
imf = img[200:-120, 220:-150] #lice

cv.imshow('Red', img_r)
cv.imshow('Green', img_g)
cv.imshow('Blue', img_b)
cv.imshow('Face', imf)
cv.waitKey(0) #ceka neograniceno dok se ne stisne taster
cv.destroyAllWindows()
