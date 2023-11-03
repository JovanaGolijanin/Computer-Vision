import cv2
import numpy as np
import matplotlib.pyplot as plt

def fft(img):
    img_fft = np.fft.fft2(img)  # pretvaranje slike u frekventni domen (FFT - Fast Fourier Transform), fft2 je jer je u 2 dimenzije
    img_fft = np.fft.fftshift(img_fft)  # pomeranje koordinatnog pocetka u centar slike
    return img_fft

def inverse_fft(magnitude_log, complex_moduo_1):
    img_fft = complex_moduo_1 * np.exp(magnitude_log) # vracamo magnitudu iz logaritma i mnozimo sa kompleksnim brojevima na slici
    img_filtered = np.abs(np.fft.ifft2(img_fft)) # funkcija ifft2 vraca sliku iz frekventnog u prostorni domen, nije potrebno raditi ifftshift jer to se implicitno izvrsava
                                                 # rezultat ifft2 je opet kompleksna slika, ali nas zanima samo moduo jer to je nasa slika zato opet treba np.abs()

    return img_filtered


def fft_noise_addition(img, center):
    img_fft = fft(img)
    img_fft_mag = np.abs(img_fft)  # slika u frekventnom domenu je kompleksan broj, nama je potrebna amplituda tog kompleksnog broja (odnosno moduo sto daje funkcija np.abs())

    img_mag_1 = img_fft / img_fft_mag  # cuvanje kompleksnih brojeva sa jedinicnim moduom, jer cemo da menjamo amplitudu

    img_fft_log = np.log(img_fft_mag)  # vrednosti su prevelike da bi se menjale direktno, pa je logaritam dobar nacin da se vizualizuje amplituda frekventnog domena.
    img_fft_log[center[0] - 50, center[1] - 50] = 16
    img_fft_log[center[0] + 50, center[1] + 50] = 16
    img_filtered = inverse_fft(img_fft_log, img_mag_1)

    return img_filtered

def low_pass_filter(img, center, radius):
    img_fft = fft(img)
    img_fft_mag = np.abs(img_fft)
    img_mag_1 = img_fft / img_fft_mag
    img_fft_log = np.log(img_fft_mag)
    for x in range(img.shape[0]):
        for y in range(img.shape[1]):
            if (x-center[0])*(x-center[0]) + (y-center[1])*(y-center[1]) > radius*radius: # jednacina kruga, zelimo da sve izvan kruga bude nula sto predstavlja idealni low pass filter
                img_fft_log[x,y] = 0

    plt.imshow(img_fft_log)
    plt.show()

    img_filtered = inverse_fft(img_fft_log, img_mag_1)

    return img_filtered

def high_pass_filter(img, center, radius):
    img_fft = fft(img)
    img_fft_mag = np.abs(img_fft)
    img_mag_1 = img_fft / img_fft_mag
    img_fft_log = np.log(img_fft_mag)
    for x in range(img.shape[0]):
        for y in range(img.shape[1]):
            if (x-center[0])*(x-center[0]) + (y-center[1])*(y-center[1]) < radius*radius: # jednacina kruga, zelimo da sve unutar kruga bude nula sto predstavlja idealni high pass filter
                img_fft_log[x,y] = 0

    plt.imshow(img_fft_log)
    plt.show()

    img_filtered = inverse_fft(img_fft_log, img_mag_1)

    return img_filtered


def filtering_in_spatial_domain(img):
    gaus_kernel = cv2.getGaussianKernel(ksize=21, sigma=7) # Gausov kernel velicine 21x21 sa standardnom devijacijom 7
    kernel = np.zeros((3, 3), dtype=np.int8) # custom 3x3 kernel popunjem svim nulama
    kernel[1, 2] = 2 # dodaje se jedinica na u srednjem redu desno, pa kernel postaje: [[0 0 0] [0 0 1] [0 0 0]]
    img_gauss_blur = cv2.filter2D(img, -1, gaus_kernel) # gausov kernel
    img_filter_custom = cv2.filter2D(img, -1, kernel) # custom kernel

    return img_gauss_blur, img_filter_custom


if __name__ == '__main__':
    img = cv2.imread("slika_3.png") # OpenCV ucitava sliku u formatu BGR
    img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY) # pretvaranje slike u nijanse sive (grayscale)

    center = (256, 256) # slika je 512x512 pa ce centar biti na poziciji (256, 256)
    radius = 50 # radius idealnog low ili high pass filtra

    img_gauss, img_custom = filtering_in_spatial_domain(img) # filtriranje u prostornom domenu gausovim i custom filtrom

    plt.imshow(img_gauss, cmap='gray')
    plt.show()
    plt.imshow(img_custom, cmap='gray')
    plt.show()

    img_noise_added = fft_noise_addition(img, center) # dodavanje periodicnog suma
    plt.imshow(img_noise_added, cmap='gray')
    plt.show()

    img_low_pass = low_pass_filter(img, center, radius) # nisko propusni filter
    plt.imshow(img_low_pass, cmap='gray')
    plt.show()

    img_high_pass = high_pass_filter(img, center, radius) # visoko propusni filter
    plt.imshow(img_high_pass, cmap='gray')
    plt.show()