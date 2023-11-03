import numpy as np
import cv2
import matplotlib.pyplot as plt


def read_and_convert_to_other_color_space(image_path, color_space):
    img = cv2.imread(image_path)
    return cv2.cvtColor(img, color_space)


def show_image_with_matplotlib(image, gray):
    if gray:
        plt.imshow(image, cmap='gray')  # ukoliko se parametar cmap postavi na 'gray' tada ce se prikazati slika kao nijansa sive
    else:
        plt.imshow(image)  # ukoliko se ne postavi parametar onda ce se slika prikazati kao spektar boja izmedju plave i zute. Zutu boju ce dobiti piksel sa najvecom vrednoscu
        # a plavu boju piksel sa najmanjom vrednoscu, ostali ce biti izmedju. Korisno je kada su vrednosti piksela bliske i sve izgleda isto na slici.
    plt.show()


def show_image_with_opencv(image):
    cv2.imshow("IMAGE", image)  # prvi parametar je ime prozora
    cv2.waitKey(0)  # ova funkcija sluzi da zaustavi zatvaranje prethodnog prozora, slicno kao kad se u c++ stavi na kraju programa da ocekujemo nesto na ulazu


def save_image(image: np.ndarray, save_path: str):  # image: np.ndarray znaci da je parametar image i da ce biti tipa np.ndarray (ovo je samo dokumentovanje, interpreter ne vidi to)
    cv2.imwrite(save_path, image)


if __name__ == '__main__':  # main, nije potreban ali je korisno da kod bude pregledniji (ne mora da se pise cela ova linija, dovoljno je samo da se napise main i pritisne TAB i autocomplete uradi ostatak)
    lena = read_and_convert_to_other_color_space("lena_rgb.png", cv2.COLOR_BGR2GRAY)
    show_image_with_matplotlib(lena, True)
    show_image_with_opencv(lena)
    save_image(lena,
               "../lena_gray.png")  # iako je slika u grayscalu, njenim cuvanjem ce se pretvoriti u RGB sliku koja ima samo sive boje
    # dakle ako neki piksel ima vrednost npr. 126 u grayscale, nakon cuvanja ce imati vrednost otprilike [126, 127, 125] (nisam proverio brojeve, ali je bitna poenta)

    # precice na tastaturi: Ctrl + Alt + L -> uredjuje kod po standardima, Ctrl + P -> pokazuje trenutan parametar funkcije koji treba uneti
    # za otvaranje foldera gde se nalazi projekat, desni klik na ime projekta sa leve strane i opcija Open In -> Explorer (za Linux zavisi, ali obicno je Files)