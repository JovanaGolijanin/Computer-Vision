# import the necessary packages
import numpy as np
import argparse
import time
import cv2
import imutils


def pyramid(imageP, scale=1.5, minSize=(30, 30)):
    # yield the original image
    yield imageP
    # keep looping over the pyramid
    while True:
        # compute the new dimensions of the image and resize it
        w = int(imageP.shape[1] / scale)
        imageP = imutils.resize(imageP, width=w)
        # if the resized image does not meet the supplied minimum
        # size, then stop constructing the pyramid
        #cv2.imshow(f"smanjena na {imageP.shape[0], imageP.shape[1]}", imageP)

        if imageP.shape[0] < minSize[1] or imageP.shape[1] < minSize[0]:
            break
        # yield the next image in the pyramid
        yield imageP


def iseci_sliku(img, w, h):
    # Konvertuje sliku u crno-beli format
    img1 = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    # Primačuje sliku kroz binarnu obradu da bi dobili konture
    ret, thresh = cv2.threshold(img1, 127, 255, cv2.THRESH_BINARY)

    #Pronalazi konture u binarnoj slici
    contours = cv2.findContours(thresh, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)[-2]

    # Koordinate za rezultat
    coords = [0, 0]
    difX = h
    difY = w

    # Iteracija kroz konture
    for cont in contours:
        # Provera površine konture
        if cv2.contourArea(cont) >= w * h:
            arc_len = cv2.arcLength(cont, True)
            approx = cv2.approxPolyDP(cont, 0.1 * arc_len, True)
            # Ako je kontura pravougaonik
            if len(approx) == 4:
                l_x, l_y = [], []

                # Izdvajanje koordinata temena pravougaonika
                for this_approx in approx:
                    cord_y = this_approx[0][0]
                    cord_x = this_approx[0][1]

                    l_y.append(cord_y)
                    l_x.append(cord_x)

                # Određivanje granica pravougaonika
                startX = np.min(l_x)
                endX = np.max(l_x)
                startY = np.min(l_y)
                endY = np.max(l_y)

                # Razlika u dimenzijama pravougaonika i ciljanih dimenzija
                pomDifX = abs(h - (endX - startX))
                pomDifY = abs(w - (endY - startY))

                # Ažuriranje rezultata ako su razlike manje od prethodnih
                if pomDifX < difX and pomDifY < difY:
                    difX = pomDifX
                    difY = pomDifY
                    coords[0] = startX
                    coords[1] = startY
                    break

    # Isecanje regiona iz slike
    return img[coords[0]:coords[0] + h, coords[1]:coords[1] + w]


window_size = 180
step_size = 180
width = 1440
height = 720

# construct the argument parse and parse the arguments
ap = argparse.ArgumentParser()
ap.add_argument("-i", "--image", required=True, help="path to input image")
ap.add_argument("-p", "--prototxt", required=True, help="path to Caffe 'deploy' prototxt file")
ap.add_argument("-m", "--model", required=True, help="path to Caffe pre-trained model")
ap.add_argument("-l", "--labels", required=True, help="path to ImageNet labels (i.e., syn-sets)")
args = vars(ap.parse_args())

# load the input image from disk
image = cv2.imread(args["image"])
image = cv2.imread("download.jpg")

# cv2.imshow("ucitana", image)
image = iseci_sliku(image, width, height)
# cv2.imshow("vracena", image)

# Učitavanje GoogleNet modela neuronske mreže
# load the class labels from disk
rows = open(args["labels"]).read().strip().split("\n")
classes = [r[r.find(" ") + 1:].split(",")[0] for r in rows]
net = cv2.dnn.readNetFromCaffe(args["prototxt"], args["model"])

for resized in pyramid(image, scale=2.0, minSize=(350, 100)):
    #sliding window starts
    for y in range(0, resized.shape[0], step_size):
        for x in range(0, resized.shape[1], step_size):

            croppedImage = resized[y:y + window_size, x:x + window_size]
            blob = cv2.dnn.blobFromImage(croppedImage, 1, (224, 224), (104, 117, 123))
            net.setInput(blob)
            preds = net.forward()

            #ovaj deo menjam
            #idxs = np.argsort(preds[0])[::-1][:1]
            #idx = idxs[0]
            idxs = np.argsort(preds[0])[::-1][:2]  # Захтевајте две највероватније лабеле
            idx1, idx2 = idxs
            if preds[0][idx1] > preds[0][idx2]:
                idx = idx1
            else:
                idx = idx2

            if preds[0][idx] > 0.5:
            #if preds[0][idx] > 0.9:

                odnos = int(image.shape[1] / resized.shape[1])
                x1 = x * odnos
                y1 = y * odnos
                wSw = window_size * odnos

                if "dog" in classes[idx]:
                    color = (0, 255, 255)
                    text = "DOG"
                elif "cat" in classes[idx]:
                    color = (0, 0, 255)
                    text = "CAT"
                else:
                    continue
                # Iscrtavanje kvadrata oko detektovanog objekta i dodavanje teksta
                cv2.putText(image, text, (x1 + 10, y1 + 25), cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)
                #cv2.putText(image, f"{preds[0][idx]}"[:4], (x1 + 10, y1 + wSw - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.55, color, 2)

                cv2.rectangle(image, (x1 + 2, y1 + 2), (x1 + wSw - 2, y1 + wSw - 2), color, 2)


# save the output image
cv2.imwrite("C:\Users\jovan\PycharmProjects\CV\Lab\LAB4 - Copy\output.png", image)

# display the output image
cv2.imshow("Image", image)
cv2.waitKey(0)
