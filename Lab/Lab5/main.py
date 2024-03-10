import cv2
from cv2 import aruco
import yaml
import numpy as np
from pathlib import Path
from tqdm import tqdm

#Postavljanje parametara za Aruco marker
root = Path(__file__).parent.absolute()
calibrate_camera = True
calib_imgs_path = root.joinpath("Aruco")
aruco_dict = aruco.getPredefinedDictionary( aruco.DICT_6X6_1000 )
markerLength = 3.75
markerSeparation = 0.5
board = aruco.GridBoard_create(5, 7, markerLength, markerSeparation, aruco_dict)


arucoParams = aruco.DetectorParameters_create()

#Kalibracija kamere (opciono)
if calibrate_camera == True:
    # Učitavanje slika za kalibraciju
    img_list = []
    calib_fnms = calib_imgs_path.glob('*.jpg')
    for idx, fn in enumerate(calib_fnms):
        img = cv2.imread( str(root.joinpath(fn) ))
        img_list.append( img )
        h, w, c = img.shape
    print('Calibration images')

    # Detekcija markera na kalibracionim slikama
    counter, corners_list, id_list = [], [], []
    first = True
    for im in tqdm(img_list):
        img_gray = cv2.cvtColor(im,cv2.COLOR_RGB2GRAY)
        corners, ids, rejectedImgPoints = aruco.detectMarkers(img_gray, aruco_dict, parameters=arucoParams)
        if first == True:
            corners_list = corners
            id_list = ids
            first = False
        else:
            corners_list = np.vstack((corners_list, corners))
            id_list = np.vstack((id_list,ids))
        counter.append(len(ids))

    # Kalibracija kamere
    counter = np.array(counter)
    print ("Calibrating camera .... Please wait...")
    ret, mtx, dist, rvecs, tvecs = aruco.calibrateCameraAruco(corners_list, id_list, counter, board, img_gray.shape, None, None )

    # Čuvanje rezultata kalibracije u YAML datoteku
    data = {'camera_matrix': np.asarray(mtx).tolist(), 'dist_coeff': np.asarray(dist).tolist()}
    with open("calibration.yaml", "w") as f:
        yaml.dump(data, f)

    #Učitavanje kalibracija i otvaranje kamere za praćenje Aruco markera
    #camera = cv2.VideoCapture("Aruco\Aruco_board.mp4")
    camera = cv2.VideoCapture(0)

    ret, img = camera.read()

    with open('calibration.yaml') as f:
        #loadeddict = yaml.load(f)
        loadeddict = yaml.load(f, Loader=yaml.FullLoader)

    mtx = loadeddict.get('camera_matrix')
    dist = loadeddict.get('dist_coeff')
    mtx = np.array(mtx)
    dist = np.array(dist)

    ret, img = camera.read()
    img_gray = cv2.cvtColor(img,cv2.COLOR_RGB2GRAY)
    h,  w = img_gray.shape[:2]
    newcameramtx, roi=cv2.getOptimalNewCameraMatrix(mtx,dist,(w,h),1,(w,h))

    pose_r, pose_t = [], []

    #Estimacija poze Aruco markera na frejmovima
    while True:
        ret, img = camera.read()
        img_aruco = img
        im_gray = cv2.cvtColor(img,cv2.COLOR_RGB2GRAY)
        h,  w = im_gray.shape[:2]
        dst = cv2.undistort(im_gray, mtx, dist, None, newcameramtx)
        corners, ids, rejectedImgPoints = aruco.detectMarkers(dst, aruco_dict, parameters=arucoParams)
        if corners == None:
            print ("pass")
        else:
            rvec = np.empty((3,1))
            tvec = np.empty((3,1))
            ret= aruco.estimatePoseBoard(corners, ids, board, newcameramtx, dist, rvec, tvec)
            print ("Rotation ", rvec, "Translation", tvec)

            if ret != 0:
                img_aruco = aruco.drawDetectedMarkers(img, corners, ids, (0, 0, 255))

                if corners is not None and len(corners) > 0 and len(corners[0]) > 0:
                    corners_tuple = tuple(map(tuple, corners[0][0].astype(int)))

                    # Crta x, y i z ose
                    img_aruco = cv2.line(img_aruco, corners_tuple[0], corners_tuple[1], (0, 0, 255),
                                         2)  # Crvena linija za X-osu
                    img_aruco = cv2.line(img_aruco, corners_tuple[0], corners_tuple[2], (0, 255, 0),
                                         2)  # Zelena linija za Y-osu
                    img_aruco = cv2.line(img_aruco, corners_tuple[0], corners_tuple[3], (255, 0, 0),
                                         2)  # Plava linija za Z-osu
                else:
                    print("Nema detektovanih markera.")

                # # Pretvaranje koordinata uglova u tuple
                # corners_tuple = tuple(map(tuple, corners[0][0].astype(int)))
                #
                # # Crta x, y i z ose i markere na slici
                # img_aruco = cv2.line(img_aruco, corners_tuple[0], corners_tuple[1], (0, 0, 255),
                #                      2)  # Crvena linija za X-osu
                # img_aruco = cv2.line(img_aruco, corners_tuple[0], corners_tuple[2], (0, 255, 0),
                #                      2)  # Zelena linija za Y-osu
                # img_aruco = cv2.line(img_aruco, corners_tuple[0], corners_tuple[3], (255, 0, 0),
                #                      2)  # Plava linija za Z-osu

            if cv2.waitKey(0) & 0xFF == ord('q'):
                break
        # Prikazivanje slike sa detektovanim markerima
        cv2.imshow("World co-ordinate frame axes", img_aruco)
cv2.destroyAllWindows()




