import cv2
import numpy as np
import os


def create_panorama(images, min_match_count=30):
    # Initialize SIFT detector
    sift = cv2.SIFT_create()

    # Placeholder for the final panorama
    panorama = None

    # Iterate over all image pairs
    for i in range(len(images) - 1):
        # Convert images to grayscale
        gray1 = cv2.cvtColor(images[i], cv2.COLOR_BGR2GRAY)
        gray2 = cv2.cvtColor(images[i + 1], cv2.COLOR_BGR2GRAY)

        # Detect keypoints and compute descriptors
        kp1, des1 = sift.detectAndCompute(gray1, None)
        kp2, des2 = sift.detectAndCompute(gray2, None)

        # FLANN matcher - pair key points
        FLANN_INDEX_KDTREE = 1
        index_params = dict(algorithm=FLANN_INDEX_KDTREE, trees=5)
        search_params = dict(checks=50)
        flann = cv2.FlannBasedMatcher(index_params, search_params)

        matches = flann.knnMatch(des1, des2, k=2)

        # Lowe's ratio test to find good matches
        good_matches = []
        for m, n in matches:
            if m.distance < 0.7 * n.distance:
                good_matches.append(m)

        if len(good_matches) > min_match_count:
            src_pts = np.float32([kp1[m.queryIdx].pt for m in good_matches]).reshape(-1, 1, 2)
            dst_pts = np.float32([kp2[m.trainIdx].pt for m in good_matches]).reshape(-1, 1, 2)

            # Calculate Homography - in order to join images
            M, mask = cv2.findHomography(src_pts, dst_pts, cv2.RANSAC, 5.0)

            # Warp images to create panorama - join images
            if panorama is None:
                panorama = images[i]
            h, w = images[i + 1].shape[:2]
            w = cv2.warpPerspective(panorama, M, (panorama.shape[1] + w, h))
            w[0:h, 0:images[i + 1].shape[1]] = images[i + 1]

            panorama = w
        else:
            print("Not enough matches are found for images {} and {} - {}/{}".format(i, i + 1, len(good_matches),
                                                                                     min_match_count))
            return None

    return panorama


# Load images
image_directory = os.path.join(os.getcwd(), "slike")
image_paths = sorted(
    [os.path.join(image_directory, path) for path in os.listdir(image_directory) if path.endswith('.JPG')])
images = [cv2.imread(path) for path in image_paths]

# Create panorama
panorama = create_panorama(images)

# Save the panorama
if panorama is not None:
    cv2.imwrite('panorama.jpg', panorama)
    print("Panorama saved.")
    cv2.imshow('Panorama', panorama)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
else:
    print("Panorama creation failed.")
