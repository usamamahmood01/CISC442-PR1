from pydoc import describe
import cv2 as cv
from cv2 import SIFT_create
from matplotlib.pyplot import show
import numpy as np
#################################################################


# Reading the image
imageLena = cv.imread('images/lena.png')
image1 = cv.imread('images/Test_A1.png')
image2 = cv.imread('images/Test_A2.png')
image12 = cv.imread('images/Test_A12.png')
image3 = cv.imread('images/Test_B1.png')
image4 = cv.imread('images/Test_B2.png')
image5 = cv.imread('images/Test_C1.png')
image6 = cv.imread('images/Test_C2.png')
image7 = cv.imread('images/Test_D1.png')
image8 = cv.imread('images/Test_D2.png')
image9 = cv.imread('images/Test_E1.jpeg')
image10 = cv.imread('images/Test_E2.jpeg')
image11 = cv.imread('images/Test_F1.jpeg')
image12 = cv.imread('images/Test_F2.jpeg')
image13 = cv.imread('images/Test_G1.jpeg')
image14 = cv.imread('images/Test_G2.jpeg')
image15 = cv.imread('images/Test_H1.jpeg')
image16 = cv.imread('images/Test_H2.jpeg')
image17 = cv.imread('images/Test_I1.jpeg')
image18 = cv.imread('images/Test_I2.jpeg')

# resize the image for blending
image12= cv.resize(image12, image1.shape[1::-1])
image2 = cv.resize(image2, image1.shape[1::-1])
image4 = cv.resize(image4, image3.shape[1::-1])
image8 = cv.resize(image8, image7.shape[1::-1])
image10 = cv.resize(image10, image9.shape[1::-1])
image12 = cv.resize(image12, image11.shape[1::-1])
image14 = cv.resize(image14, image13.shape[1::-1])
image16 = cv.resize(image16, image15.shape[1::-1])
image18 = cv.resize(image18, image17.shape[1::-1])


# analyses the use of corner detectors to compute correspondences between images and the 
# possibility of the automatic creation of image mosaics. There will be 25 points for a completely automated 
# solution and 15 points for a semi-automatic solution. Apply the Harris corner detector (MATLAB/OpenCV 
# library) to compute point correspondences between two image feature (corner) points. Image matching can be 
# done using normalized cross-correlation (or any other method such as SSD).

# Algorithm:
# 1. Compute Harris corner detector response for each pixel in the image
# 2. Select the top N corner points
# 3. For each corner point, compute the normalized cross-correlation (NCC) with a small window around the 
#    point in the other image
# 4. Select the top M NCC matches
# 5. Display the matches
# 6. Compute the homography between the two images using the M matches
# 7. Warp the second image to the first image using the homography
# 8. Blend the two images using the homography
# 9. Display the result

# Application:


# function that returns the homography matrix between two images
def getHomographyMatrix(img1, img2):
    sift = cv.xfeatures2d.SIFT_create()
    kp1, des1 = sift.detectAndCompute(img1, None)
    kp2, des2 = sift.detectAndCompute(img2, None)
    FLANN_INDEX_KDTREE = 0
    index_params = dict(algorithm=FLANN_INDEX_KDTREE, trees=5)
    search_params = dict(checks=50)
    flann = cv.FlannBasedMatcher(index_params, search_params)
    matches = flann.knnMatch(des1, des2, k=2)
    good = []
    for m, n in matches:
        if m.distance < 0.7 * n.distance:
            good.append(m)

    src_pts = np.float32([kp1[m.queryIdx].pt for m in good]).reshape(-1, 1, 2)
    dst_pts = np.float32([kp2[m.trainIdx].pt for m in good]).reshape(-1, 1, 2)
    H, M = cv.findHomography(src_pts, dst_pts, cv.RANSAC, 5.0)

    return H


# wrap images using the homography, and display the result
def warpTwoImages(img1, img2, H):
    '''warp img2 to img1 with homograph H'''
    h1,w1 = img1.shape[:2]
    h2,w2 = img2.shape[:2]
    pts1 = np.float32([[0,0],[0,h1],[w1,h1],[w1,0]]).reshape(-1,1,2)
    pts2 = np.float32([[0,0],[0,h2],[w2,h2],[w2,0]]).reshape(-1,1,2)
    pts2_ = cv.perspectiveTransform(pts2, H)
    pts = np.concatenate((pts1, pts2_), axis=0)
    [xmin, ymin] = np.int32(pts.min(axis=0).ravel() - 0.5)
    [xmax, ymax] = np.int32(pts.max(axis=0).ravel() + 0.5)
    t = [-xmin,-ymin]
    Ht = np.array([[1,0,t[0]],[0,1,t[1]],[0,0,1]]) # translate

    result = cv.warpPerspective(img1, Ht.dot(H), (xmax-xmin, ymax-ymin))
    result[t[1]:h2+t[1],t[0]:w2+t[0]] = img2
    return result





M3 = getHomographyMatrix(image6, image5)
result = warpTwoImages(image6, image5, M3)
cv.imwrite('Results/Question_8/auto3.jpg', result)

M4 = getHomographyMatrix(image8, image7)
result = warpTwoImages(image8, image7, M4)
cv.imwrite('Results/Question_8/auto4.jpg', result)

M5 = getHomographyMatrix(image9, image10)
result = warpTwoImages(image9, image10, M5)
cv.imwrite('Results/Question_8/auto5.jpg', result)

M6 = getHomographyMatrix(image12, image11)
result = warpTwoImages(image12, image11, M6)
cv.imwrite('Results/Question_8/auto6.jpg', result)

M7 = getHomographyMatrix(image13, image14)
result = warpTwoImages(image13, image14, M7)
cv.imwrite('Results/Question_8/auto7.jpg', result)

M8 = getHomographyMatrix(image16, image15)
result = warpTwoImages(image16, image15, M8)
cv.imwrite('Results/Question_8/auto8.jpg', result)

M9 = getHomographyMatrix(image17, image18)
result = warpTwoImages(image17, image18, M9)
cv.imwrite('Results/Question_8/auto9.jpg', result)
