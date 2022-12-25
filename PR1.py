from pydoc import describe
import cv2 as cv
from matplotlib.pyplot import show
import numpy as np
#################################################################


# Reading the image
imageLena = cv.imread('images/lena.png')
image1 = cv.imread('images/Test_A1.png')
image2 = cv.imread('images/Test_A2.png')
image3 = cv.imread('images/Test_B1.png')
image4 = cv.imread('images/Test_B2.png')
image5 = cv.imread('images/Test_C1.png')
image6 = cv.imread('images/Test_C2.png')
image7 = cv.imread('images/Test_D1.png')
image8 = cv.imread('images/Test_D2.png')
image12 = cv.imread('images/Test_A12.png')


# resize the image for blending
image12= cv.resize(image12, image1.shape[1::-1])
#image2 = cv.resize(image2, image1.shape[1::-1])
image4 = cv.resize(image4, image3.shape[1::-1])
image8 = cv.resize(image8, image7.shape[1::-1])


#Helper function: covert the image to grayscale
def grayscale(img):
    return cv.cvtColor(img, cv.COLOR_BGR2GRAY)

# Helper function: function that adds the padding to the image
def padding(img, padding):
    # create an empty image with the padding
    padded_img = np.zeros((img.shape[0] + 2 * padding, img.shape[1] + 2 * padding,3), np.uint8)
    # add the original image to the padded image
    if(img[0,0].shape == (3,)):
        padded_img[padding:padded_img.shape[0] - padding, padding:padded_img.shape[1] - padding,0] = img[:,:,0]
        padded_img[padding:padded_img.shape[0] - padding, padding:padded_img.shape[1] - padding,1] = img[:,:,1]
        padded_img[padding:padded_img.shape[0] - padding, padding:padded_img.shape[1] - padding,2] = img[:,:,2]

    else:
        padded_img = np.zeros((img.shape[0] + 2 * padding, img.shape[1] + 2 * padding), np.uint8)
        padded_img[padding:padded_img.shape[0] - padding, padding:padded_img.shape[1] - padding] = img

    return padded_img   


# Helper function: one side padding
def sidePadding(img, dim, sides):
    new_img = np.array(img)
    new_h, new_w = dim
    h, w = img.shape[:2]
    zero_row = np.array([[0]*w], dtype=np.uint8)
    zero_column = np.zeros((h,1), dtype=np.uint8)
    if len(cv.split(img)) == 3:
        #print(len(cv.split(img)))
        zero_column = np.array([[[0]*3]]*(h), dtype=np.uint8)
        zero_row = np.array([[[0]*3]*w], dtype=np.uint8)
    if 'bottom' in sides:
        while(new_h>new_img.shape[:2][0]):
            new_img = np.concatenate((new_img, zero_row), axis=0)
    if 'top' in sides:
         while(new_h>new_img.shape[:2][0]):
            new_img = np.concatenate((zero_row, new_img), axis=0)
    if 'right' in sides:
        while (new_w>new_img.shape[:2][1]):
            new_img = np.concatenate((new_img, zero_column), axis=1)
    if 'left' in sides:
        while (new_w>new_img.shape[:2][1]):
            new_img = np.concatenate((zero_column, new_img), axis=1)
    
    return new_img


# Write a function Convolve (I, H). I is an image of varying size, H is a kernel of varying size.
# The output of the function should be the convolution result that is displayed.
# The function should be able to handle both grayscale and color images.
def convolve(I, H):
    I = padding(I, 2)
    # Get the image size
    imgSize = I.shape
    # Get the kernel size
    kernelSize = H.shape
    # Get the kernel center
    kernelCenter = (int(kernelSize[0] / 2), int(kernelSize[1] / 2))
    # Create a new image with the same size as the original image
    newImage = np.zeros(imgSize)
    # Loop through the image
    for y in range(imgSize[0]):
        for x in range(imgSize[1]):
            # Initialize the sum
            sum = 0
            # Loop through the kernel
            for i in range(kernelSize[0]):
                for j in range(kernelSize[1]):
                    # Get the image index
                    imgIndexY = y + (i - kernelCenter[0])
                    imgIndexX = x + (j - kernelCenter[1])
                    # Make sure the index is within the bounds of the image
                    if imgIndexY >= 0 and imgIndexY < imgSize[0] and imgIndexX >= 0 and imgIndexX < imgSize[1]:
                        # Sum the image pixel value times the kernel value
                        sum += I[imgIndexY, imgIndexX] * H[i, j]
            # Set the new image pixel value to the sum
            newImage[y, x] = sum
    # Return the new image
    return newImage

#converting the image to grayscale
grayImage = grayscale(imageLena)
 # gaussian kernel 
H = np.array([[1, 2, 1], [2, 4, 2], [1, 2, 1]]) * 1/16

#sobel kernel
# x-direction
H2 = np.array([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]])
# y-direction
H3 = np.array([[-1, -2, -1], [0, 0, 0], [1, 2, 1]])

#output of RGB image
output = convolve(imageLena, H)
cv.imwrite('Results/Question_1/RGBoutputOf1.jpg', output)

outputGRAY = convolve(grayImage, H2)
#outputGRAY2 = convolve(outputGRAY, H3)
cv.imwrite('Results/Question_1/GRAYoutputOf1.jpg', outputGRAY)

# Applying the filter2D() function on the image to check the result.
checker = cv.filter2D(imageLena, -1, H)
Graychecker = cv.filter2D(grayImage, -1, H2)
# Showing the original and output image
cv.imwrite('Results/Question_1/RGBchecker.jpg', checker)
cv.imwrite('Results/Question_1/Graychecker.jpg', Graychecker)

#################################################################

# Write a function Reduce(I) that takes image I as input and outputs a copy of the image resampled
# by half the width and height of the input. Remember to Gaussian filter the image before reducing it; 
# use separable 1D Gaussian kernels.

def reduce(I):
    # apply 1D Gaussian filter
    I = cv.GaussianBlur(I, (1, 1), 0)
    # reduce the image by half
    reducedImage = I[::2, ::2]
    
    return reducedImage

reduced = reduce(imageLena)
cv.imwrite('Results/Question_2/reduced.jpg', reduced)

#################################################################

# Write a function Expand(I) that takes image I as input and outputs a copy of the image expanded, 
# twice the width and height of the input.

def expand(I):
    # Get the image size
    imgSize = I.shape
    # Create a new image with the same size as the original image
    newImage = np.zeros((imgSize[0] * 2, imgSize[1] * 2, 3))
    # Loop through the image
    for y in range(imgSize[0]):
        for x in range(imgSize[1]):
            # Set the new image pixel value to the sum
            newImage[y*2, x*2] = I[y, x]
            newImage[y*2+1, x*2] = I[y, x]
            newImage[y*2, x*2+1] = I[y, x]
            newImage[y*2+1, x*2+1] = I[y, x]
    return newImage

expanded = expand(imageLena)
cv.imwrite('Results/Question_3/expanded.jpg', expanded)

#################################################################

# Use the Reduce() function to write the GaussianPyramid(I,n) function, where n is the no. of levels.
# Note that a higher level (lower resolution) in a Gaussian Pyramid is formed by removing consecutive rows and 
#columns in a lower level (higher resolution) image. 

def gaussianPyramid(I, n):
    #add the gaussian blur
    I = cv.GaussianBlur(I, (1, 1), 0)
    # Create a list to store the pyramid
    pyramid = []
    # Add the original image to the list
    pyramid.append(I)
    # Loop through the levels exacatly n times
    for i in range(0, n):
        # Reduce the previous level
        reduced = reduce(pyramid[i])
        # Add the reduced level to the list
        pyramid.append(reduced)
    # Return the pyramid
    return pyramid

pyramid = gaussianPyramid(imageLena, 4)
# Show the images in the pyramid
for i in range(len(pyramid)):
    cv.imwrite('Results/Question_4/GausianPyramid' + str(i+1) + '.jpg', pyramid[i])

#################################################################

#Use the above functions to write LaplacianPyramids(Iinp, n) that produces n level Laplacian pyramid of I. 
#Note that a level in Laplacian Pyramid is formed by the difference between that level in the Gaussian Pyramid 
#and expanded version of its upper level in the Gaussian Pyramid. 

def laplacianPyramid(I, n):
    # Create a list to store the pyramid
    pyramid = []
    # Get the Gaussian pyramid
    gaussian = gaussianPyramid(I, n)
    # Loop through the levels
    for i in range(n):
        # Expand the next level
        expanded = expand(gaussian[i+1])
        # handles the dimensions of expanded img
        dim = np.shape(gaussian[i])
        if dim != np.shape(expanded):
            expanded = expanded[0:dim[0], 0:dim[1]]
        # Add the difference between the Gaussian level and its expanded version
        pyramid.append(gaussian[i] - expanded)
    pyramid[n-1] = gaussian[n-1]
    # Return the pyramid
    return pyramid

LapPyramid = laplacianPyramid(imageLena, 4)
# Show the images in the pyramid
for i in range(len(LapPyramid)):
    cv.imwrite('Results/Question_5/LaplacianPyramid' + str(i+1) + '.jpg', LapPyramid[i])


#################################################################

# Write the Reconstruct(LI,n) function which collapses the Laplacian pyramid LI of n levels 
# to generate the original image. Report the error in reconstruction using image difference.

def reconstruct(LI, n):
    # Create a list to store the pyramid
    pyramid = []
    # Get the Laplacian pyramid
    #laplacian = laplacianPyramid(LI, n)
    # Add the last level of the Laplacian pyramid to the list
    pyramid.append(LI[n-1])
    # Loop through the levels
    for i in range(n-1, 0, -1):
        # Expand the previous level
        expanded = expand(pyramid[n-i-1])
        # handles the dimensions of expanded img
        dim = np.shape(LI[i-1])
        if dim != np.shape(expanded):
            expanded = expanded[0:dim[0], 0:dim[1]]
        # Add the sum of the Laplacian level and its expanded version
        pyramid.append(expanded + LI[i-1])
    # Return the last level of the pyramid
    return pyramid[n-1]

pyramid = reconstruct(LapPyramid, 4)
cv.imwrite('Results/Question_6/reconstructed.jpg', pyramid)
# Error in reconstruction
Error = imageLena - pyramid
cv.imwrite('Results/Question_6/Error.jpg', Error)

#################################################################

# Finally, you will be mosaicking images using Laplacian plane based reconstruction (Note that your program 
# should handle color images). Let the user pick the blend boundaries of all images by mouse. Blend 
# the left image with the right image. Note the left and right images share a joint region. Submit four results of 
# mosaicking. Each mosaic can be comprised of 2 individual images from different cameras/viewpoints.


# Helper function: get the mask of I1 and I2 using the mouse
def getMask2(I1, I2):
    # Create a list to store the masks
    masks = []
    # Create a list to store the images
    images = [I1, I2]
    # Loop through the images
    for i in range(len(images)):
        # Create a copy of the image
        image = images[i].copy()
        # Create a window
        cv.namedWindow('image')
        # Create a list to store the points
        points = []
        # Create a function to handle the mouse events
        def mouse_event(event, x, y, flags, param):
            # If the left button is clicked
            if event == cv.EVENT_LBUTTONDOWN:
                # Add the point to the list
                points.append((x, y))
                # Draw a circle on the image
                cv.circle(image, (x, y), 3, (255, 0, 0), -1)
                # Show the image
                cv.imshow('image', image)
        # Set the mouse callback
        cv.setMouseCallback('image', mouse_event)
        # Show the image
        cv.imshow('image', image)
        # Wait for a key
        cv.waitKey(0)
        # Destroy the window
        cv.destroyWindow('image')
        # Create a mask
        mask = np.zeros(images[i].shape, np.uint8)
        # For I1 mask the right side of the image from the point to the right
        if i == 0:
            mask[:, :points[0][0], :] = 255
            # multiply the mask with the image
            masked = cv.bitwise_and(images[i], mask)
        # For I2 mask the left side of the image from the point to the left
        else:
            mask[:, points[0][0]:, :] = 255
            masked = cv.bitwise_and(images[i], mask)
        # Add the mask to the list
        masks.append(masked)
    # Return the masks
    return masks

# Helper function: mask the image using the mouse
def getMask1(I):
    # Create a copy of the image
    image = I.copy()
    # Create a window
    cv.namedWindow('image')
    # Create a list to store the points
    points = []
    # Create a function to handle the mouse events
    def mouse_event(event, x, y, flags, param):
        # If the left button is clicked
        if event == cv.EVENT_LBUTTONDOWN:
            # Add the point to the list
            points.append((x, y))
            # Draw a circle on the image
            cv.circle(image, (x, y), 3, (255, 0, 0), -1)
            # Show the image
            cv.imshow('image', image)
    # Set the mouse callback
    cv.setMouseCallback('image', mouse_event)
    # Show the image
    cv.imshow('image', image)
    # Wait for a key
    cv.waitKey(0)
    # Destroy the window
    cv.destroyWindow('image')
    # Create a mask
    mask = np.zeros(I.shape, dtype=np.uint8)
    # fill the right side of the image from the point to the right with white and rest with 1
    mask[:, :points[0][0], :] = 255
    # return the mask
    return mask


# Helper function: a function that returns the x and y coordinates of the mouse in the getMask function
def getMouseCoordinates(I,I2):
    # Create a copy of the image
    image = I.copy()
    image2 = I2.copy()
    # Create a window
    cv.namedWindow('image')
    # Create a list to store the points
    points = []
    # Create a function to handle the mouse events
    def mouse_event(event, x, y, flags, param):
        # If the left button is clicked
        if event == cv.EVENT_LBUTTONDOWN:
            # Add the point to the list
            points.append((x, y))
            # Draw a circle on the image
            cv.circle(image, (x, y), 3, (255, 0, 0), -1)
            # Show the image
            cv.imshow('image', image)
    # Set the mouse callback
    cv.setMouseCallback('image', mouse_event)
    # Show the image
    cv.imshow('image', image)
    cv.imshow('image2', image2)
    # Wait for a key
    cv.waitKey(0)
    # Destroy the window
    cv.destroyWindow('image')
    # return the points
    return points


# Helper function: takes cordiantes of the mouse and returns the mask
def getMask(cor, shape):
    # Create a mask
    x = cor[0][0]
    mask = np.zeros(shape, dtype=np.uint8)
    # fill the right side of the image black and rest with white
    mask[:, :x+1, :] = 1
    mask = np.uint8(mask)
    # return the mask
    return mask


# Helper function: returns the dimensions of the image
def get_desired_dimensions(shape0, shape1, boundary):
    h1, w1 = shape0[:2]
    h2, w2 = shape1[:2]
    # boundry is the x point where the two images meet
    # the desired width is the width of the two images minus the overlap

    x = boundary[0][0]
    desired_w = None
    desired_h = None
    desired_w = w1 - (w1 - x) + w2
    if h1 == h2:
        return (h1, desired_w)
    else:
        if h1<h2:
            desired_h = h1 + abs(h1-h2)
        else:
            desired_h = h2 + abs(h1-h2) 
            return (desired_h,desired_w)


# blend the two images, I1 and I2, using the mask and other helper functions.
def blend(I1, I2):
    n = 2
    boundary = getMouseCoordinates(I1, I2)
    dim = get_desired_dimensions(I1.shape, I2.shape, boundary)
    I1 = sidePadding(I1, dim, 'right')
    I2 = sidePadding(I2, dim, 'left')

    mask = getMask(boundary, I1.shape)
    mask = sidePadding(mask, dim, 'right')

    # checker if the size of image is not the same, evens it by adding padding
    if I1.shape[:2][0] < I2.shape[:2][0]:
        I1 = sidePadding(I1, dim, 'bottom')
    else:
        I2 = sidePadding(I2, dim, 'bottom')

    if I1.shape[:2][0] > I2.shape[:2][0]:
        I1 = sidePadding(I1, dim, 'top')
    else:
        I2 = sidePadding(I2, dim, 'top')


    I1 = np.uint8(I1)
    I2 = np.uint8(I2)
    mask = np.uint8(mask)

    # pyramids
    mask_pyr = gaussianPyramid(mask, n-1)
    I1_pyr = laplacianPyramid(I1, n)
    I2_pyr = laplacianPyramid(I2, n)

    merge_pyr = [None] * n

    for i in range(0, n):
        merge_pyr[i] = mask_pyr[i] * I1_pyr[i] + (1.0 - mask_pyr[i]) * I2_pyr[i]
        merge_pyr[i] = np.uint8(merge_pyr[i])
        #cv.imshow('merged_py'+str(i), merge_pyr[i])
        #cv.waitKey(0)
    
    mosaic = reconstruct(merge_pyr, n)
    #cv.imshow('mosaic', mosaic)
    #cv.waitKey(0)
    mosaic = np.uint8(mosaic)

    return mosaic

#blend1 = blend(image12, image1)
#cv.imwrite('Results/Question_7/TestAmosaic.jpg', blend1)

#blend2 = blend(image3, image4)
#cv.imwrite('Results/Question_7/TestBmosaic.jpg', blend2)

blend3 = blend(image5, image6)
cv.imwrite('Results/Question_7/TestCmosaic.jpg', blend3)

#blend4 = blend(image7, image8)
#cv.imwrite('Results/Question_7/TestDmosaic.jpg', blend4)


############################################################################################################
