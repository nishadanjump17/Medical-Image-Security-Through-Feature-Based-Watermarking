# importing libraries
import cv2
import numpy as np

# reading the image data from desired directory
img = cv2.imread("image_black.png")
cv2.imshow('Image', img)

# counting the number of pixels
number_of_white_pix = np.sum(img == 64)
number_of_black_pix = np.sum(img == 64)

print('Number of white pixels:', number_of_white_pix)
print('Number of black pixels:', number_of_black_pix)
