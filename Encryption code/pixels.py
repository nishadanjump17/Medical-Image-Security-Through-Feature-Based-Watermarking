import cv2
import numpy as np
import matplotlib as plt

img = cv2.imread("image_black.png",0)
img_rgb = cv2.imread("image_black.png")

h = img.shape[0]
w = img.shape[1]

img_thres= np.zeros((h,w))
n_pix = 0
# loop over the image, pixel by pixel
for y in range(0, h):
    for x in range(0, w):
        # threshold the pixel
        pixel = img[y, x]
        if pixel < 64: # because pixel value will be between 0-255.
            n_pix = 0
        else:
            n_pix = 1

        img_thres[y, x] = n_pix 
        #print(pixel)

cv2.imshow("image_black.png", img_thres)
cv2.waitKey(1)
