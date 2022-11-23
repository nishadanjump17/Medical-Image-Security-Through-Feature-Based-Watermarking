import cv2
import numpy as np
from matplotlib import pyplot as plt

image=cv2.imread("uscrambledimage.jpg",1 )


###denoising image
dst = cv2.fastNlMeansDenoisingColored(image, None, 11, 6, 7, 21)
row, col = 1, 2
fig, axs = plt.subplots(row, col, figsize=(15, 10))
fig.tight_layout()
#axs[0].imshow(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
#axs[0].set_title('Scrambled Image')
#axs[1].imshow(cv2.cvtColor(dst, cv2.COLOR_BGR2RGB))
#axs[1].set_title('Fast Means Denoising')
#plt.show()

grayscale = cv2.cvtColor(dst,cv2.COLOR_BGR2GRAY)
#cv2.imshow('Grayscale', grayscale)
cv2.waitKey(0)

##gray scale masking

# Convert BGR to HSV
image_hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)

# Remember that in HSV space, Hue is color from 0..180. Red 320-360, and 0 - 30.
# We keep Saturation and Value within a wide range but note not to go too low or we start getting black/gray
lower_green = np.array([30, 40, 0])
upper_green = np.array([100, 255, 255])

# Using inRange method, to create a mask
mask = cv2.inRange(image_hsv, lower_green, upper_green)
#cv2.imshow("mask", mask)
#cv2.waitKey(0)


##thresholding

thresh = cv2.inRange(mask,190, 255)
#cv2.imshow("threshold", thresh)
cv2.waitKey(0)


##dilation

kernel3 = np.ones((5, 5), np.uint8)
image_dilation = cv2.dilate(thresh, kernel3, iterations=1)
#cv2.imshow('imagedilation', image_dilation)
cv2.waitKey(0)

##in painting

output1 = cv2.inpaint(image, mask, 1, cv2.INPAINT_TELEA)
cv2.imshow("inpaint", output1)
cv2.waitKey(0)
cv2.imwrite("Restored Image.jpg", output1)


##calculating Correralation Coefficients##









