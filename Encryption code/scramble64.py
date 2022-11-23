import cv2
import numpy as np

image = cv2.imread("scrambledimage.jpg", 1)
# Loading the image

half = cv2.resize(image, (0, 0), fx = 0.1, fy = 0.1)
bigger = cv2.resize(image, (64, 64))

stretch_near = cv2.resize(image, (64, 64),
			interpolation = cv2.INTER_NEAREST)


Titles =["Original"]
images =[image]
count = 1
cv2.imshow('scrambledimage.jpg', image)
cv2.waitKey(0)

#for i in range(count):
	#plt.subplot(2, 2, i + 1)
	#plt.title(Titles[i])
	##plt.imshow(images[i])

#plt.show()
