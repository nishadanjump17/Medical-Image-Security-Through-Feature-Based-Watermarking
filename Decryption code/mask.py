import cv2
import numpy as np

# load the image
img = cv2.imread('uscrambledimage.jpg')
img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
#cv2.imshow("maskimage", img)

gray = cv2.medianBlur(cv2.cvtColor(img, cv2.COLOR_RGB2GRAY), 5)
cv2.imshow("grayimage", gray)
mask = np.full((img.shape[0], img.shape[1]), 0, dtype=np.uint8)

cv2.imshow("mask", mask)
cv2.imwrite("grayimage.jpg", gray)
cv2.imwrite("maskimage.jpg", mask)
cv2.waitKey(0)
cv2.destroyAllWindows()
