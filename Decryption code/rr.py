import cv2
import numpy as np

img=cv2.imread("uscrambledimage.jpg")

averaging=cv2.blur(img,(5,5))
gaussian=cv2.GaussianBlur(img,(5,5), 0)
median=cv2.medianBlur(img, 5)

#cv2.imshow("uscrabledimage", img)
#cv2.imshow("average", averaging)
#cv2.imshow("gaussian", gaussian)
cv2.imshow("median", median)
cv2.imwrite("restoreimage.jpg", median)
#print(median)


cv2.waitKey(0)
cv2.destroyAllWindows()
