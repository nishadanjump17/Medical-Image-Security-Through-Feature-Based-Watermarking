import cv2



#reading images
img=cv2.imread('images/mark.jpg', 0)
print(img)

#display images
cv2.imshow('images/mark.jpg', img)
cv2.waitKey(0)
cv2.destroyAllWindows()

