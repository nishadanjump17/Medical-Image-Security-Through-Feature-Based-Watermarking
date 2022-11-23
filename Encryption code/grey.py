import cv2

# open image file
image = cv2.imread('scrambledimage.jpg', cv2.IMREAD_UNCHANGED)

# convert image to grayscale
img1 = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)

# convert image to black and white
thresh, image_black = cv2.threshold(
    img1, 64, 64, cv2.THRESH_BINARY)

# display image in the window
#cv2.imshow('image.jpeg', image)                 # default view
#cv2.imshow('image.jpeg', image_grayscale)       # grayscale
cv2.imshow('scrambledimage.jpg', image_black)            # black and white image

# store output images
cv2.imwrite('image_grayscale.png', img1)
cv2.imwrite('image_black.png', image_black)

# break out on a key press
cv2.waitKey(0)

# clean up windows
cv2.destroyAllWindows()


