import numpy as np
from PIL import Image

# Read Image
img = Image.open('scrambledimage.jpg')
# Convert Image to Numpy as array
img = np.array(img)
# Put threshold to make it binary
binarr = np.where(img > 128, 255, 0)
# Covert numpy array back to image
binimg = Image.fromarray(binarr)

