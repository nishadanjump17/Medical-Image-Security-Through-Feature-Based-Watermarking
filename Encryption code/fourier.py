# fourier_synthesis.py

import numpy as np
import matplotlib.pyplot as plt

image_filename = "images/brain.jpg"

#FOURIER TRANSFORM
def calculate_2dft(input):
    ft = np.fft.ifftshift(input)
    ft = np.fft.fft2(ft)
    return np.fft.fftshift(ft)


# Read and process image
image = plt.imread('images/brain.jpg')
image = image[:, :, :3].mean(axis=2)  # Convert to grayscale

plt.set_cmap("gray")

# FOURIER COEFFICIENTS
ft = calculate_2dft(image)
#print(ft)
#print(type(ft))

# creating a NumPy array
#ft = np.array(0)

arr=np.array(0)
hashcode=list()
# traversing the list
for i in range(len(ft)):
   arr= np.append(arr,ft[i].real)

#arr = np.arange(1, 5)

# AVERAGE THE REAL PARTS
avg = np.average(arr)
print(avg)
for i in range(len(ft)):
   if (arr[i]>avg):
       hashcode.append('1')
   else:
      hashcode.append('0')

print(hashcode)




# Function to convert list to string
#def listToString(s):

	# initialize an empty string
	#str1 = ""

	# traverse in the string
	#for ele in s:
		#str1 += ele

	# return string
	#return str1


# Driver code
#s =['1','1','0']
#print(listToString(s))

    
    
    


