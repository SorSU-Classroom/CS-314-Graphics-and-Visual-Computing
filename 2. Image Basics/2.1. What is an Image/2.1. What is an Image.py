'''
Image
- An image is a 2D representation of a scene or object.
- It is a collection of pixels.
- Each pixel is a small square of illumination.
- Each pixel has a specific location and a specific color.
- The color of a pixel is defined by the intensity of the light at that location.
- The intensity of light is defined by the amount of red, green, and blue light.

Types of Images
- Binary Image: An image with only two colors, black and white.
- Grayscale Image: An image with shades of gray.
- Color Image: An image with multiple colors.

Python Libraries for Image Processing
- OpenCV: Open Source Computer Vision Library.
    - It is used for image processing, video analysis, and machine learning.
    - It is written in C++ and has Python bindings.
    - In image processing, it is used for reading, writing, and processing images.
- PIL: Python Imaging Library, also known as Pillow.
    - It is used for opening, manipulating, and saving many different image file formats.
    - It is written in C and has Python bindings.
    - In image processing, it is used for basic image processing tasks.
- Scikit-Image: A collection of algorithms for image processing.
    - It is built on top of NumPy, SciPy, and Matplotlib.
    - In image processing, it is used for filtering, segmentation, and feature extraction.
- Matplotlib: A 2D plotting library for Python.
    - It is used for creating static, animated, and interactive visualizations.
    - In image processing, it is used for displaying images and plots.
- NumPy: A library for numerical computing in Python.
    - It is used for creating and manipulating arrays.
    - In image processing, it is used for representing images as arrays.
- SciPy: A library for scientific computing in Python.
    - It is used for scientific and technical computing.
    - In image processing, it is used for image processing algorithms.
'''

'''
Python Implementation of OpenCV
'''

# Importing the OpenCV library
import cv2

# Reading an image from the file using OpenCV
imageOpenCV = cv2.imread('../../assets/images/parrot.jpg')

# Create a window and display the image
cv2.namedWindow('OpenCV Image', cv2.WINDOW_NORMAL)

# Resizing the window to 1/4th of the image size
cv2.resizeWindow('OpenCV Image', imageOpenCV.shape[1] // 4, imageOpenCV.shape[0] // 4)

# Displaying the image
cv2.imshow('OpenCV Image', imageOpenCV)

'''
Python Implementation of OpenCV with Matplotlib
'''

# Importing the Matplotlib libraries
import matplotlib.pyplot as plt

# Reading an image from the file using OpenCV
imageOpenCVwithMatplotlib = cv2.imread('../../assets/images/parrot.jpg')

# # Convert color from BGR to RGB
# imageOpenCVwithMatplotlib = cv2.cvtColor(imageOpenCVwithMatplotlib, cv2.COLOR_BGR2RGB)

# Displaying the image using Matplotlib
print("Displaying the image using OpenCV with Matplotlib")
plt.figure('OpenCV with Matplotlib Image')
plt.imshow(imageOpenCVwithMatplotlib)
plt.axis('off')
plt.show()

'''
Python Implementation of PIL
'''

# Importing the PIL
import PIL.Image as Image

# Reading an image from the file using PIL
imagePIL = Image.open('../../assets/images/parrot.jpg')

# Displaying the image using PIL
# print("Displaying the image using PIL")
# imagePIL.show()

# Displaying the image using Matplotlib
print("Displaying the PIL Image using Matplotlib")
plt.figure('Matplotlib Image')
plt.imshow(imagePIL)
plt.axis('off')
plt.show()

'''
Python Implementation of Scikit-Image
'''

# Importing the Scikit-Image library
from skimage import io

# Reading an image from the file using Scikit-Image
imageScikit = io.imread('../../assets/images/parrot.jpg')

# Displaying the image using Scikit-Image
print("Displaying the image using Scikit-Image")
io.imshow(imageScikit)
io.show()

'''
Python Implementation of NumPy
'''

# Importing the NumPy library
import numpy as np

# Reading an image from the file using OpenCV
imageOpenCV = cv2.imread('../../assets/images/parrot.jpg')

# Converting the image to a NumPy array
imageNumPy = np.array(imageOpenCV)

# Convert color from BGR to RGB
imageNumPy = cv2.cvtColor(imageNumPy, cv2.COLOR_BGR2RGB)

# Displaying the image using NumPy
print("Displaying the image using NumPy")
plt.figure('NumPy Image')
plt.imshow(imageNumPy)
plt.axis('off')
plt.show()

'''
Python Implementation of SciPy
'''

# Importing the SciPy library
from scipy import ndimage

# Reading an image from the file using OpenCV
imageOpenCV = cv2.imread('../../assets/images/parrot.jpg')

# Converting color from BGR to RGB
imageOpenCV = cv2.cvtColor(imageOpenCV, cv2.COLOR_BGR2RGB)

# Rotating the image by 45 degrees
imageSciPy = ndimage.rotate(imageOpenCV, 45)

# Displaying the image using SciPy
print("Displaying the image using SciPy")
plt.figure('SciPy Image')
plt.imshow(imageSciPy)
plt.axis('off')
plt.show()

# Waiting for a key press and then closing the window
cv2.waitKey(0)
cv2.destroyAllWindows()