'''
Color Models

A color model is a mathematical model describing the way colors can be represented as
tuples of numbers. It is used to describe the way colors are represented in an image.
There are several color models used in image processing, such as RGB, HSV, YUV and
CIE L*a*b.

These color models are used to represent colors in different ways. For example, the RGB
color model is used to represent colors in terms of red, green, and blue components.
Similarly, the HSV color model is used to represent colors in terms of hue, saturation,
and value components. The YUV color model is used to represent colors in terms of
luminance and chrominance components. The CIE L*a*b color model is used to represent
colors in terms of lightness, green-red, and blue-yellow components.

In image processing, different color models are used for different purposes. For
example, the RGB color model is used for displaying images on a computer screen, while
the YUV color model is used for video compression. The HSV color model is used for
color segmentation, and the CIE L*a*b color model is used for color correction.

In this section, we will learn about the RGB color model, which is the most commonly
used color model in image processing. We will also learn how to convert images from one
color model to another using OpenCV and Matplotlib.

@see: https://en.wikipedia.org/wiki/Color_model
'''

# Initial Setup
# Importing the OpenCV and Matplotlib libraries
import cv2
import numpy as np
import matplotlib.pyplot as plt

image_path = '../../assets/images/parrot.jpg'
# image_path = '../../assets/images/rgb_volcano_1.jpeg'

# Reading an image from the file using OpenCV
original_image = cv2.imread(image_path)

'''
Converting an Image to Grayscale

We convert an image to grayscale by taking the average of the three color channels. The
average of the three color channels is calculated using the formula:

    Gray = (R + G + B) / 3

where R, G, and B are the red, green, and blue color channels of the image, respectively.

@see: https://en.wikipedia.org/wiki/Grayscale
'''
# Read Image in Grayscale Mode
grayscale_image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)

'''
Convert from BGR to RGB

OpenCV reads images in the BGR (Blue-Green-Red) color space by default. However,
Matplotlib expects images to be in the RGB (Red-Green-Blue) color space. Therefore, we
need to convert the image from the BGR color space to the RGB color space before
displaying it using Matplotlib.
'''
# Convert color from BGR to RGB
original_image = cv2.cvtColor(original_image, cv2.COLOR_BGR2RGB)
grayscale_image = cv2.cvtColor(grayscale_image, cv2.COLOR_BGR2RGB)

# Displaying the image using Matplotlib
print("Displaying the image using OpenCV with Matplotlib")
plt.figure('OpenCV with Matplotlib Image')

plt.subplot(1, 2, 1)
plt.imshow(original_image)
plt.axis('off')
plt.title('Original Image')

plt.subplot(1, 2, 2)
plt.imshow(grayscale_image, cmap='gray')
plt.axis('off')
plt.title('Grayscale Image')

plt.show()

'''
RGB Color Model

The RGB color model is an additive color model in which red, green, and blue light are
added together in various ways to reproduce a broad array of colors. The name of the
model comes from the initials of the three additive primary colors, red, green, and blue.

In the RGB color model, colors are represented as tuples of three numbers, where each
number represents the intensity of one of the three primary colors. The intensity of
each color is defined by the amount of red, green, and blue light present in the color.

The RGB color model is used to represent colors in digital images, computer graphics,
and television screens. It is the most commonly used color model in image processing
and computer vision.

Example of RGB Color Model:
- Red: (255, 0, 0)
- Green: (0, 255, 0)
- Blue: (0, 0, 255)
- White: (255, 255, 255)
- Black: (0, 0, 0)
- Yellow: (255, 255, 0)
- Cyan: (0, 255, 255)
- Magenta: (255, 0, 255)
- Gray: (128, 128, 128)

@see: https://en.wikipedia.org/wiki/RGB_color_model
'''

# Splitting the RGB image into its three color channels
channel_b, channel_g, channel_r = cv2.split(original_image)

# Displaying the three color channels
print("Displaying the RGB Color Model and its channels")
plt.figure('RGB Color Model')

# Display the Red Channel
plt.subplot(3, 3, 1)
plt.imshow(channel_r)
plt.title('Red Channel')
plt.axis('off')

# Display the Green Channel
plt.subplot(3, 3, 2)
plt.imshow(channel_g)
plt.title('Green Channel')
plt.axis('off')

# Display the Blue Channel
plt.subplot(3, 3, 3)
plt.imshow(channel_b)
plt.title('Blue Channel')
plt.axis('off')

# Display the Red Channel in Grayscale
plt.subplot(3, 3, 4)
plt.imshow(channel_r, cmap='gray')
plt.title('Red Channel (BW)')
plt.axis('off')

# Display the Green Channel in Grayscale
plt.subplot(3, 3, 5)
plt.imshow(channel_g, cmap='gray')
plt.title('Green Channel (BW)')
plt.axis('off')

# Display the Blue Channel in Grayscale
plt.subplot(3, 3, 6)
plt.imshow(channel_b, cmap='gray')
plt.title('Blue Channel (BW)')
plt.axis('off')

plt.show()

'''
HSV Color Model

The HSV color model is a cylindrical color model that describes colors in terms of hue,
saturation, and value. The hue component represents the color type, the saturation
component represents the color intensity, and the value component represents the
brightness of the color.

The HSV color model is used to represent colors in a way that is more intuitive to
humans than the RGB color model. It is often used in image processing algorithms that
require
color segmentation, color correction, and color enhancement.

In the HSV color model, colors are represented as tuples of three numbers, where each
number represents the hue, saturation, and value of the color. The hue component is
represented as an angle between 0 and 360 degrees, the saturation component is
represented as a percentage between 0 and 100%, and the value component is represented
as a percentage between 0 and 100%.

Example of HSV Color Model
- Red: (0, 100%, 100%)
- Green: (120, 100%, 100%)
- Blue: (240, 100%, 100%)
- White: (0, 0%, 100%)
- Black: (0, 0%, 0%)

In OpenCV, the maximum value for the hue component is 180, the maximum value for the
saturation and value components is 255. This is because OpenCV uses the 8-bit unsigned
integer data type to represent the color components which have a range of 0 to 255. So
the hue component is scaled down to the range of 0 to 180 degrees in OpenCV or by
dividing the original 0-360 degrees hue value by 2.

@see: https://en.wikipedia.org/wiki/HSL_and_HSV
'''

'''
Converting an RGB Image to HSV Image

We convert an RGB image to an HSV image using the cv2.cvtColor() function in OpenCV.
The cv2.cvtColor() function takes two arguments: the input image and the color space
conversion code. The color space conversion code for converting an RGB image to an HSV
image is cv2.COLOR_RGB2HSV.

The cv2.cvtColor() function returns the HSV image as a NumPy array. We can display the
HSV image using Matplotlib by converting it to the RGB color space using the
cv2.cvtColor() function with the cv2.COLOR_HSV2RGB color space conversion code.

@see: https://docs.opencv.org/4.x/d8/d01/group__imgproc__color__conversions.html
'''

# Converting the RGB image to the HSV image
hsv_image = cv2.cvtColor(original_image, cv2.COLOR_RGB2HSV)

# Splitting the HSV image into its three color channels
channel_h, channel_s, channel_v = cv2.split(hsv_image)

# Displaying the three color channels
print("Displaying the HSV Color Model and its channels")
plt.figure('HSV Color Model')

plt.subplot(2, 3, 1)
plt.imshow(channel_h, cmap='hsv')
plt.title('Hue Channel')
plt.axis('off')

plt.subplot(2, 3, 2)
plt.imshow(channel_s, cmap='gray')
plt.title('Saturation Channel')
plt.axis('off')

plt.subplot(2, 3, 3)
plt.imshow(channel_v, cmap='gray')
plt.title('Value Channel')
plt.axis('off')

plt.subplot(2, 3, 4)
plt.imshow(channel_h, cmap='gray')
plt.title('Hue Channel (BW)')
plt.axis('off')

plt.subplot(2, 3, 5)
plt.imshow(channel_s, cmap='gray')
plt.title('Saturation Channel (BW)')
plt.axis('off')

plt.subplot(2, 3, 6)
plt.imshow(channel_v, cmap='gray')
plt.title('Value Channel (BW)')
plt.axis('off')

plt.show()

'''
Masking Red Pixels in an Image

We can mask red pixels in an image by converting the image to the HSV color space and
applying a mask to the hue channel. The mask is created by defining the lower and upper
bounds of the red color range in the hue channel. The mask is then applied to the hue
and saturation channels to remove the red pixels from the image.

@see: https://en.wikipedia.org/wiki/Color_mask
@see: https://docs.opencv.org/4.x/d2/de8/group__core__array.html#ga48af0ab51e36436c5d04340e036ce981
'''

# Convert image to HSV
hsv_image = cv2.cvtColor(original_image, cv2.COLOR_RGB2HSV)

# Divide the HSV image into its three color channels
channel_h, channel_s, channel_v = cv2.split(hsv_image)

# Define the lower and upper bounds of the red color range

# Mask 1 are the red pixels with hue values between 0 and 10
lower_red = np.array([0, 100, 100])
upper_red = np.array([10, 255, 255])
mask_red_1 = cv2.inRange(hsv_image, lower_red, upper_red)

# Mask 2 are the red pixels with hue values between 160 and 180
lower_red = np.array([160, 100, 100])
upper_red = np.array([180, 255, 255])
mask_red_2 = cv2.inRange(hsv_image, lower_red, upper_red)

# Combine the two masks using the cv2.bitwise_or() function
mask_red = cv2.bitwise_or(mask_red_1, mask_red_2)

# Apply the mask to the hue and saturation channels
channel_h[mask_red == 0] = 0
channel_s[mask_red == 0] = 0

# Combine the modified hue and saturation channels with the original value channel
modified_HSV_image = cv2.merge([channel_h, channel_s, channel_v])

# Convert the modified HSV image to the RGB color space
modified_RGB_image = cv2.cvtColor(modified_HSV_image, cv2.COLOR_HSV2RGB)

# Display the modified image
print("Displaying the modified image with masked red pixels")
plt.figure('Image with Masked Red Pixels')

plt.subplot(1, 3, 1)
plt.imshow(original_image)
plt.title('Original Image')
plt.axis('off')

plt.subplot(1, 3, 2)
plt.imshow(mask_red, cmap='gray')
plt.title('Red Mask')
plt.axis('off')

plt.subplot(1, 3, 3)
plt.imshow(modified_RGB_image)
plt.title('Masked Red Pixels')
plt.axis('off')

plt.show()

'''
Masking Green Pixels in an Image

We can mask green pixels in an image by converting the image to the HSV color space and
applying a mask to the hue channel. The mask is created by defining the lower and upper
bounds of the green color range in the hue channel. The mask is then applied to the hue
and saturation channels to remove the green pixels from the image.

@see: https://en.wikipedia.org/wiki/Color_mask
'''

# Convert image to HSV
hsv_image = cv2.cvtColor(original_image, cv2.COLOR_RGB2HSV)

# Divide the HSV image into its three color channels
channel_h, channel_s, channel_v = cv2.split(hsv_image)

# Define the lower and upper bounds of the green color range
lower_green = np.array([36, 40, 40])
upper_green = np.array([72, 255, 255])
mask_green = cv2.inRange(hsv_image, lower_green, upper_green)

# Apply the mask to the hue and saturation channels
channel_h[mask_green == 0] = 0
channel_s[mask_green == 0] = 0

# Combine the modified hue and saturation channels with the original value channel
modified_HSV_image = cv2.merge([channel_h, channel_s, channel_v])

# Convert the modified HSV image to the RGB color space using the cv2.cvtColor() function
modified_RGB_image = cv2.cvtColor(modified_HSV_image, cv2.COLOR_HSV2RGB)

# Display the modified image
print("Displaying the modified image with masked green pixels")
plt.figure('Image with Masked Green Pixels')

plt.subplot(1, 3, 1)
plt.imshow(original_image)
plt.title('Original Image')
plt.axis('off')

plt.subplot(1, 3, 2)
plt.imshow(mask_green, cmap='gray')
plt.title('Green Mask')
plt.axis('off')

plt.subplot(1, 3, 3)
plt.imshow(modified_RGB_image)
plt.title('Masked Green Pixels')
plt.axis('off')

plt.show()

'''
Masking Blue Pixels in an Image

We can mask blue pixels in an image by converting the image to the HSV color space and
applying a mask to the hue channel. The mask is created by defining the lower and upper
bounds of the blue color range in the hue channel. The mask is then applied to the hue
and saturation channels to remove the blue pixels from the image.

@see: https://en.wikipedia.org/wiki/Color_mask
'''

# Convert image to HSV
hsv_image = cv2.cvtColor(original_image, cv2.COLOR_RGB2HSV)

# Divide the HSV image into its three color channels
channel_h, channel_s, channel_v = cv2.split(hsv_image)

# Define the lower and upper bounds of the blue color range
lower_blue = np.array([84, 40, 40])
upper_blue = np.array([140, 255, 255])
mask_blue = cv2.inRange(hsv_image, lower_blue, upper_blue)

# Apply the mask to the hue and saturation channels
channel_h[mask_blue == 0] = 0
channel_s[mask_blue == 0] = 0

# Combine the modified hue and saturation channels with the original value channel
modified_HSV_image = cv2.merge([channel_h, channel_s, channel_v])

# Convert the modified HSV image to the RGB color space using the cv2.cvtColor() function
modified_RGB_image = cv2.cvtColor(modified_HSV_image, cv2.COLOR_HSV2RGB)

# Display the modified image
print("Displaying the modified image with masked blue pixels")
plt.figure('Image with Masked Blue Pixels')

plt.subplot(1, 3, 1)
plt.imshow(original_image)
plt.title('Original Image')
plt.axis('off')

plt.subplot(1, 3, 2)
plt.imshow(mask_blue, cmap='gray')
plt.title('Blue Mask')
plt.axis('off')

plt.subplot(1, 3, 3)
plt.imshow(modified_RGB_image)
plt.title('Masked Blue Pixels')
plt.axis('off')

plt.show()

'''
Modifying the Hue of an Image

We can modify the hue of an image by converting the image to the HSV color space and
applying a transformation to the hue channel. The transformation is applied to the hue
channel by adding or subtracting a constant value from each pixel in the channel.

In the example below, we modify the hue of an image by adding a constant value to the
hue channel. We then combine the modified hue channel with the original saturation and
value channels to create a new HSV image. Finally, we convert the new HSV image to the
RGB color space to display the modified image.

Note: The hue channel in the HSV color space is represented as an angle between 0 and
360 degrees. Therefore, when adding or subtracting a constant value from the hue
channel, we need to ensure that the resulting hue values are within the range of 0 to
360 degrees.

@see: https://en.wikipedia.org/wiki/Hue
'''

# Convert image to HSV
hsv_image = cv2.cvtColor(original_image, cv2.COLOR_RGB2HSV)

# Divide the HSV image into its three color channels
channel_h, channel_s, channel_v = cv2.split(hsv_image)

# Modify the hue channel by adding a constant value
hue_shift = 60
channel_h = (channel_h + hue_shift) % 180

# Combine the modified hue and saturation channels with the original value channel
modified_HSV_image = cv2.merge([channel_h, channel_s, channel_v])

# Convert the modified HSV image to the RGB color space using the cv2.cvtColor() function
modified_RGB_image = cv2.cvtColor(modified_HSV_image, cv2.COLOR_HSV2RGB)

# Display the modified image
print("Displaying the modified image with shifted hue")
plt.figure('Image with Shifted Hue')

plt.subplot(1, 2, 1)
plt.imshow(original_image)
plt.title('Original Image')
plt.axis('off')

plt.subplot(1, 2, 2)
plt.imshow(modified_RGB_image)
plt.title('Image with Shifted Hue')
plt.axis('off')

plt.show()

'''
Selective Color Modification

We can selectively modify the color of an image by converting the image to the HSV color
space and applying a mask to the hue channel. The mask is created by defining the lower
and upper bounds of the color range in the hue channel. The mask is then applied to the
hue channel to modify the color of the pixels within the specified range.

@see: https://en.wikipedia.org/wiki/Color_mask
'''

# Convert image to HSV
hsv_image = cv2.cvtColor(original_image, cv2.COLOR_RGB2HSV)

# Divide the HSV image into its three color channels
channel_h, channel_s, channel_v = cv2.split(hsv_image)

# Define the lower and upper bounds of the color range

# Mask 1 are the red pixels with hue values between 0 and 10
lower_color = np.array([0, 100, 100])
upper_color = np.array([10, 255, 255])
mask_color_1 = cv2.inRange(hsv_image, lower_color, upper_color)

# Mask 2 are the red pixels with hue values between 160 and 180
lower_color = np.array([160, 100, 100])
upper_color = np.array([180, 255, 255])
mask_color_2 = cv2.inRange(hsv_image, lower_color, upper_color)

# Combine the two masks using the cv2.bitwise_or() function
mask_color = cv2.bitwise_or(mask_color_1, mask_color_2)

# Modify the hue channel by adding a constant value to the pixels within the color range
hue_shift = 15
channel_h[mask_color > 0] = (channel_h[mask_color > 0] + hue_shift) % 180

# Combine the modified hue and saturation channels with the original value channel
modified_HSV_image = cv2.merge([channel_h, channel_s, channel_v])

# Convert the modified HSV image to the RGB color space using the cv2.cvtColor() function
modified_RGB_image = cv2.cvtColor(modified_HSV_image, cv2.COLOR_HSV2RGB)

# Display the modified image
print("Displaying the modified image with selective color modification")
plt.figure('Image with Color Changed')

plt.subplot(1, 3, 1)
plt.imshow(original_image)
plt.title('Original Image')
plt.axis('off')

plt.subplot(1, 3, 2)
plt.imshow(mask_color, cmap='gray')
plt.title('Color Mask')
plt.axis('off')

plt.subplot(1, 3, 3)
plt.imshow(modified_RGB_image)
plt.title('Image with Selective Color Modification')
plt.axis('off')

plt.show()

'''
YUV Color Family

The YUV color family is a color model used in video compression and television
broadcasting.

It consists of three components:
- Y (luminance)
    - Represents the brightness of the color
    - Similar to the grayscale image
- U (chrominance, primarily blue-yellow)
    - U = B - Y (chrominance, primarily blue-yellow)
- V (chrominance, primarily red-green)
    - V = R - Y (chrominance, primarily red-green)

The Y component represents the brightness of the color, while the U and V components represent
the color information.

The YUV color family is used to represent colors in a way that is more efficient for
video compression than the RGB color model. It separates the brightness and color
information of the image, allowing for better compression and transmission of video
data.

@see: https://en.wikipedia.org/wiki/YUV
'''

# Convert image to YUV
yuv_image = cv2.cvtColor(original_image, cv2.COLOR_RGB2YUV)

# Divide the YUV image into its three color channels
channel_y, channel_u, channel_v = cv2.split(yuv_image)

# Displaying the three color channels
print("Displaying the YUV Color Model and its channels")
plt.figure('YUV Color Model')

plt.subplot(2, 3, 1)
plt.imshow(channel_y)
plt.title('Y Channel')
plt.axis('off')

plt.subplot(2, 3, 2)
plt.imshow(channel_u)
plt.title('U Channel')
plt.axis('off')

plt.subplot(2, 3, 3)
plt.imshow(channel_v)
plt.title('V Channel')
plt.axis('off')

plt.subplot(2, 3, 4)
plt.imshow(channel_y, cmap='gray')
plt.title('Y Channel (BW)')
plt.axis('off')

plt.subplot(2, 3, 5)
plt.imshow(channel_u, cmap='gray')
plt.title('U Channel (BW)')
plt.axis('off')

plt.subplot(2, 3, 6)
plt.imshow(channel_v, cmap='gray')
plt.title('V Channel (BW)')
plt.axis('off')

plt.show()

'''
CIE L*a*b* Color Model

The CIE L*a*b* color model is a color model used in color science and color
management. CIE stands for the International Commission on Illumination or
Commission Internationale de l'Eclairage in French, an organization that
develops standards for color science.

The CIE L*a*b* color model consists of three components:
- L* (lightness)
    - Represents the brightness of the color
- a* (green-red)
    - a* = (green - red) component
- b* (blue-yellow)
    - b* = (blue - yellow) component
    
The L* component represents the lightness of the color, while the a* and b*
components represent the color information. The CIE L*a*b* color model is used
to represent colors in a way that is more perceptually uniform than the RGB color
model. It is often used in color science, color management, and color correction
applications.

@see: https://en.wikipedia.org/wiki/CIELAB_color_space
'''

# Convert image to CIE L*a*b*
lab_image = cv2.cvtColor(original_image, cv2.COLOR_RGB2LAB)

# Divide the CIE L*a*b* image into its three color channels
channel_l, channel_a, channel_b = cv2.split(lab_image)

# Displaying the three color channels
print("Displaying the CIE L*a*b* Color Model and its channels")
plt.figure('CIE L*a*b* Color Model')

plt.subplot(2, 3, 1)
plt.imshow(channel_l)
plt.title('L* Channel')
plt.axis('off')

plt.subplot(2, 3, 2)
plt.imshow(channel_a)
plt.title('a* Channel')
plt.axis('off')

plt.subplot(2, 3, 3)
plt.imshow(channel_b)
plt.title('b* Channel')
plt.axis('off')

plt.subplot(2, 3, 4)
plt.imshow(channel_l, cmap='gray')
plt.title('L* Channel (BW)')
plt.axis('off')

plt.subplot(2, 3, 5)
plt.imshow(channel_a, cmap='gray')
plt.title('a* Channel (BW)')
plt.axis('off')

plt.subplot(2, 3, 6)
plt.imshow(channel_b, cmap='gray')
plt.title('b* Channel (BW)')
plt.axis('off')

plt.show()