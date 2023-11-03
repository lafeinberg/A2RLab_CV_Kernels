import cv2
import numpy as np
import matplotlib.pyplot as plt


# Sobel operator is used in greyscale images

# load greyscale image
# image = cv2.imread('image.jpg', cv2.IMREAD_GRAYSCALE)

# load image colored image 
# convert colored image to greyscale
colored_image = cv2.imread('img/bittle.jpeg', cv2.IMREAD_COLOR)
image = cv2.cvtColor(colored_image, cv2.COLOR_BGR2GRAY)

# sobel operators 
sobel_x = np.array([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]])
sobel_y = np.array([[-1, -2, -1], [0, 0, 0], [1, 2, 1]])

# cv2.filter2D : apply cutom convolutional kernel to an image 
# -1 : indicated the output image should have the same data type as the input image
sobel_x_result = cv2.filter2D(image, -1, sobel_x)
sobel_y_result = cv2.filter2D(image, -1, sobel_y)

# formula compute magnitude and orientation/direction 
gradient_magnitude = np.sqrt(sobel_x_result ** 2 + sobel_y_result ** 2)
gradient_direction = np.arctan2(sobel_y_result, sobel_x_result)

plt.subplot(1, 3, 1)
plt.imshow(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
plt.title('Greyscale Image')
plt.xticks([]), plt.yticks([])

plt.subplot(1, 3, 2)
plt.imshow(sobel_x_result, cmap='gray')
plt.title('SobelX')
plt.xticks([]), plt.yticks([])

plt.subplot(1,3,3)
plt.imshow(sobel_y_result, cmap='gray')
plt.title('SobelY')
plt.xticks([]), plt.yticks([])

plt.show()





