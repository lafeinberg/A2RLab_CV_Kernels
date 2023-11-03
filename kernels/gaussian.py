import cv2
import numpy as np
import matplotlib.pyplot as plt

# gaussian can be used both in colored or greyscale input image

# load greyscale image
# image = cv2.imread('img/bittle.jpeg', cv2.IMREAD_GRAYSCALE)

# load colored image 
image = cv2.imread('img/a2r.png', cv2.IMREAD_COLOR)

# the bigger the sigma , it gets very blurred 
sigma =  1.0 
kernel_size = (5,5)

smoothed_image = cv2.GaussianBlur(image, kernel_size, sigma)

plt.subplot(1, 2, 1)
plt.imshow(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
plt.title('Original Image')
plt.xticks([]), plt.yticks([])

plt.subplot(1, 2, 2)
plt.imshow(cv2.cvtColor(smoothed_image, cv2.COLOR_BGR2RGB))
plt.title('Gaussian Filter (Smoothed)')
plt.xticks([]), plt.yticks([])

plt.show()



