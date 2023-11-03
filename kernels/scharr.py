import cv2
import numpy as np
import matplotlib.pyplot as plt

# Load colored image and convert it to grayscale
colored_image = cv2.imread('img/bittle.jpeg', cv2.IMREAD_COLOR)
image = cv2.cvtColor(colored_image, cv2.COLOR_BGR2GRAY)

# Scharr operators
scharr_x = np.array([[-3, 0, 3], [-10, 0, 10], [-3, 0, 3]])
scharr_y = np.array([[-3, -10, -3], [0, 0, 0], [3, 10, 3]])

# Apply Scharr operators using cv2.filter2D
scharr_x_result = cv2.filter2D(image, -1, scharr_x)
scharr_y_result = cv2.filter2D(image, -1, scharr_y)

# Compute magnitude and direction
gradient_magnitude = np.sqrt(scharr_x_result ** 2 + scharr_y_result ** 2)
gradient_direction = np.arctan2(scharr_y_result, scharr_x_result)

plt.subplot(1, 3, 1)
plt.imshow(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
plt.title('Greyscale Image')
plt.xticks([]), plt.yticks([])

plt.subplot(1, 3, 2)
plt.imshow(scharr_x_result, cmap='gray')
plt.title('ScharrX')
plt.xticks([]), plt.yticks([])

plt.subplot(1, 3, 3)
plt.imshow(scharr_y_result, cmap='gray')
plt.title('ScharrY')
plt.xticks([]), plt.yticks([])

plt.show()