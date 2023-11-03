import cv2
import numpy as np
import matplotlib.pyplot as plt

def prewitt():

    colored_image = cv2.imread('img/bittle.jpeg', cv2.IMREAD_COLOR)
    image = cv2.cvtColor(colored_image, cv2.COLOR_BGR2GRAY)

    
    # Define the Prewitt kernel for horizontal edge detection
    kernel_x = np.array([[-1, 0, 1],
                        [-1, 0, 1],
                        [-1, 0, 1]])
    
    # Define the Prewitt kernel for vertical edge detection
    kernel_y = np.array([[-1, -1, -1],
                        [0, 0, 0],
                        [1, 1, 1]])
    
    kernel_x_result = cv2.filter2D(image, -1, kernel_x)
    kernel_y_result = cv2.filter2D(image, -1, kernel_y)

    plt.subplot(1, 3, 1)
    plt.imshow(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
    plt.title('Greyscale Image')
    plt.xticks([]), plt.yticks([])

    plt.subplot(1, 3, 2)
    plt.imshow(kernel_x_result, cmap='gray')
    plt.title('PrewittX')
    plt.xticks([]), plt.yticks([])

    plt.subplot(1,3,3)
    plt.imshow(kernel_y_result, cmap='gray')
    plt.title('PrewittY')
    plt.xticks([]), plt.yticks([])

    plt.show()

prewitt()
    