import cv2
import numpy as np
import matplotlib.pyplot as plt


# create the 2D LoG kernel filter
# LoG kernel is dervied from the second derivative of Gaussian 
# https://homepages.inf.ed.ac.uk/rbf/HIPR2/log.htm
def create_log_kernel(sigma, kernel_size): 
    center = kernel_size+1/2 
    y, x = np.ogrid[-center: center + 1, -center: center + 1]
    kernel = -(1 / (np.pi * sigma**4)) * (1 - (x**2 + y**2) / (2 * sigma**2)) * np.exp(-(x**2 + y**2) / (2 * sigma**2))
    
    return kernel

def main():

    # load greyscale image
    # image = cv2.imread('image.jpg', cv2.IMREAD_GRAYSCALE)

    # load image colored image 
    # convert colored image to greyscale
    colored_image = cv2.imread('img/bittle.jpeg', cv2.IMREAD_COLOR)
    image = cv2.cvtColor(colored_image, cv2.COLOR_BGR2GRAY)

    sigma = 1.0

    # should be an odd number 
    kernel_size = 5

    log_kernel = create_log_kernel(sigma, kernel_size)

    filtered_image = cv2.filter2D(image, cv2.CV_64F, log_kernel)

     # Apply thresholding to create a binary image
    threshold_value = 0  # Adjust this threshold as needed
    _, binary_image = cv2.threshold(filtered_image, threshold_value, 255, cv2.THRESH_BINARY)


    plt.subplot(1, 3, 1)
    plt.imshow(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
    plt.title('Greyscale Image')
    plt.xticks([]), plt.yticks([])

    plt.subplot(1, 3, 2)
    plt.imshow(filtered_image, cmap='gray')
    plt.title('Laplacian of Gaussian (LoG) Filter')
    plt.xticks([]), plt.yticks([])

    plt.subplot(1, 3, 3)
    plt.imshow(binary_image, cmap='gray')
    plt.title('Binary Image (Thresholded) LoG Filter')
    plt.xticks([]), plt.yticks([])

    plt.show()

if __name__ == "__main__":
    main()
