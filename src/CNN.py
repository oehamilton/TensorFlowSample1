import cv2
import numpy as np
from scipy import datasets

i = datasets.ascent()

import matplotlib.pyplot as plt
plt.grid(False)
plt.gray()
plt.axis('off')
# Display the image using matplotlib
plt.imshow(i)
plt.show()


#Save the size of the image 

i_transformed = np.copy(i)
size_x = i_transformed.shape[0]
size_y = i_transformed.shape[1]

print("Size of the image: ", size_x, size_y)
# Resize the image to 28x28 pixels