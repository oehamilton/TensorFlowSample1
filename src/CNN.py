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
# Save the image using OpenCV   