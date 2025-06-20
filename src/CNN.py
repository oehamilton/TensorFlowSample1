import cv2
import numpy as np
from scipy import datasets
import matplotlib.pyplot as plt

# Load the image and convert to float32
i = datasets.ascent().astype(np.float32)  # Convert to float32 to handle negative values

# Display the original image
plt.grid(False)
plt.gray()
plt.axis('off')
plt.imshow(i, vmin=0, vmax=255)  # Specify range for float32 display
plt.show()

# Save the size of the image
i_transformed = np.copy(i).astype(np.float32)  # Use float32 for convolution
size_x = i_transformed.shape[0]
size_y = i_transformed.shape[1]

print("Size of the image: ", size_x, size_y)

# Define the filter
filter = [[0, -1, 0], [1, -4, 1], [0, 2, 0]]  # Filter with negative value

# Normalize the filter with a weight
weight = 1.0

# Apply the filter to the image
for x in range(1, size_x-1):
    for y in range(1, size_y-1):
        output_pixel = 0.0
        output_pixel += i[x-1, y-1] * filter[0][0]
        output_pixel += i[x, y-1] * filter[0][1]
        output_pixel += i[x+1, y-1] * filter[0][2]
        output_pixel += i[x-1, y] * filter[1][0]
        output_pixel += i[x, y] * filter[1][1]
        output_pixel += i[x+1, y] * filter[1][2]
        output_pixel += i[x-1, y+1] * filter[2][0]
        output_pixel += i[x, y+1] * filter[2][1]
        output_pixel += i[x+1, y+1] * filter[2][2]
        output_pixel *= weight
        # Clip to valid uint8 range
        i_transformed[x, y] = np.clip(output_pixel, 0, 255)

# Convert to uint8 for display and pooling
i_transformed = i_transformed.astype(np.uint8)

# Print the size of the filtered image
size_x_filtered = i_transformed.shape[0]
size_y_filtered = i_transformed.shape[1]
print("Size of the filtered image: ", size_x_filtered, size_y_filtered)

# Display the filtered image
plt.gray()
plt.grid(False)
plt.imshow(i_transformed)
plt.axis('off')
plt.show()

# Pooling
new_x = int(size_x / 2)
new_y = int(size_y / 2)
newImage = np.zeros((new_x, new_y), dtype=np.uint8)  # Use uint8 for output
for x in range(0, size_x, 2):
    for y in range(0, size_y, 2):
        pixels = []
        pixels.append(i_transformed[x, y])
        # Safely handle boundaries
        pixels.append(i_transformed[x+1, y] if x+1 < size_x else 0)
        pixels.append(i_transformed[x, y+1] if y+1 < size_y else 0)
        pixels.append(i_transformed[x+1, y+1] if x+1 < size_x and y+1 < size_y else 0)
        newImage[int(x/2), int(y/2)] = max(pixels)

# Print the size of the pooled image
size_x_pooled = newImage.shape[0]
size_y_pooled = newImage.shape[1]
print("Size of the pooled image: ", size_x_pooled, size_y_pooled)

# Display the pooled image
plt.gray()
plt.grid(False)
plt.imshow(newImage)
plt.axis('off')
plt.show()