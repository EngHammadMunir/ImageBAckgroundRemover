import cv2
import numpy as np

# Load the image
img = cv2.imread("image.jpg")

# Convert the image to grayscale
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

# Apply a threshold to the image to turn it into a binary image
thresh = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)[1]

# Find the contours in the image
contours, hierarchy = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

# Find the largest contour
largest_contour = max(contours, key=cv2.contourArea)

# Create a mask with the same size as the original image, filled with zeros
mask = np.zeros(img.shape[:2], dtype=np.uint8)

# Draw the largest contour on the mask
cv2.drawContours(mask, [largest_contour], -1, (255, 255, 255), -1)

# Apply the mask to the original image to remove the background
result = cv2.bitwise_and(img, img, mask=mask)

# Save the result
cv2.imwrite("result.jpg", result)
