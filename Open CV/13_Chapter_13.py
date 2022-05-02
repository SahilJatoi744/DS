# basic functions or manipulation in one cv2

import cv2
import numpy
img = cv2.imread("Resources/GateM.jpg")

# resize
resized_img = cv2.resize(img, (500, 700))

# gray
gray_img = cv2.cv2tColor(img, cv2.COLOR_BGR2GRAY)

#black and white

(thresh, b_w) = cv2.threshold(resized_img, 127, 255, cv2.THRESH_BINARY)

# blurred image
blur_img = cv2.GaussianBlur(img, (23, 23), 0)  # 7,7 odd matrix

# Edge detection
edge_img = cv2.Canny(resized_img, 53, 53)

# Thickness of lines

# 7,7 always odd numbers
mat_kernal = numpy.ones((3, 3), numpy.uint8)
dilated_img = cv2.dilate(edge_img, (3, 3), iterations=1)
# make thinner outline or erosion

ero_img = cv2.erode(dilated_img, mat_kernal, iterations=1)

# cropping image and for it we will be using numpy

print("The size of our image is: ", img.shape)
# first one is height and second one is width
cropped_img = resized_img[0:200, 200:300]

#cv2.imshow("Original", resized_img)
#cv2.imshow("Gray Image", gray_img)
#cv2.imshow("Blur Image", blur_img)
cv2.imshow("Black and white Image", b_w)
#cv2.imshow("Edge Image", edge_img)
#cv2.imshow("Dilated Image", dilated_img)
#cv2.imshow("Erosion Image", ero_img)
cv2.imshow("Cropped Image", cropped_img)


#cv2.imshow("Resized image", resized_img)

cv2.waitKey(0)
cv2.destroyAllWindows()
