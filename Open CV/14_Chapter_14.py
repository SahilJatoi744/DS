# how to draw lines, and shape in python

import cv2 as cv
import numpy as np

# Draw a canvas
# 0 is for black
# 1 is for white
img = np.zeros((600, 600))  # black
img1 = np.ones((600, 600))  # white

#print size
print("The size of our canvas is : ", img.shape)
#print(img) to print the matrix: black
#print(img1) to print the matrix: white

# adding colors to canvas
colored_img = np.zeros((600, 600, 3), np.uint8)  # color channed addition

colored_img[:] = 255, 0, 179  # color complete image

colored_img[150:230, 100:207] = 255, 0, 0  # color part of image

# adding line

cv.line(colored_img, (0, 0),
        (colored_img.shape[0], (colored_img.shape[1])), (255, 0, 0), 3)

cv.line(colored_img, (100, 100), (300, 300), (255, 255, 50), 3)

# adding rectangle
cv.rectangle(colored_img, (50, 100), (300, 400), (255, 253, 355), 3)
# color filled
cv.rectangle(colored_img, (50, 100), (300, 400), (255, 253, 355), cv.FILLED)
#adding Circle
cv.circle(colored_img, (400, 300), 50, (250, 100, 0), 5)
# filled
cv.circle(colored_img, (400, 300), 50, (250, 100, 0), cv.FILLED)

# adding text
text = 'Chilla \n from \n Pakistan'
y = 300
for i, txt in enumerate(text.split('\n')):
    cv.putText(colored_img, txt, (200, y),
               cv.FONT_HERSHEY_COMPLEX_SMALL, 1, (255, 255, 9), 2)
    y = y+30
#cv.imshow("Black",img)
#cv.imshow("White",img1)
cv.imshow("Colored Image", colored_img)

cv.waitKey(0)
cv.destroyAllWindows()
