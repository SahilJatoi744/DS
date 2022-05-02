# How to capture a webcame inside python

# Step 1 Import Libraries
import cv2 as cv
import numpy as np
# Step 2 Read the Frames from Camera
cap = cv.VideoCapture(0) #webcame no 1 or pc one
# read until the end
# Step 3 Display frame by frame
while(cap.isOpened()):
    # capture frame by frame
    ret, frame = cap.read()
    if ret == True:
        # to display frame
        cv.imshow("Frame",frame)
        if cv.waitKey(1) & 0XFF == ord('q'):
            break
    else:
        break
# Step 4 release or close windows easily
cap.release()
cv.destroyAllWindows()
