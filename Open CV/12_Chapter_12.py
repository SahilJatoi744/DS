# Setting of Camera or video

import cv2
import numpy as np

cap = cv2.VideoCapture(0)

cap.set(10, 50) # 10 is the key to set brightness
cap.set(3, 200)  # width
cap.set(4, 480)  # height

while(True):
    ret, frame = cap.read()
    if ret == True:
        cv2.imshow("frame",frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    else:
        break

cap.release()
cv2.destroyAllWindows()