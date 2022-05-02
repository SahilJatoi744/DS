# how to change from out Resolution of cam

from re import L
import cv2 as cv
import numpy as np

cap = cv.VideoCapture(0)
cap.set(cv.CAP_PROP_FOURCC, cv.VideoWriter_fourcc(
    'M', 'J', 'P', 'G'))  # depends on fourcc available camera
#cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
#cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
cap.set(cv.CAP_PROP_FPS, 30)
print(cap.get(cv.CAP_PROP_FPS))
# resolution


def hd_resolution():
    cap.set(3, 1280)  # width
    cap.set(4, 720)  # height


def sd_resolution():
    cap.set(3, 640)  # width
    cap.set(4, 480)  # height


def fhd_resolution():
    cap.set(3, 1920)  # width
    cap.set(4, 1080)  # height


fhd_resolution()


while(True):
    ret, frame = cap.read()
    if ret == True:
        cv.imshow("Camera", frame)
        if cv.waitKey(1) & 0xFF == ord('q'):
            break
    else:
        break

cap.release()
cv.destroyAllWindows()
