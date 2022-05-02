# saving HD recording of cam steaming

# how to change from out Resolution of cam

from re import L
import cv2 as cv
import numpy as np

cap = cv.VideoCapture(0)
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
#writing format, codec, video writter object and file output
frame_width = int(cap.get(3))
frame_height = int(cap.get(4))
out = cv.VideoWriter("Resources/VideoCP16.avi", cv.VideoWriter_fourcc('M',
                     "J", 'P', 'G'), 30, (frame_width, frame_height))

while (True):
    (ret, frame) = cap.read()
    # tp show in player
    if ret == True:
        out.write(frame)
        cv.imshow('Video', frame)
        # to quit with q key
        if cv.waitKey(1) & 0XFF == ord('q'):
            break
    else:
        break
cap.release()
cv.destroyAllWindows()
