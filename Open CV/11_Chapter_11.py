# Writting videos from cam
import cv2

cap = cv2.VideoCapture(0)
#writing format, codec, video writter object and file output
frame_width = int(cap.get(3))
frame_height = int(cap.get(4))
out = cv2.VideoWriter("Resources/Cam_video.avi", cv2.VideoWriter_fourcc('M', "J", 'P', 'G'), 20, (frame_width, frame_height))

while (True):
    (ret, frame) = cap.read()
    (thresh, b_w) = cv2.threshold(frame, 127, 255, cv2.THRESH_BINARY)
    # tp show in player
    if ret == True:
        out.write(b_w)
        cv2.imshow('Video', b_w)
        # to quit with q key
        if cv2.waitKey(1) & 0XFF == ord('q'):
            break
    else:
        break
cap.release()
out.release()
cv2.destroyAllWindows()
