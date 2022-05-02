import cv2 as cv

cap = cv.VideoCapture(0)  # webcame no 1 or pc one

while(True):
    (ret, frame) = cap.read()
    gray = cv.cvtColor(frame, cv.COLOR_BGR2GRAY)
    (thresh, b_w) = cv.threshold(gray, 127, 255, cv.THRESH_BINARY)

    cv.imshow("Original", frame)
    cv.imshow("Gray", gray)
    cv.imshow("Black & White",b_w)

    if cv.waitKey(1) & 0XFF == ord('q'):
        break

cap.release()
cv.destroyAllWindows()
