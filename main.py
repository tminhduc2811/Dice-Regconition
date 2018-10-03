from cv2 import *
import numpy as np
import imutils


def calc_dice_num(cont):
    temp = 0
    for cn in cont:
        appr = approxPolyDP(cn, 0.04 * arcLength(cn, True), True)
        if len(appr) > 4:
            temp += 1
    return str(temp)


def reg_dice_number(dice_cont, cx_dice, cy_dice):
    temp = 0
    for i in dice_cont:
        if len(approxPolyDP(i, 0.04 * arcLength(i, True), True)) > 4:
            md = moments(i)
            if md["m00"] != 0:
                dx = int((md["m10"] / md["m00"]))
                dy = int((md["m01"] / md["m00"]))
                if (abs(dx - cx_dice) <= 60) & (abs(dy - cy_dice) <= 60):
                    temp += 1
    return str(temp)


# Define output
fourcc = cv2.VideoWriter_fourcc(*'mp4v')
out = cv2.VideoWriter('D:/Learning/Computer Vision/Sources/output.mp4', fourcc, 20.0, (1300, 731))

cap = VideoCapture('D:/Learning/Computer Vision/Sources/dices.mov')
played = True
while cap.isOpened():
    if played is True:
        ret, frame = cap.read()
        if frame is None:
            break
        frame = imutils.resize(frame, width=1300)
        w, h = frame.shape[:2]
        gray = cvtColor(frame, COLOR_BGR2GRAY)
        r, bw = threshold(gray, 165, 255, THRESH_BINARY)
        im2, contours, hierarchy = findContours(bw, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
        num = calc_dice_num(contours)
        putText(frame, num, (100, 120), FONT_HERSHEY_DUPLEX, 5, (250, 0, 250), 10)
        for c in contours:
            approx = approxPolyDP(c, 0.04 * arcLength(c, True), True)
            if len(approx) > 4:
                drawContours(frame, [c], -1, (0, 0, 255), 6)
            elif len(approx) == 4:
                M = moments(c)
                cx = int((M["m10"] / M["m00"]))
                cy = int((M["m01"] / M["m00"]))
                result = reg_dice_number(contours, cx, cy)
                putText(frame, result, (cx + 60, cy + 60), FONT_HERSHEY_DUPLEX, 5, (250, 0, 250), 10)
        out.write(frame)
        imshow('frame', frame)
    if waitKey(1) & 0xFF == ord('q'):
        break

out.release()
cap.release()
destroyAllWindows()
