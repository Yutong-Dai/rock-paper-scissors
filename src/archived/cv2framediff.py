'''
# File: cv2framediff.py
# Project: src
# Created Date: 2021-10-11 5:29
# Author: Yutong Dai yutongdai95@gmail.com
# -----
# Last Modified: 2021-10-11 5:48
# Modified By: Yutong Dai yutongdai95@gmail.com
# 
# This code is published under the MIT License.
# -----
# HISTORY:
# Date      	By 	Comments
# ----------	---	----------------------------------------------------------
'''
import cv2
cap = cv2.VideoCapture(0)
fgbg = cv2.createBackgroundSubtractorMOG2()
while True:
    success, frame = cap.read()
    if not success:
        print("Ignoring empty camera frame.")
        continue
    mask = fgbg.apply(frame)
    cv2.imshow("Frame", cv2.flip(frame, 1))
    cv2.imshow("FG MASK Frame", cv2.flip(mask, 1))
    if cv2.waitKey(5) & 0xFF == 27:
        break
cap.release()
