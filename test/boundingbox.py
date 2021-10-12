'''
# File: boundingbox.py
# Project: rock-paper-scissors
# Created Date: 2021-10-10 9:36
# Author: Yutong Dai yutongdai95@gmail.com
# -----
# Last Modified: 2021-10-11 9:41
# Modified By: Yutong Dai yutongdai95@gmail.com
# 
# This code is published under the MIT License.
# -----
# HISTORY:
# Date      	By 	Comments
# ----------	---	----------------------------------------------------------
'''
import numpy as np
import cv2
import mediapipe as mp
mphands = mp.solutions.hands
hands = mphands.Hands(min_detection_confidence=0.5, min_tracking_confidence=0.5, max_num_hands=2)
mp_drawing = mp.solutions.drawing_utils
cap = cv2.VideoCapture(0)
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 900)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 600)
_, frame = cap.read()
bgimage = np.ones_like(frame, dtype=np.uint8) * 225
# img_binary = None
h, w, c = frame.shape
tol=0
target_ratio = 2/3
while cap.isOpened():
    success, frame = cap.read()
    if not success:
      print("Ignoring empty camera frame.")
      # If loading a video, use 'break' instead of 'continue'.
      continue
    frame.flags.writeable = False
    framergb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    result = hands.process(framergb)
    hand_landmarks = result.multi_hand_landmarks
    if hand_landmarks:
        bgimage = np.ones_like(frame, dtype=np.uint8) * 225
        for idx, handLMs in enumerate(hand_landmarks):
            x_max = 0
            y_max = 0
            x_min = w
            y_min = h
            for lm in handLMs.landmark:
                x, y = int(lm.x * w), int(lm.y * h)
                if x > x_max:
                    x_max = x
                if x < x_min:
                    x_min = x
                if y > y_max:
                    y_max = y
                if y < y_min:
                    y_min = y
            x_min, y_min, x_max, y_max = x_min-tol, y_min-tol, x_max+tol, y_max+tol
            xlen = x_max - x_min
            ylen = y_max - y_min
            ratio = xlen / ylen
            if ratio <= target_ratio:
                xlen = ylen * target_ratio
                offset = int(xlen//2 + 1)
                x_mid = (x_min + x_max) // 2
                x_min = max(0,x_mid - offset)
                x_max = min(x_mid + offset, w)
            else:
                ylen = xlen / target_ratio
                offset = int(ylen // 2 + 1)
                y_mid = (y_min + y_max) // 2
                y_min = max(y_mid - offset, 0)
                y_max = min(y_mid + offset, h)
            print(f"hand:{idx} | BBOX:{x_min, y_min, x_max, y_max}")
            cv2.rectangle(frame, (x_min, y_min), (x_max, y_max), (0, 255, 0), 2)
            bgimage[y_min:y_max, x_min:x_max, :] = frame[y_min:y_max, x_min:x_max, :]
            # grayimg = cv2.cvtColor(bgimage, cv2.COLOR_BGR2GRAY)
            # _, img_binary = cv2.threshold(grayimg, 0, 255, cv2.THRESH_BINARY+cv2.THRESH_OTSU)
    cv2.imshow('Live', cv2.flip(frame, 1))
    cv2.imshow('Hands', cv2.flip(bgimage, 1))
    # if img_binary is not None:
        # cv2.imshow('Binary Hands', cv2.flip(img_binary, 1))
    if hand_landmarks:
        print("==== next frame ====")
    if cv2.waitKey(5) & 0xFF == 27:
      break
cap.release()