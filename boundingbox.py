'''
# File: boundingbox.py
# Project: rock-paper-scissors
# Created Date: 2021-10-10 9:36
# Author: Yutong Dai yutongdai95@gmail.com
# -----
# Last Modified: 2021-10-10 11:07
# Modified By: Yutong Dai yutongdai95@gmail.com
# 
# This code is published under the MIT License.
# -----
# HISTORY:
# Date      	By 	Comments
# ----------	---	----------------------------------------------------------
'''
import cv2
import mediapipe as mp
mphands = mp.solutions.hands
hands = mphands.Hands(min_detection_confidence=0.5, min_tracking_confidence=0.5, max_num_hands=2)
mp_drawing = mp.solutions.drawing_utils
cap = cv2.VideoCapture(0)

_, frame = cap.read()

h, w, c = frame.shape
tol=1
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
            print(f"hand:{idx} | BBOX:{x_min, y_min, x_max, y_max}")
            x_min, y_min, x_max, y_max = x_min-tol, y_min-tol, x_max+tol, y_max+tol
            cv2.rectangle(frame, (x_min, y_min), (x_max, y_max), (0, 255, 0), 2)
    cv2.imshow('MediaPipe Hands', cv2.flip(frame, 1))
    if hand_landmarks:
        print("==== next frame ====")
    if cv2.waitKey(5) & 0xFF == 27:
      break
cap.release()