from __future__ import print_function
import cv2


backSub = cv2.createBackgroundSubtractorMOG2()

#backSub = cv.createBackgroundSubtractorKNN()


frame = cv2.imread("resources/openai.gym.1568127083.838687.41524/state/1.jpg")
fgMask = backSub.apply(frame)
cv2.rectangle(frame, (10, 2), (100, 20), (255, 255, 255), -1)
cv2.imshow('Frame', frame)
cv2.imshow('FG Mask', fgMask)
