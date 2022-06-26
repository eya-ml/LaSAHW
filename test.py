import cv2
import numpy as np
import mediapipe as mp
import matplotlib.pyplot as plt
def percent(ch):
    mp_selfie_segmentation = mp.solutions.selfie_segmentation
    selfie_segmentation = mp_selfie_segmentation.SelfieSegmentation(model_selection=1)
    while True:
        frame = cv2.imread('images/'+ch)

        RGB = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
  # get the result
        results = selfie_segmentation.process(RGB)
  # extract segmented mask
        mask = results.segmentation_mask
  # show outputs
        cv2.imshow("mask", mask)
        cv2.imshow("Frame", frame)
        key = cv2.waitKey(1)
        ratio_white = cv2.countNonZero(mask) / (frame.size / 3)
        percent = ratio_white * 100
        print("percent = ", percent)
        if key == ord('q'):
            break

    return percent