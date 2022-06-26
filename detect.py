import cv2
import pickle
import numpy as np
from mediapipe.python.solutions import selfie_segmentation
import os
from LaSAHW.image import capter
import mediapipe as mp

mp_pose = mp.solutions.pose
mp_draw = mp.solutions.drawing_utils
pose = mp_pose.Pose()

mp_selfie_segmentation = mp.solutions.selfie_segmentation
selfie_segmentation = mp_selfie_segmentation.SelfieSegmentation(model_selection=1)
width, height = 450, 480
posList = []
image_path = 'images'
images = os.listdir(image_path)

image_index= 0
bg_image = cv2.imread(image_path+'/'+images[image_index])
capter("image.png")
while True:
    cap = cv2.VideoCapture(0)
    while cap.isOpened():
            _, frame = cap.read()
            # flip the frame to horizontal direction
            frame = cv2.flip(frame, 1)
            cv2.rectangle(frame,(200,150),(450,480),(255,0,255),2)
            cv2.imshow('detect',frame)
            cv2.waitKey(1)

            RGB = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            results = selfie_segmentation.process(RGB)
            mask = results.segmentation_mask
            cv2.imshow('mask',mask)
            # it returns true or false where the condition applies in the mask
            condition = np.stack(
                    (results.segmentation_mask,) * 3, axis=-1) > 0.6
            output_image = np.where(condition, frame, bg_image)
            bg_image = cv2.resize(mask, (width, height))
            font = cv2.FONT_HERSHEY_SIMPLEX
            ratio_white =cv2.countNonZero(bg_image)/216000
            percent =ratio_white
            print(percent)
            if percent > 60:
                    output_image = cv2.putText(output_image,
                               'bonne position',
                                (10, 10),
                                font, 1,
                                (255, 0, 0),
                                2,
                                cv2.LINE_4)
            else :
                    output_image = cv2.putText(output_image,
                        'positionnez-vous sur le bonhomme',
                        (10, 10),
                        font, 1,
                        (255, 0, 0),
                        2,
                        cv2.LINE_4)
            cv2.imshow("Output", output_image)