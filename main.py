
import os
import cv2
import time
import numpy as np
import mediapipe as mp

from LaSAHW.teste import position


def background_removal(record):
# initialize mediapipe
    mp_pose = mp.solutions.pose
    mp_draw = mp.solutions.drawing_utils
    pose = mp_pose.Pose()

    mp_selfie_segmentation = mp.solutions.selfie_segmentation
    selfie_segmentation = mp_selfie_segmentation.SelfieSegmentation(model_selection=1)

# store background images in a list
    image_path = 'images'
    images = os.listdir(image_path)

    image_index= 0
    bg_image = cv2.imread(image_path+'/'+images[image_index])

# create videocapture object to access the webcam
    cap = cv2.VideoCapture(0)

    while cap.isOpened():
        _, frame = cap.read()

    # flip the frame to horizontal direction
        frame = cv2.flip(frame, 1)
        height, width, channel = frame.shape

        RGB = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    # get the result
        results = selfie_segmentation.process(RGB)
        results1 = pose.process(frame)
    # draw the detected pose on original video/ live stream
        mp_draw.draw_landmarks(frame, results1.pose_landmarks, mp_pose.POSE_CONNECTIONS,
                           mp_draw.DrawingSpec((255, 0, 0), 2, 2),
                           mp_draw.DrawingSpec((255, 0, 255), 2, 2)
                           )
    # Display pose on original video/live stream

    # Extract and draw pose on plain white image
        h, w, c = frame.shape   # get shape of original frame
        opImg = np.zeros([h, w, c])  # create blank image with original frame size
        opImg.fill(255)  # set white background. put 0 if you want to make it black
        print(mp_draw.draw_landmarks)
    # draw extracted pose on black white image
        mp_draw.draw_landmarks(opImg, results1.pose_landmarks, mp_pose.POSE_CONNECTIONS,
                           mp_draw.DrawingSpec((255, 0, 0), 2, 2),
                           mp_draw.DrawingSpec((255, 0, 255), 2, 2)
                           )
    # display extracted pose on blank images
        cv2.imshow("Extracted Pose", opImg)

    # print all landmarks
    #print(results1.pose_landmarks)

        cv2.waitKey(1)
    # extract segmented mask
        mask = results.segmentation_mask
    # it returns true or false where the condition applies in the mask
        condition = np.stack(
            (results.segmentation_mask,) * 3, axis=-1) > 0.6

    # resize the background image to the same size of the original frame
        bg_image = cv2.resize(bg_image, (width, height))

    # combine frame and background image using the condition
        output_image = np.where(condition, frame, bg_image)

    # show outputs
    #cv2.imshow("mask", mask)
        white = [255,255,255]
        font = cv2.FONT_HERSHEY_SIMPLEX
        ratio_white = cv2.countNonZero(mask)/(frame.size/3)
        percent = ratio_white*100
        print(percent)
        if (percent<record['wo9of.png']+5) :
            cv2.putText(output_image,
                'wou9ouf',
                (50, 50),
                font, 1,
                (255,0,0),
                2,
                cv2.LINE_4)
        elif (percent <record['roukou3.png']+5) :
            cv2.putText(output_image,
                'Roukou3',
                (50, 50),
                font, 1,
                (255,0,0),
                2,
                cv2.LINE_4)
        else :
            cv2.putText(output_image,
                'Soujoud',
                (50, 50),
                font, 1,
                (255,0,0),
                2,
                cv2.LINE_4)
        cv2.imshow("Output", output_image)
        cv2.imshow("Frame", frame)
        cv2.imshow('mask', mask)
    #time.sleep(1)
        key = cv2.waitKey(1)
        if key == ord('q'):
            break

    # if 'd' key is pressed then change the background image
        elif key == ord('d'):

            if image_index != len(images)-1:
                image_index += 1
            else:
                image_index = 0
            bg_image = cv2.imread(image_path+'/'+images[image_index])


# release the capture object and close all active windows
    cap.release()
    cv2.destroyAllWindows()


# fixer les mesures.
from LaSAHW.image import capter
from LaSAHW.test import percent
position()
print ("position de wo9of")
ch="wo9of.png"
capter(ch)
pourcentage= percent(ch)
record={ch:pourcentage,}
print ("position de roukou3")
ch="roukou3.png"
capter(ch)
pourcentage= percent(ch)
record[ch]=pourcentage
print ("position de wo9of")
ch="soujoud.png"
capter(ch)
pourcentage= percent(ch)
record[ch]=pourcentage
print(record)
background_removal(record)
