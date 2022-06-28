import cv2

import mediapipe as mp
import numpy as np
def position():
    mp_drawing = mp.solutions.drawing_utils
    mp_pose = mp.solutions.pose

# VIDEO FEED
    cap = cv2.VideoCapture(0)
    while cap.isOpened():
        ret, frame = cap.read()
        cv2.imshow('Mediapipe Feed', frame)

        if cv2.waitKey(10) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()
    cap = cv2.VideoCapture(0)
## Setup mediapipe instance
    with mp_pose.Pose(min_detection_confidence=0.5, min_tracking_confidence=0.5) as pose:
        while cap.isOpened():
            ret, frame = cap.read()
            cv2.rectangle(frame, (150,50), (450, 480), (255, 0, 255), 2)
        # Recolor image to RGB
            image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            image.flags.writeable = False
        # Make detection
            results = pose.process(image)

        # Recolor back to BGR
            image.flags.writeable = True
            image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)

        # Extract landmarks
            try:
                landmarks = results.pose_landmarks.landmark
            #print(landmarks)
                nose_x = [landmarks[mp_pose.PoseLandmark.NOSE.value].x,landmarks[mp_pose.PoseLandmark.NOSE.value].y]
                print("nose_x",nose_x[0],"nose_y",nose_x[1])
                condition = (nose_x[0] < 0.6) & (nose_x[0] > 0.4)
                font = cv2.FONT_HERSHEY_SIMPLEX
                print(condition)

                if condition == True:
                    cv2.putText(image,
                            'Bonne position',
                            (50, 50),
                            font, 1,
                            (255, 0, 0),
                            2,
                            cv2.LINE_4)
                else:
                    cv2.putText(image,
                            'Mauvaise position',
                            (50, 50),
                            font, 1,
                            (255, 0, 0),
                            2,
                            cv2.LINE_4)
            except:
                pass

        # Render detections
            mp_drawing.draw_landmarks(image, results.pose_landmarks, mp_pose.POSE_CONNECTIONS,
                                  mp_drawing.DrawingSpec(color=(245, 117, 66), thickness=2, circle_radius=2),
                                  mp_drawing.DrawingSpec(color=(245, 66, 230), thickness=2, circle_radius=2)
                                  )

            cv2.imshow('Mediapipe Feed', image)

            if cv2.waitKey(10) & 0xFF == ord('q'):
                break

        cap.release()
        cv2.destroyAllWindows()
    len(landmarks)
