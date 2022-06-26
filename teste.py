import cv2
<<<<<<< HEAD
import mediapipe as mp
import numpy as np
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
            shoulder_x = [landmarks[mp_pose.PoseLandmark.LEFT_SHOULDER.value].x,landmarks[mp_pose.PoseLandmark.LEFT_SHOULDER.value].y]
            print("left_x",shoulder_x[0],"left_y",shoulder_x[1])
            shoulder_y = [landmarks[mp_pose.PoseLandmark.RIGHT_SHOULDER.value].x,
                          landmarks[mp_pose.PoseLandmark.RIGHT_SHOULDER.value].y]
            condition = (shoulder_x[0] < 0.68) & (shoulder_y[0] < 0.26)
            font = cv2.FONT_HERSHEY_SIMPLEX
            print(condition)
            print("right_x", shoulder_y[0], "right_y", shoulder_y[1])

            if condition == True:
                cv2.putText(image,
                            'mauvaise position',
                            (50, 50),
                            font, 1,
                            (255, 0, 0),
                            2,
                            cv2.LINE_4)
            else:
                cv2.putText(image,
                            'bonne position',
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
=======
import numpy as np

# Let's load a simple image with 3 black squares
image = cv2.imread('images/image.png')
cv2.waitKey(0)

# Grayscale
gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

# Find Canny edges
edged = cv2.Canny(gray, 30, 200)
cv2.waitKey(0)

# Finding Contours
# Use a copy of the image e.g. edged.copy()
# since findContours alters the image
contours, hierarchy = cv2.findContours(edged,
	cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)

cv2.imshow('Canny Edges After Contouring', edged)
cv2.waitKey(0)

print("Number of Contours found = " + str(len(contours)))

# Draw all contours
# -1 signifies drawing all contours
cv2.drawContours(image, contours, -1, (0, 255, 0), 3)

cv2.imshow('Contours', image)
cv2.waitKey(0)
cv2.destroyAllWindows()
cap = cv2.VideoCapture(0)
while cap.isOpened():
    _, frame = cap.read()
    # flip the frame to horizontal direction
    frame = cv2.flip(frame, 1)
    cv2.drawContours(image, contours, -1, (0, 255, 0), 3)
    cv2.rectangle(frame, (200, 150), (450, 480), (255, 0, 255), 2)
    cv2.imshow('detect', frame)
    cv2.waitKey(1)
>>>>>>> 7f55e691af851e7f7db194e189fe025e32bb7efc


