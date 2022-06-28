import cv2
import numpy as np
import mediapipe as mp
import matplotlib.pyplot as plt
def capter (ch):
#Créer une instance de la classe VideoCapture
#Spécifiez 0 lors de l'utilisation de la webcam intégrée
    cap = cv2.VideoCapture(0)

    #S'il ne peut pas être lu normalement
    if cap.isOpened() is False:
        print("IO Error")
    #Lorsqu'il est chargé normalement
    else:
        #méthode de lecture
        #Valeur de retour 1 ret:L'image du cadre a-t-elle été chargée?/Valeur de retour 2 frame:Tableau d'images (ndarray)
        ret, frame = cap.read()
        image_path = "./images/"
        if (ret):
            #méthode imwrite
            #Argument 1: chemin de l'image Argument 2: objet ndarray représentant l'image
            #L'extension du chemin de l'image est.jpg et.png peut être utilisé.
            cv2.imwrite(image_path + ch, frame)
        else:
            print("Read Error")

    cap.release() #Quittez l'appareil photo
