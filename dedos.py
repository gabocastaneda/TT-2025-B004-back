import mediapipe as mp
import cv2
import numpy as np
from math import acos, degrees

def centroide(lista_coordenadas):
    coordenadas = np.array(lista_coordenadas)
    centroid = np.mean(coordenadas, axis=0)
    centroid = int(centroid[0]), int(centroid[1])
    return centroid

mp_drawing = mp.solutions.drawing_utils
mp_drawing_styles = mp.solutions.drawing_styles
mp_hands = mp.solutions.hands

cap = cv2.VideoCapture(0)


# Pulgar 
pulgar = [1, 2, 4]
# indice, medio, anular y meñique
puntos_palma = [0, 1, 2, 5, 9, 13, 17]
bases = [6, 10, 14, 18]
puntas = [8, 12, 16, 20]

with mp_hands.Hands(model_complexity=1, max_num_hands=1) as hands:
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        frame = cv2.flip(frame, 1)
        height, width, _ = frame.shape
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = hands.process(frame_rgb)

        if results.multi_hand_landmarks:
            coordinadas_pulgar = []
            coordenadas_palma = []
            coordenadas_puntas = []
            coordenadas_bases = []
            for handLms in results.multi_hand_landmarks:
                # unicamente dibujamos los 21 puntos de la mano
                mp_drawing.draw_landmarks(frame, handLms, mp_hands.HAND_CONNECTIONS, mp_drawing_styles.get_default_hand_landmarks_style(), mp_drawing_styles.get_default_hand_connections_style())

                for i in pulgar:
                    x = int(handLms.landmark[i].x * width)
                    y = int(handLms.landmark[i].y * height)
                    coordinadas_pulgar.append([x, y])

                for i in puntos_palma:
                    x = int(handLms.landmark[i].x * width)
                    y = int(handLms.landmark[i].y * height)
                    coordenadas_palma.append([x, y])

                for i in bases:
                    x = int(handLms.landmark[i].x * width)
                    y = int(handLms.landmark[i].y * height)
                    coordenadas_bases.append([x, y])

                for i in puntas:
                    x = int(handLms.landmark[i].x * width)
                    y = int(handLms.landmark[i].y * height)
                    coordenadas_puntas.append([x, y])

            # pulgar (se utiliza la ley de los cosenos para saber si esta doblado o extendido)
            p1 = np.array(coordinadas_pulgar[0])
            p2 = np.array(coordinadas_pulgar[1])
            p3 = np.array(coordinadas_pulgar[2])

            l1 = np.linalg.norm(p2 - p3)
            l2 = np.linalg.norm(p1 - p3)
            l3 = np.linalg.norm(p1 - p2)

            # Calcular el angulo
            angle = degrees(acos((l1**2 + l3**2 - l2**2) / (2 * l1 * l3)))
            dedo_pulgar = 0
            if angle > 150:                
                dedo_pulgar = 1

            # indice, medio, anular y meñique
            nx, ny = centroide(coordenadas_palma)
            # visualizar el centroide en la imagen
            cv2.circle(frame, (nx, ny), 3, (0, 255, 0), 2)

            coordenadas_centroide = np.array([nx, ny])
            coordenadas_bases = np.array(coordenadas_bases)
            coordenadas_puntas = np.array(coordenadas_puntas)

            # Calcular las distancias
            dis_centroid_puntas = np.linalg.norm(coordenadas_centroide - coordenadas_puntas, axis=1) # axis=1 para que nos de las 4 distancias, sino solo entrega 1
            dis_centroid_bases = np.linalg.norm(coordenadas_centroide - coordenadas_bases, axis=1)
            diferencia = dis_centroid_puntas - dis_centroid_bases

            dedos = (diferencia > 0).astype(int)
            dedos = np.append(dedo_pulgar, dedos)
            print(dedos)

        cv2.imshow('Frame', frame)
        if cv2.waitKey(1) & 0xFF == 27:
            break

cap.release()
cv2.destroyAllWindows()