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

# Diccionario para almacenar el estado de los dedos
estado_manos = {"derecha": [0, 0, 0, 0, 0], "izquierda": [0, 0, 0, 0, 0]}

with mp_hands.Hands(model_complexity=1, max_num_hands=2) as hands:
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        frame = cv2.flip(frame, 1)
        height, width, _ = frame.shape
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = hands.process(frame_rgb)

        # Reiniciar el estado de las manos
        estado_manos = {"derecha": [0, 0, 0, 0, 0], "izquierda": [0, 0, 0, 0, 0]}

        if results.multi_hand_landmarks:
            for hand_idx, hand_landmarks in enumerate(results.multi_hand_landmarks):
                # Determinar si es mano izquierda o derecha
                hand_label = results.multi_handedness[hand_idx].classification[0].label

                # Convertir a español y verificar que sea "derecha" o "izquierda"
                if hand_label == "Right":
                    mano_actual = "derecha"
                else:
                    mano_actual = "izquierda"

                # Dibujar los puntos y conexiones de la mano
                mp_drawing.draw_landmarks(
                    frame,
                    hand_landmarks,
                    mp_hands.HAND_CONNECTIONS,
                    mp_drawing_styles.get_default_hand_landmarks_style(),
                    mp_drawing_styles.get_default_hand_connections_style()
                )

                # Reiniciar listas de coordenadas para cada mano
                coordinadas_pulgar = []
                coordenadas_palma = []
                coordenadas_puntas = []
                coordenadas_bases = []

                # Obtener coordenadas del pulgar
                for i in pulgar:
                    x = int(hand_landmarks.landmark[i].x * width)
                    y = int(hand_landmarks.landmark[i].y * height)
                    coordinadas_pulgar.append([x, y])

                # Obtener coordenadas de la palma
                for i in puntos_palma:
                    x = int(hand_landmarks.landmark[i].x * width)
                    y = int(hand_landmarks.landmark[i].y * height)
                    coordenadas_palma.append([x, y])

                # Obtener coordenadas de las bases de los dedos
                for i in bases:
                    x = int(hand_landmarks.landmark[i].x * width)
                    y = int(hand_landmarks.landmark[i].y * height)
                    coordenadas_bases.append([x, y])

                # Obtener coordenadas de las puntas de los dedos
                for i in puntas:
                    x = int(hand_landmarks.landmark[i].x * width)
                    y = int(hand_landmarks.landmark[i].y * height)
                    coordenadas_puntas.append([x, y])

                # Detectar estado del pulgar (ley de los cosenos)
                p1 = np.array(coordinadas_pulgar[0])
                p2 = np.array(coordinadas_pulgar[1])
                p3 = np.array(coordinadas_pulgar[2])

                l1 = np.linalg.norm(p2 - p3)
                l2 = np.linalg.norm(p1 - p3)
                l3 = np.linalg.norm(p1 - p2)

                # Calcular el ángulo
                angle = degrees(acos((l1 ** 2 + l3 ** 2 - l2 ** 2) / (2 * l1 * l3)))
                dedo_pulgar = 0
                if angle > 150:
                    dedo_pulgar = 1

                # Detectar estado de los otros dedos (índice, medio, anular y meñique)
                nx, ny = centroide(coordenadas_palma)

                # Visualizar el centroide en la imagen
                cv2.circle(frame, (nx, ny), 3, (0, 255, 0), 2)

                coordenadas_centroide = np.array([nx, ny])
                coordenadas_bases_array = np.array(coordenadas_bases)
                coordenadas_puntas_array = np.array(coordenadas_puntas)

                # Calcular las distancias
                dis_centroid_puntas = np.linalg.norm(coordenadas_centroide - coordenadas_puntas_array, axis=1)
                dis_centroid_bases = np.linalg.norm(coordenadas_centroide - coordenadas_bases_array, axis=1)
                diferencia = dis_centroid_puntas - dis_centroid_bases

                dedos = (diferencia > 0).astype(int)
                dedos = np.append(dedo_pulgar, dedos)

                # Actualizar el diccionario con el estado de los dedos
                estado_manos[mano_actual] = dedos.tolist()

        # Mostrar el estado de las manos en la consola
        print("Estado de manos:", estado_manos)

        # También puedes mostrar el estado en la ventana de video si lo deseas
        cv2.putText(frame, f"Derecha: {estado_manos['derecha']}", (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        cv2.putText(frame, f"Izquierda: {estado_manos['izquierda']}", (10, 60),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)

        cv2.imshow('Frame', frame)
        if cv2.waitKey(1) & 0xFF == 27:
            break

cap.release()
cv2.destroyAllWindows()