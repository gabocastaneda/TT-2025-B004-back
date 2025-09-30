import mediapipe as mp
import cv2
import numpy as np
import time

mp_drawing = mp.solutions.drawing_utils
mp_drawing_styles = mp.solutions.drawing_styles
mp_hands = mp.solutions.hands

puntos_palma = [0, 1, 2, 5, 9, 13, 17]
trayecto = []
grabando = False
inicio_grabacion = 0

def centroide(lista_coordenadas):
    coordenadas = np.array(lista_coordenadas)
    centroid = np.mean(coordenadas, axis=0)
    centroid = int(centroid[0]), int(centroid[1])
    return centroid

def crear_pizarron(trayectoria, nombre):
    if trayectoria:
        xs = [p[0] for p in trayectoria]
        ys = [p[1] for p in trayectoria]

        x_min, x_max = min(xs), max(xs)
        y_min, y_max = min(ys), max(ys)

        ancho = x_max - x_min + 20
        alto = y_max - y_min + 20

        pizarron = 255 * np.ones((alto, ancho, 3), dtype=np.uint8)
        puntos_recentrados = [((x - x_min + 10), (y - y_min + 10)) for (x, y) in trayectoria]

        for i in range(1, len(puntos_recentrados)):
            pt1 = puntos_recentrados[i - 1]
            pt2 = puntos_recentrados[i]
            cv2.line(pizarron, pt1, pt2, (0, 0, 0), 2)

        pizarron = cv2.resize(pizarron, (150,150)) 
        print(f"{nombre}: {len(puntos_recentrados)} puntos dibujados.")
    else:
        # Pizarrón vacío si no hubo trayectoria
        pizarron = 255 * np.ones((150, 150, 3), dtype=np.uint8)
        cv2.putText(pizarron, "Esperando...", (20, 75), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (100, 100, 100), 1)
        print(f"{nombre}: sin trayectoria.")
    return pizarron

cap = cv2.VideoCapture(0)

print("🎥 Iniciando cámara...")
print("⏰ Tienes 3 segundos para subir la mano...")
time.sleep(3)
print("🟢 ¡Comenzando grabación por 5 segundos!")
grabando = True
inicio_grabacion = time.time()

with mp_hands.Hands(model_complexity=1, max_num_hands=1) as hands:
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        frame = cv2.flip(frame, 1)
        height, width, _ = frame.shape
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = hands.process(frame_rgb)

        # Mostrar tiempo restante
        tiempo_transcurrido = time.time() - inicio_grabacion
        tiempo_restante = max(0, 5 - tiempo_transcurrido)
        
        if grabando:
            estado_texto = f"GRABANDO: {tiempo_restante:.1f}s"
            color_estado = (0, 255, 0)
        else:
            estado_texto = "GRABACION TERMINADA"
            color_estado = (0, 0, 255)
        
        cv2.putText(frame, estado_texto, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, color_estado, 2)

        # Verificar si ha pasado el tiempo de grabación
        if grabando and tiempo_transcurrido >= 5:
            grabando = False
            print("🔴 Grabación terminada después de 5 segundos")
            print(f"📊 Puntos capturados: {len(trayecto)}")

        if results.multi_hand_landmarks:
            coordenadas_palma = []
            
            for handLms in results.multi_hand_landmarks:
                # unicamente dibujamos los 21 puntos de la mano
                mp_drawing.draw_landmarks(frame, handLms, mp_hands.HAND_CONNECTIONS, mp_drawing_styles.get_default_hand_landmarks_style(), mp_drawing_styles.get_default_hand_connections_style())

                for i in puntos_palma:
                    x = int(handLms.landmark[i].x * width)
                    y = int(handLms.landmark[i].y * height)
                    coordenadas_palma.append([x, y])

            # calcular el centroide
            nx, ny = centroide(coordenadas_palma)
            
            # visualizar el centroide en la imagen
            if grabando:
                cv2.circle(frame, (nx, ny), 3, (0, 255, 0), 2)  # Verde cuando graba
                coordenadas_centroide = np.array([nx, ny])
                trayecto.append(coordenadas_centroide)
            else:
                cv2.circle(frame, (nx, ny), 3, (255, 0, 0), 2)  # Azul cuando no graba

        # Actualizar pizarra siempre
        pizarra = crear_pizarron(trayecto, 'derecha')
        cv2.imshow('pizarra', pizarra)
        cv2.imshow('Frame', frame)
        
        if cv2.waitKey(1) & 0xFF == 27:
            break

print("📹 Programa terminado")
cap.release()
cv2.destroyAllWindows()