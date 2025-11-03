import mediapipe as mp
import cv2
import numpy as np
from math import acos, degrees
import time
import joblib
import warnings

warnings.filterwarnings("ignore", category=UserWarning, module="google.protobuf")

pulgar = [1, 2, 4]
puntos_palma = [0, 1, 2, 5, 9, 13, 17]
bases = [6, 10, 14, 18]
puntas = [8, 12, 16, 20]

# Archivo del modelo final
modelo_knn = "modelo_final.pkl"

# Variables globales
prediccion_actual = ""
confianza_actual = 0.0

# Umbral de confianza
UMBRAL_CONFIANZA = 0.50  


def centroide(lista_coordenadas):
    coordenadas = np.array(lista_coordenadas)
    centroid = np.mean(coordenadas, axis=0)
    return int(centroid[0]), int(centroid[1])

def detectar_dedos_mejorado(hand_landmarks, width, height, label):
    """Detecta el estado (abierto/cerrado) de los 5 dedos"""
    try:
        coordinadas_pulgar = []
        coordenadas_palma = []
        coordenadas_bases = []
        coordenadas_puntas = []

        for i in pulgar:
            coordinadas_pulgar.append([int(hand_landmarks.landmark[i].x * width),
                                       int(hand_landmarks.landmark[i].y * height)])
        for i in puntos_palma:
            coordenadas_palma.append([int(hand_landmarks.landmark[i].x * width),
                                      int(hand_landmarks.landmark[i].y * height)])
        for i in bases:
            coordenadas_bases.append([int(hand_landmarks.landmark[i].x * width),
                                      int(hand_landmarks.landmark[i].y * height)])
        for i in puntas:
            coordenadas_puntas.append([int(hand_landmarks.landmark[i].x * width),
                                       int(hand_landmarks.landmark[i].y * height)])

        p1, p2, p3 = map(np.array, coordinadas_pulgar)
        l1, l2, l3 = np.linalg.norm(p2 - p3), np.linalg.norm(p1 - p3), np.linalg.norm(p1 - p2)
        angle = degrees(acos((l1**2 + l3**2 - l2**2) / (2 * l1 * l3)))
        dedo_pulgar = 1 if angle > 150 else 0

        nx, ny = centroide(coordenadas_palma)
        coordenadas_centroide = np.array([nx, ny])
        dis_centroid_puntas = np.linalg.norm(coordenadas_centroide - np.array(coordenadas_puntas), axis=1)
        dis_centroid_bases = np.linalg.norm(coordenadas_centroide - np.array(coordenadas_bases), axis=1)
        dedos = (dis_centroid_puntas > dis_centroid_bases).astype(int)

        return np.append(dedo_pulgar, dedos), (nx, ny)

    except Exception as e:
        print(f"Error en detectar_dedos_mejorado: {e}")
        return np.zeros(5, dtype=int), (0, 0)

def extraer_centroide_y_dedos(results, width, height):
    """Extrae centroide y estado de dedos para cada mano"""
    hands_data = []
    model_labels = []

    if results.multi_hand_landmarks:
        for idx, hand_landmarks in enumerate(results.multi_hand_landmarks):
            model_label = results.multi_handedness[idx].classification[0].label
            dedos, centroide_pos = detectar_dedos_mejorado(hand_landmarks, width, height, model_label)
            hands_data.append((centroide_pos, dedos))
            model_labels.append(model_label)

    data = {'Left': None, 'Right': None}

    if len(hands_data) == 0:
        return data
    elif len(hands_data) == 1:
        data[model_labels[0]] = hands_data[0]
    else:
        sorted_indices = sorted(range(2), key=lambda i: hands_data[i][0][0])
        left_idx, right_idx = sorted_indices
        data['Left'] = hands_data[left_idx]
        data['Right'] = hands_data[right_idx]

    return data

def crear_pizarron(trayectoria, nombre):
    """Crea una imagen con la trayectoria de la mano"""
    if trayectoria and len(trayectoria) > 0:
        puntos = np.array(trayectoria)
        x_min, y_min = np.min(puntos, axis=0)
        x_max, y_max = np.max(puntos, axis=0)
        ancho = max(x_max - x_min + 40, 50)
        alto = max(y_max - y_min + 40, 50)
        pizarron = 255 * np.ones((alto, ancho, 3), dtype=np.uint8)
        puntos_recentrados = [((x - x_min + 20), (y - y_min + 20)) for (x, y) in trayectoria]

        for i in range(1, len(puntos_recentrados)):
            cv2.line(pizarron, puntos_recentrados[i - 1], puntos_recentrados[i], (0, 0, 0), 2)

        for punto in puntos_recentrados:
            cv2.circle(pizarron, punto, 2, (255, 0, 0), -1)

        return cv2.resize(pizarron, (200, 200))
    else:
        pizarron = 255 * np.ones((200, 200, 3), dtype=np.uint8)
        cv2.putText(pizarron, "Sin datos", (50, 100), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (100, 100, 100), 1)
        return pizarron

def pizarron_a_vector_binario(pizarron, tamano_salida=(20, 20), umbral=128):
    gris = cv2.cvtColor(pizarron, cv2.COLOR_BGR2GRAY)
    gris_redim = cv2.resize(gris, tamano_salida)
    _, binaria = cv2.threshold(gris_redim, umbral, 255, cv2.THRESH_BINARY)
    return binaria.flatten(), binaria

def estandarizar_frames(frames, num_frames_deseado=30):
    if len(frames) == 0:
        return []
    if len(frames) == num_frames_deseado:
        return frames
    if len(frames) < num_frames_deseado:
        return frames + [frames[-1]] * (num_frames_deseado - len(frames))
    indices = np.linspace(0, len(frames) - 1, num_frames_deseado, dtype=int)
    return [frames[i] for i in indices]

def obtener_frames_medios(frames, num_frames=5):
    if len(frames) <= num_frames:
        return frames
    inicio = (len(frames) - num_frames) // 2
    return frames[inicio:inicio + num_frames]

def preparar_caracteristicas_inferencia(secuencia_dedos_left, secuencia_dedos_right,
                                        vector_binario_left, vector_binario_right):
    secuencia_left_flat = np.array(secuencia_dedos_left).flatten()
    secuencia_right_flat = np.array(secuencia_dedos_right).flatten()
    vector_left_norm = (vector_binario_left / 255.0).astype(np.float64)
    vector_right_norm = (vector_binario_right / 255.0).astype(np.float64)
    caracteristicas = np.hstack([secuencia_left_flat, secuencia_right_flat,
                                vector_left_norm, vector_right_norm])
    return caracteristicas.reshape(1, -1)

def cargar_modelo(nombre_archivo=modelo_knn):
    """Carga el modelo entrenado con soporte para distintos tipos"""
    try:
        modelo_cargado = joblib.load(nombre_archivo)
        print(f"Modelo cargado: {nombre_archivo}")
        print(f"Tipo de modelo: {modelo_cargado.get('nombre_modelo', 'Desconocido')}")
        print(f"Clases disponibles: {list(modelo_cargado['classes'])}")
        print(f"Umbral de confianza: {UMBRAL_CONFIANZA*100}%")
        return modelo_cargado
    except Exception as e:
        print(f"Error cargando modelo: {e}")
        return None

def determinar_prediccion_final(prediccion_encoded, confianza, label_encoder):
    if confianza >= UMBRAL_CONFIANZA:
        prediccion = label_encoder.inverse_transform([prediccion_encoded])[0]
        return prediccion, "RECONOCIDO", (0, 255, 0)
    else:
        return "Vuelve a intentarlo", "BAJA_CONFIANZA", (0, 165, 255)

def main():
    modelo_data = cargar_modelo()
    if modelo_data is None:
        return

    model = modelo_data['model']
    label_encoder = modelo_data['label_encoder']
    scaler = modelo_data.get('scaler', None)

    mp_drawing = mp.solutions.drawing_utils
    mp_drawing_styles = mp.solutions.drawing_styles
    mp_hands = mp.solutions.hands
    cap = cv2.VideoCapture(0)

    grabando = False
    frames_video = []
    centroides_trayectoria_left, centroides_trayectoria_right = [], []
    estados_dedos_trayectoria_left, estados_dedos_trayectoria_right = [], []
    mano_detectada_anteriormente = False
    ultima_deteccion = time.time()
    prediccion_actual = "Esperando..."
    confianza_actual = 0.0
    estado_prediccion = "INICIAL"
    color_prediccion = (255, 255, 255)

    print("\nSISTEMA DE INFERENCIA EN TIEMPO REAL")
    print(f"   Umbral de confianza: {UMBRAL_CONFIANZA*100}%")
    print("   La grabación comenzará automáticamente cuando detecte una mano")
    print("   Presione ESC para salir")

    with mp_hands.Hands(model_complexity=0, max_num_hands=2, min_detection_confidence=0.80, min_tracking_confidence = 0.70) as hands:
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            frame = cv2.flip(frame, 1)
            height, width, _ = frame.shape
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            results = hands.process(frame_rgb)
            mano_actualmente_detectada = results.multi_hand_landmarks is not None

            if not grabando and mano_actualmente_detectada and not mano_detectada_anteriormente:
                grabando = True
                frames_video = []
                centroides_trayectoria_left, centroides_trayectoria_right = [], []
                estados_dedos_trayectoria_left, estados_dedos_trayectoria_right = [], []
                ultima_deteccion = time.time()
                prediccion_actual = "Grabando..."
                estado_prediccion = "GRABANDO"
                color_prediccion = (0, 255, 255)
                print("¡Mano detectada! Comenzando grabación...")

            if grabando:
                if mano_actualmente_detectada:
                    ultima_deteccion = time.time()
                    frames_video.append(frame.copy())
                    data = extraer_centroide_y_dedos(results, width, height)

                    if data['Left']:
                        centroide_pos, dedos = data['Left']
                        centroides_trayectoria_left.append(centroide_pos)
                        estados_dedos_trayectoria_left.append(dedos)

                    if data['Right']:
                        centroide_pos, dedos = data['Right']
                        centroides_trayectoria_right.append(centroide_pos)
                        estados_dedos_trayectoria_right.append(dedos)

                    estado_texto = f"GRABANDO: {len(frames_video)} frames"
                    color_estado = (0, 255, 0)
                else:
                    if time.time() - ultima_deteccion > 1.0:
                        grabando = False
                        print(f"Grabación terminada. {len(frames_video)} frames capturados")

                        secuencia_dedos_array_left = np.zeros((5, 5), dtype=int)
                        secuencia_dedos_array_right = np.zeros((5, 5), dtype=int)
                        vector_binario_left = np.zeros(400, dtype=np.uint8)
                        vector_binario_right = np.zeros(400, dtype=np.uint8)

                        if len(centroides_trayectoria_left) > 0:
                            centroides_medios_left = obtener_frames_medios(
                                estandarizar_frames(centroides_trayectoria_left, 30), 5)
                            estados_medios_left = obtener_frames_medios(
                                estandarizar_frames(estados_dedos_trayectoria_left, 30), 5)
                            secuencia_dedos_array_left = np.array(estados_medios_left)
                            pizarra_left = crear_pizarron(centroides_medios_left, 'Left')
                            vector_binario_left, _ = pizarron_a_vector_binario(pizarra_left)

                        if len(centroides_trayectoria_right) > 0:
                            centroides_medios_right = obtener_frames_medios(
                                estandarizar_frames(centroides_trayectoria_right, 30), 5)
                            estados_medios_right = obtener_frames_medios(
                                estandarizar_frames(estados_dedos_trayectoria_right, 30), 5)
                            secuencia_dedos_array_right = np.array(estados_medios_right)
                            pizarra_right = crear_pizarron(centroides_medios_right, 'Right')
                            vector_binario_right, _ = pizarron_a_vector_binario(pizarra_right)

                        if len(centroides_trayectoria_left) > 0 or len(centroides_trayectoria_right) > 0:
                            try:
                                X_inferencia = preparar_caracteristicas_inferencia(
                                    secuencia_dedos_array_left, secuencia_dedos_array_right,
                                    vector_binario_left, vector_binario_right
                                )
                                if scaler is not None:
                                    X_inferencia = scaler.transform(X_inferencia)

                                prediccion_encoded = model.predict(X_inferencia)[0]
                                probabilidades = model.predict_proba(X_inferencia)[0]
                                confianza = np.max(probabilidades)

                                prediccion, estado, color_pred = determinar_prediccion_final(
                                    prediccion_encoded, confianza, label_encoder)

                                prediccion_actual = prediccion
                                confianza_actual = confianza
                                estado_prediccion = estado
                                color_prediccion = color_pred

                                print(f"Predicción: {prediccion} -- Confianza: {confianza:.2f}")

                            except Exception as e:
                                print(f"Error en prediccion: {e}")
                                prediccion_actual = "Error en prediccion"
                                confianza_actual = 0.0
                                estado_prediccion = "ERROR"
                                color_prediccion = (0, 0, 255)
                        else:
                            prediccion_actual = "Sin datos"
                            confianza_actual = 0.0
                            estado_prediccion = "SIN DATOS"
                            color_prediccion = (128, 128, 128)

                        estado_texto = "PROCESADO"
                        color_estado = (255, 0, 0)
                    else:
                        estado_texto = f"GRABANDO: {len(frames_video)} frames (sin mano)"
                        color_estado = (0, 165, 255)
            else:
                if mano_actualmente_detectada:
                    estado_texto = "LISTO - Mueve la mano para comenzar"
                    color_estado = (0, 255, 255)
                else:
                    estado_texto = "ESPERANDO MANO..."
                    color_estado = (0, 0, 255)

            if results.multi_hand_landmarks:
                for hand_landmarks in results.multi_hand_landmarks:
                    mp_drawing.draw_landmarks(
                        frame, hand_landmarks, mp_hands.HAND_CONNECTIONS,
                        mp_drawing_styles.get_default_hand_landmarks_style(),
                        mp_drawing_styles.get_default_hand_connections_style()
                    )

            cv2.putText(frame, estado_texto, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, color_estado, 2)
            cv2.putText(frame, f"Frames: {len(frames_video)}", (10, 60),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1)
            cv2.putText(frame, f"Prediccion: {prediccion_actual}", (10, 90),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, color_prediccion, 2)
            cv2.putText(frame, f"Confianza: {confianza_actual*100}%", (10, 120),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 0), 2)
            cv2.putText(frame, f"Estado: {estado_prediccion}", (10, 150),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, color_prediccion, 2)

            cv2.imshow('Inferencia en Tiempo Real - Deteccion de Gestos', frame)

            key = cv2.waitKey(1) & 0xFF
            if key == 27:  # ESC
                break

            mano_detectada_anteriormente = mano_actualmente_detectada

    print("Programa terminado")
    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
