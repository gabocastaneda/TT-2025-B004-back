import mediapipe as mp
import cv2
import numpy as np
from math import acos, degrees
import time
import os
import pandas as pd


def centroide(lista_coordenadas):
    coordenadas = np.array(lista_coordenadas)
    centroid = np.mean(coordenadas, axis=0)
    centroid = int(centroid[0]), int(centroid[1])
    return centroid


# Configuración de dedos
pulgar = [1, 2, 4]
puntos_palma = [0, 1, 2, 5, 9, 13, 17]
bases = [6, 10, 14, 18]
puntas = [8, 12, 16, 20]

# Variables globales para grabación y datos
frames_video = []
grabando = False
mano_detectada_anteriormente = False
archivo_csv = "dataset_dedos.csv"
clase_actual = "Si"

# Diccionario para almacenar el estado de los dedos en tiempo real
estado_manos = {"derecha": [0, 0, 0, 0, 0], "izquierda": [0, 0, 0, 0, 0]}

# Variables para trayectorias
centroides_trayectoria_left = []
centroides_trayectoria_right = []
estados_dedos_trayectoria_left = []
estados_dedos_trayectoria_right = []


def detectar_dedos_mejorado(hand_landmarks, width, height, label):
    """Detecta el estado de los 5 dedos (0 = cerrado, 1 = abierto) para ambas manos"""
    coordinadas_pulgar = []
    coordenadas_palma = []
    coordenadas_puntas = []
    coordenadas_bases = []

    try:
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

        coordenadas_centroide = np.array([nx, ny])
        coordenadas_bases_array = np.array(coordenadas_bases)
        coordenadas_puntas_array = np.array(coordenadas_puntas)

        # Calcular las distancias
        dis_centroid_puntas = np.linalg.norm(coordenadas_centroide - coordenadas_puntas_array, axis=1)
        dis_centroid_bases = np.linalg.norm(coordenadas_centroide - coordenadas_bases_array, axis=1)
        diferencia = dis_centroid_puntas - dis_centroid_bases

        dedos = (diferencia > 0).astype(int)
        dedos_completo = np.append(dedo_pulgar, dedos)

        return dedos_completo, (nx, ny)

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
        label = model_labels[0]
        data[label] = hands_data[0]

    else:
        # Sort by x coordinate for consistent left/right assignment
        sorted_indices = sorted(range(2), key=lambda i: hands_data[i][0][0])
        left_idx = sorted_indices[0]
        right_idx = sorted_indices[1]

        # Recompute dedos using position-based label
        dedos_left, centroide_left = detectar_dedos_mejorado(results.multi_hand_landmarks[left_idx], width, height,
                                                             'Left')
        data['Left'] = (centroide_left, dedos_left)

        dedos_right, centroide_right = detectar_dedos_mejorado(results.multi_hand_landmarks[right_idx], width, height,
                                                               'Right')
        data['Right'] = (centroide_right, dedos_right)

    return data


def estandarizar_frames(frames, num_frames_deseado=30):
    """Estandariza los frames a un número fijo"""
    if len(frames) == 0:
        return []

    if len(frames) == num_frames_deseado:
        return frames

    if len(frames) < num_frames_deseado:
        frames_estandarizados = frames.copy()
        while len(frames_estandarizados) < num_frames_deseado:
            frames_estandarizados.append(frames[-1])
        return frames_estandarizados

    indices = np.linspace(0, len(frames) - 1, num_frames_deseado, dtype=int)
    return [frames[i] for i in indices]


def obtener_frames_medios(frames, num_frames=5):
    """Obtiene los frames del medio de la secuencia"""
    if len(frames) <= num_frames:
        return frames

    inicio = (len(frames) - num_frames) // 2
    return frames[inicio:inicio + num_frames]


def crear_pizarron(trayectoria, nombre):
    """Crea una imagen de pizarrón con la trayectoria del centroide"""
    if trayectoria:
        puntos = np.array(trayectoria)

        x_min, y_min = np.min(puntos, axis=0)
        x_max, y_max = np.max(puntos, axis=0)

        ancho = max(x_max - x_min + 40, 50)
        alto = max(y_max - y_min + 40, 50)

        pizarron = 255 * np.ones((alto, ancho, 3), dtype=np.uint8)

        puntos_recentrados = [((x - x_min + 20), (y - y_min + 20)) for (x, y) in trayectoria]

        for i in range(1, len(puntos_recentrados)):
            pt1 = puntos_recentrados[i - 1]
            pt2 = puntos_recentrados[i]
            cv2.line(pizarron, pt1, pt2, (0, 0, 0), 2)

        for punto in puntos_recentrados:
            cv2.circle(pizarron, punto, 2, (255, 0, 0), -1)

        pizarron = cv2.resize(pizarron, (200, 200))
        print(f"{nombre}: {len(puntos_recentrados)} puntos dibujados.")
    else:
        pizarron = 255 * np.ones((200, 200, 3), dtype=np.uint8)
        cv2.putText(pizarron, "Sin datos", (50, 100), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (100, 100, 100), 1)
        print(f"{nombre}: sin trayectoria.")

    return pizarron


def pizarron_a_vector_binario(pizarron, tamano_salida=(20, 20), umbral=128):
    """Convierte el pizarrón a un vector binario (0 y 255)"""
    gris = cv2.cvtColor(pizarron, cv2.COLOR_BGR2GRAY)
    gris_redim = cv2.resize(gris, tamano_salida)
    _, binaria = cv2.threshold(gris_redim, umbral, 255, cv2.THRESH_BINARY)
    vector_binario = binaria.flatten()

    return vector_binario, binaria


def generar_encabezados(tamano_vector=400):
    """Genera encabezados descriptivos para las columnas"""
    encabezados = ["clase"]

    encabezados.append("secuencia_dedos_centrales_left")
    encabezados.append("secuencia_dedos_centrales_right")

    for i in range(tamano_vector):
        fila = i // 20
        columna = i % 20
        encabezados.append(f"pixel_left_{fila:02d}_{columna:02d}")

    for i in range(tamano_vector):
        fila = i // 20
        columna = i % 20
        encabezados.append(f"pixel_right_{fila:02d}_{columna:02d}")

    return encabezados


def inicializar_csv():
    """Inicializa el archivo CSV con los encabezados"""
    if not os.path.exists(archivo_csv):
        encabezados = generar_encabezados()
        df = pd.DataFrame(columns=encabezados)
        df.to_csv(archivo_csv, index=False)
        print(f"Archivo CSV creado: {archivo_csv}")
        print(
            f"Estructura: Clase + sec_dedos_left + sec_dedos_right + 400 pixels_left + 400 pixels_right = 803 columnas")
    else:
        print(f"Archivo CSV existente: {archivo_csv}")
        print(f"Se agregarán nuevos registros al final")


def guardar_en_csv(clase, secuencia_dedos_centrales_left, secuencia_dedos_centrales_right, vector_binario_left,
                   vector_binario_right):
    """Guarda un nuevo registro en el CSV"""
    vector_normalizado_left = (vector_binario_left / 255.0).astype(np.float64)
    vector_normalizado_right = (vector_binario_right / 255.0).astype(np.float64)

    secuencia_dedos_str_left = str(secuencia_dedos_centrales_left.tolist())
    secuencia_dedos_str_right = str(secuencia_dedos_centrales_right.tolist())

    datos_completos = [clase, secuencia_dedos_str_left,
                       secuencia_dedos_str_right] + vector_normalizado_left.tolist() + vector_normalizado_right.tolist()

    if os.path.exists(archivo_csv):
        df = pd.read_csv(archivo_csv)
    else:
        encabezados = generar_encabezados()
        df = pd.DataFrame(columns=encabezados)

    nueva_fila = pd.DataFrame([datos_completos], columns=df.columns)
    df = pd.concat([df, nueva_fila], ignore_index=True)

    df.to_csv(archivo_csv, index=False)

    valores_unicos_left = np.unique(vector_binario_left)
    valores_unicos_right = np.unique(vector_binario_right)
    print(f"Valores únicos en vector left: {valores_unicos_left}")
    print(f"Valores únicos en vector right: {valores_unicos_right}")

    return datos_completos


def cambiar_clase():
    """Permite cambiar la clase actual mediante input"""
    global clase_actual
    print(f"\nCLASE ACTUAL: '{clase_actual}'")
    nueva_clase = input("   Ingrese nueva clase (o Enter para mantener actual): ").strip()
    if nueva_clase:
        clase_actual = nueva_clase
        print(f"   ✅ Nueva clase establecida: '{clase_actual}'")
    else:
        print(f"   🔄 Manteniendo clase: '{clase_actual}'")
    return clase_actual


# Inicializar MediaPipe
mp_drawing = mp.solutions.drawing_utils
mp_drawing_styles = mp.solutions.drawing_styles
mp_hands = mp.solutions.hands

cap = cv2.VideoCapture(0)

# Inicializar sistema
inicializar_csv()

print("\nCONTROLES:")
print("   C - Cambiar clase actual")
print("   G - Guardar gesto actual en CSV")
print("   ESC - Salir del programa")
print(f"\nCLASE INICIAL: '{clase_actual}'")

# Variables para visualización
pizarra_left = crear_pizarron([], 'Left Esperando')
pizarra_right = crear_pizarron([], 'Right Esperando')
vector_left = np.zeros(400, dtype=np.uint8)
vector_right = np.zeros(400, dtype=np.uint8)
matriz_left = np.zeros((20, 20), dtype=np.uint8)
matriz_right = np.zeros((20, 20), dtype=np.uint8)

with mp_hands.Hands(model_complexity=1, max_num_hands=2, min_detection_confidence=0.80) as hands:
    ultima_deteccion = time.time()

    while True:
        ret, frame = cap.read()
        if not ret:
            break
        frame = cv2.flip(frame, 1)
        height, width, _ = frame.shape
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = hands.process(frame_rgb)

        # Reiniciar el estado de las manos para este frame
        estado_manos = {"derecha": [0, 0, 0, 0, 0], "izquierda": [0, 0, 0, 0, 0]}

        mano_actualmente_detectada = results.multi_hand_landmarks is not None

        # Iniciar grabación cuando aparece una mano
        if not grabando and mano_actualmente_detectada and not mano_detectada_anteriormente:
            grabando = True
            frames_video = []
            centroides_trayectoria_left = []
            centroides_trayectoria_right = []
            estados_dedos_trayectoria_left = []
            estados_dedos_trayectoria_right = []
            ultima_deteccion = time.time()
            print("¡Mano detectada! Comenzando grabación...")

        # Durante la grabación
        if grabando:
            if mano_actualmente_detectada:
                ultima_deteccion = time.time()
                frames_video.append(frame.copy())

                # Extraer datos de ambas manos
                data = extraer_centroide_y_dedos(results, width, height)

                # Guardar datos de cada mano detectada
                if data['Left'] is not None:
                    centroide_pos, dedos = data['Left']
                    centroides_trayectoria_left.append(centroide_pos)
                    estados_dedos_trayectoria_left.append(dedos)
                    # Actualizar estado en tiempo real
                    estado_manos["izquierda"] = dedos.tolist()

                if data['Right'] is not None:
                    centroide_pos, dedos = data['Right']
                    centroides_trayectoria_right.append(centroide_pos)
                    estados_dedos_trayectoria_right.append(dedos)
                    # Actualizar estado en tiempo real
                    estado_manos["derecha"] = dedos.tolist()

                estado_texto = f"GRABANDO: {len(frames_video)} frames"
                color_estado = (0, 255, 0)
            else:
                # Si no hay mano, verificar timeout
                if time.time() - ultima_deteccion > 1.0:
                    grabando = False
                    print(f"Grabación terminada. {len(frames_video)} frames capturados")

                    # Procesar mano izquierda
                    if len(centroides_trayectoria_left) > 0:
                        centroides_30_left = estandarizar_frames(centroides_trayectoria_left, 30)
                        estados_dedos_30_left = estandarizar_frames(estados_dedos_trayectoria_left, 30)
                        centroides_medios_left = obtener_frames_medios(centroides_30_left, 5)
                        estados_dedos_medios_left = obtener_frames_medios(estados_dedos_30_left, 5)
                        secuencia_dedos_array_left = np.array(estados_dedos_medios_left)
                        pizarra_left = crear_pizarron(centroides_medios_left, 'Trayectoria Media Left')
                        vector_binario_left, matriz_binaria_left = pizarron_a_vector_binario(pizarra_left)
                        vector_left = vector_binario_left
                        matriz_left = matriz_binaria_left
                    else:
                        secuencia_dedos_array_left = np.zeros((5, 5), dtype=int)
                        vector_binario_left = np.zeros(400, dtype=np.uint8)
                        pizarra_left = crear_pizarron([], 'Left Sin Datos')

                    # Procesar mano derecha
                    if len(centroides_trayectoria_right) > 0:
                        centroides_30_right = estandarizar_frames(centroides_trayectoria_right, 30)
                        estados_dedos_30_right = estandarizar_frames(estados_dedos_trayectoria_right, 30)
                        centroides_medios_right = obtener_frames_medios(centroides_30_right, 5)
                        estados_dedos_medios_right = obtener_frames_medios(estados_dedos_30_right, 5)
                        secuencia_dedos_array_right = np.array(estados_dedos_medios_right)
                        pizarra_right = crear_pizarron(centroides_medios_right, 'Trayectoria Media Right')
                        vector_binario_right, matriz_binaria_right = pizarron_a_vector_binario(pizarra_right)
                        vector_right = vector_binario_right
                        matriz_right = matriz_binaria_right
                    else:
                        secuencia_dedos_array_right = np.zeros((5, 5), dtype=int)
                        vector_binario_right = np.zeros(400, dtype=np.uint8)
                        pizarra_right = crear_pizarron([], 'Right Sin Datos')

                    # Guardar automáticamente si hay datos
                    if len(centroides_trayectoria_left) > 0 or len(centroides_trayectoria_right) > 0:
                        print("Guardando automáticamente...")
                        guardar_en_csv(clase_actual, secuencia_dedos_array_left, secuencia_dedos_array_right,
                                    vector_binario_left, vector_binario_right)

                        print(f"\nGesto guardado en CSV:")
                        print(f"   Clase: '{clase_actual}'")
                        print(f"   Secuencia dedos left: {secuencia_dedos_array_left.shape}")
                        print(f"   Secuencia dedos right: {secuencia_dedos_array_right.shape}")
                        print(f"   Puntos trayectoria left: {len(centroides_trayectoria_left)}")
                        print(f"   Puntos trayectoria right: {len(centroides_trayectoria_right)}")

                        if len(centroides_trayectoria_left) > 0:
                            cv2.imshow('Imagen Binaria Left', matriz_left)
                        if len(centroides_trayectoria_right) > 0:
                            cv2.imshow('Imagen Binaria Right', matriz_right)

                    estado_texto = "PROCESADO Y GUARDADO"
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

        # Dibujar landmarks y actualizar estado actual de dedos
        if results.multi_hand_landmarks:
            for hand_idx, hand_landmarks in enumerate(results.multi_hand_landmarks):
                # Determinar si es mano izquierda o derecha
                hand_label = results.multi_handedness[hand_idx].classification[0].label

                # Convertir a español
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

                # Detectar dedos para esta mano
                dedos, centroide_pos = detectar_dedos_mejorado(hand_landmarks, width, height, hand_label)

                # Actualizar estado en tiempo real
                estado_manos[mano_actual] = dedos.tolist()

                # Dibujar centroide
                if centroide_pos != (0, 0):
                    cv2.circle(frame, centroide_pos, 8, (0, 255, 0), -1)
                    cv2.circle(frame, centroide_pos, 10, (255, 255, 255), 2)

        # Mostrar información en pantalla
        cv2.putText(frame, estado_texto, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, color_estado, 2)
        cv2.putText(frame, f"Frames: {len(frames_video)}", (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1)
        cv2.putText(frame, f"Clase: {clase_actual}", (10, 90), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 0), 2)

        # Mostrar estado de dedos en tiempo real
        cv2.putText(frame, f"Derecha: {estado_manos['derecha']}", (10, 120),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
        cv2.putText(frame, f"Izquierda: {estado_manos['izquierda']}", (10, 150),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)

        # Mostrar el estado de las manos en la consola
        print("Estado de manos:", estado_manos)

        cv2.imshow('Trayectoria Media Left', pizarra_left)
        cv2.imshow('Trayectoria Media Right', pizarra_right)
        cv2.imshow('Camara - Detección de Dedos', frame)

        key = cv2.waitKey(1) & 0xFF
        if key == 27:  # ESC
            break
        elif key == ord('c') or key == ord('C'):
            cambiar_clase()
        elif key == ord('g') or key == ord('G'):
            # Guardar manualmente con datos actuales
            secuencia_left = np.array([estado_manos["izquierda"]] * 5)
            secuencia_right = np.array([estado_manos["derecha"]] * 5)
            guardar_en_csv(clase_actual, secuencia_left, secuencia_right, vector_left, vector_right)
            print(f"Gesto guardado manualmente - Clase: '{clase_actual}'")

        mano_detectada_anteriormente = mano_actualmente_detectada

# Estadísticas finales
if os.path.exists(archivo_csv):
    df = pd.read_csv(archivo_csv)
    print(f"\nESTADÍSTICAS FINALES:")
    print(f"   Total de gestos guardados: {len(df)}")
    print(f"   Clases registradas: {df['clase'].unique().tolist()}")

print("Programa terminado")
cap.release()
cv2.destroyAllWindows()