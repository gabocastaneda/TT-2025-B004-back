import mediapipe as mp
import cv2
import numpy as np
import time
import os
import pandas as pd
from math import acos, degrees
import joblib

mp_drawing = mp.solutions.drawing_utils
mp_drawing_styles = mp.solutions.drawing_styles
mp_hands = mp.solutions.hands

# Configuración de dedos
pulgar = [1, 2, 4]
puntos_palma = [0, 1, 2, 5, 9, 13, 17]
bases = [6, 10, 14, 18]
puntas = [8, 12, 16, 20]

# Variables globales para inferencia
frames_inferencia = []
mano_detectada_anteriormente = False
estado_dedos_actual = []
modelo_knn = "knn_model.pkl"
prediccion_actual = ""
confianza_actual = 0.0

# Buffers para inferencia en tiempo real
centroides_buffer = []
estados_dedos_buffer = []
grabando_inferencia = False
inicio_grabacion = 0

def centroide(lista_coordenadas):
    coordenadas = np.array(lista_coordenadas)
    centroid = np.mean(coordenadas, axis=0)
    centroid = int(centroid[0]), int(centroid[1])
    return centroid

def detectar_dedos(hand_landmarks, width, height):
    """Detecta el estado de los 5 dedos (0 = cerrado, 1 = abierto)"""
    coordinadas_pulgar = []
    coordenadas_palma = []
    coordenadas_puntas = []
    coordenadas_bases = []
    
    # Procesar puntos del pulgar
    for i in pulgar:
        x = int(hand_landmarks.landmark[i].x * width)
        y = int(hand_landmarks.landmark[i].y * height)
        coordinadas_pulgar.append([x, y])
    
    # Procesar puntos de la palma
    for i in puntos_palma:
        x = int(hand_landmarks.landmark[i].x * width)
        y = int(hand_landmarks.landmark[i].y * height)
        coordenadas_palma.append([x, y])
    
    # Procesar bases de dedos
    for i in bases:
        x = int(hand_landmarks.landmark[i].x * width)
        y = int(hand_landmarks.landmark[i].y * height)
        coordenadas_bases.append([x, y])
    
    # Procesar puntas de dedos
    for i in puntas:
        x = int(hand_landmarks.landmark[i].x * width)
        y = int(hand_landmarks.landmark[i].y * height)
        coordenadas_puntas.append([x, y])
    
    # Detectar pulgar (ley de cosenos)
    p1 = np.array(coordinadas_pulgar[0])
    p2 = np.array(coordinadas_pulgar[1])
    p3 = np.array(coordinadas_pulgar[2])
    
    l1 = np.linalg.norm(p2 - p3)
    l2 = np.linalg.norm(p1 - p3)
    l3 = np.linalg.norm(p1 - p2)
    
    if l1 * l3 == 0:
        angle = 0
    else:
        angle = degrees(acos((l1**2 + l3**2 - l2**2) / (2 * l1 * l3)))
    
    dedo_pulgar = 1 if angle > 150 else 0
    
    # Detectar otros dedos (índice, medio, anular, meñique)
    nx, ny = centroide(coordenadas_palma)
    coordenadas_centroide = np.array([nx, ny])
    coordenadas_bases = np.array(coordenadas_bases)
    coordenadas_puntas = np.array(coordenadas_puntas)
    
    dis_centroid_puntas = np.linalg.norm(coordenadas_centroide - coordenadas_puntas, axis=1)
    dis_centroid_bases = np.linalg.norm(coordenadas_centroide - coordenadas_bases, axis=1)
    diferencia = dis_centroid_puntas - dis_centroid_bases
    
    dedos = (diferencia > 0).astype(int)
    dedos = np.append(dedo_pulgar, dedos)
    
    return dedos, (nx, ny)

def extraer_centroide_y_dedos(frame, hands):
    """Extrae centroide y estado de dedos de un frame"""
    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = hands.process(frame_rgb)
    
    if results.multi_hand_landmarks:
        for hand_landmarks in results.multi_hand_landmarks:
            dedos, centroide_pos = detectar_dedos(hand_landmarks, frame.shape[1], frame.shape[0])
            return centroide_pos, dedos
    return None, None

def estandarizar_frames(frames, num_frames_deseado=40):
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
    
    indices = np.linspace(0, len(frames)-1, num_frames_deseado, dtype=int)
    return [frames[i] for i in indices]

def obtener_frames_medios(frames, num_frames=10):
    """Obtiene los frames del medio de la secuencia"""
    if len(frames) <= num_frames:
        return frames
    
    inicio = (len(frames) - num_frames) // 2
    return frames[inicio:inicio + num_frames]

def crear_pizarron(trayectoria, nombre):
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

def cargar_modelo_knn():
    """Carga el modelo KNN entrenado"""
    if not os.path.exists(modelo_knn):
        print(f"No se encuentra el modelo: {modelo_knn}")
        print("Ejecuta primero: python entrenamiento.py")
        return None
    
    try:
        modelo = joblib.load(modelo_knn)
        print(f"Modelo KNN cargado exitosamente")
        print(f"Clases disponibles: {modelo.classes_.tolist()}")
        print(f"Características: {modelo.n_features_in_}")
        return modelo
    except Exception as e:
        print(f"Error al cargar el modelo: {e}")
        return None

def predecir_gesto(secuencia_dedos_centrales, vector_binario, modelo):
    """Predice la clase del gesto usando el modelo KNN"""
    # Preparar características en el mismo formato que durante el entrenamiento
    vector_dedos = np.array(secuencia_dedos_centrales).flatten()
    vector_trayectoria = (vector_binario / 255.0).astype(np.float64)
    
    caracteristicas = np.concatenate([vector_dedos, vector_trayectoria]).reshape(1, -1)
    
    # Predecir
    prediccion = modelo.predict(caracteristicas)[0]
    probabilidades = modelo.predict_proba(caracteristicas)[0]
    
    # Obtener confianza
    clase_idx = list(modelo.classes_).index(prediccion)
    confianza = probabilidades[clase_idx]
    
    return prediccion, confianza

# Inicializar sistema
print("SISTEMA DE INFERENCIA EN TIEMPO REAL - LSM")
print("=" * 50)

# Cargar modelo KNN
modelo = cargar_modelo_knn()
if modelo is None:
    exit()

print("\nCONTROLES:")
print("   R - Reiniciar inferencia")
print("   ESPACIO - Forzar predicción con datos actuales")
print("   ESC - Salir del programa")
print("=" * 50)

cap = cv2.VideoCapture(0)

with mp_hands.Hands(model_complexity=1, max_num_hands=1, min_detection_confidence=0.95) as hands:
    ultima_deteccion = time.time()
    pizarra_actual = crear_pizarron([], 'Esperando')
    vector_actual = None
    matriz_actual = None
    
    # Variables para control de inferencia
    ultima_prediccion = time.time()
    frames_capturados = 0
    
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        
        frame = cv2.flip(frame, 1)
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = hands.process(frame_rgb)
        
        mano_actualmente_detectada = results.multi_hand_landmarks is not None
        
        # Lógica de detección de inicio/fin de seña
        if not grabando_inferencia and mano_actualmente_detectada and not mano_detectada_anteriormente:
            # Inicio de seña detectado
            grabando_inferencia = True
            inicio_grabacion = time.time()
            centroides_buffer = []
            estados_dedos_buffer = []
            frames_capturados = 0
            print("Inicio de seña detectado - Capturando...")
        
        if grabando_inferencia:
            if mano_actualmente_detectada:
                ultima_deteccion = time.time()
                frames_capturados += 1
                
                # Extraer centroide y dedos del frame actual
                centroide_pos, dedos = extraer_centroide_y_dedos(frame, hands)
                if centroide_pos is not None and dedos is not None:
                    centroides_buffer.append(centroide_pos)
                    estados_dedos_buffer.append(dedos)
                    estado_dedos_actual = dedos
                
                # Verificar si la seña ha terminado (sin mano por 0.5 segundos)
                estado_texto = f"GRABANDO MANO: {frames_capturados} frames"
                color_estado = (0, 255, 0)
                
            else:
                # Si no hay mano detectada por 0.5 segundos, considerar que la seña terminó
                if time.time() - ultima_deteccion > 0.5:
                    grabando_inferencia = False
                    duracion_seña = time.time() - inicio_grabacion
                    print(f"Seña completada: {frames_capturados} frames ({duracion_seña:.1f}s)")
                    
                    # Realizar predicción con los datos capturados
                    if len(centroides_buffer) >= 5:  # Mínimo 5 frames para una seña válida
                        print("Procesando datos para inferencia...")
                        
                        # Estandarizar a 40 frames (rellenar si es necesario)
                        centroides_estandarizados = estandarizar_frames(centroides_buffer, 40)
                        estados_dedos_estandarizados = estandarizar_frames(estados_dedos_buffer, 40)
                        
                        # Obtener 10 frames medios
                        centroides_medios = obtener_frames_medios(centroides_estandarizados, 10)
                        estados_dedos_medios = obtener_frames_medios(estados_dedos_estandarizados, 10)
                        
                        # Crear pizarrón
                        pizarra_actual = crear_pizarron(centroides_medios, 'Trayectoria Inferencia')
                        
                        # Convertir a vector binario
                        vector_binario, matriz_binaria = pizarron_a_vector_binario(pizarra_actual)
                        
                        # Predecir gesto
                        secuencia_dedos_array = np.array(estados_dedos_medios)
                        prediccion, confianza = predecir_gesto(secuencia_dedos_array, vector_binario, modelo)
                        
                        prediccion_actual = prediccion
                        confianza_actual = confianza
                        
                        print(f"Predicción: '{prediccion}' (confianza: {confianza:.2f})")
                        
                        # Guardar para mostrar en ventanas
                        vector_actual = vector_binario
                        matriz_actual = matriz_binaria
                        
                        # Mostrar imagen binaria
                        # cv2.imshow('Imagen Binaria', matriz_binaria)
                        
                        ultima_prediccion = time.time()
                    else:
                        print("Seña demasiado corta para procesar (mínimo 5 frames)")
                    
                    estado_texto = f"PROCESANDO: {frames_capturados} frames"
                    color_estado = (255, 0, 0)
                else:
                    estado_texto = f"GRABANDO: {frames_capturados} frames (sin mano)"
                    color_estado = (0, 165, 255)
        
        else:
            if mano_actualmente_detectada:
                estado_texto = "LISTO - Esperando mano"
                color_estado = (0, 255, 255)
            else:
                estado_texto = "ESPERANDO MANO..."
                color_estado = (0, 0, 255)
        
        # Dibujar landmarks y mostrar dedos en tiempo real
        if results.multi_hand_landmarks:
            for hand_landmarks in results.multi_hand_landmarks:
                mp_drawing.draw_landmarks(
                    frame,
                    hand_landmarks,
                    mp_hands.HAND_CONNECTIONS,
                    mp_drawing_styles.get_default_hand_landmarks_style(),
                    mp_drawing_styles.get_default_hand_connections_style())
                
                # Detectar y mostrar dedos en tiempo real
                dedos_actuales, centroide_pos = detectar_dedos(hand_landmarks, frame.shape[1], frame.shape[0])
                estado_dedos_actual = dedos_actuales
                
                # Mostrar centroide
                if centroide_pos:
                    cv2.circle(frame, centroide_pos, 4, color_estado, -1)
        
        # Mostrar información en pantalla
        cv2.putText(frame, estado_texto, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, color_estado, 2)
        
        if grabando_inferencia:
            cv2.putText(frame, f"Frames: {frames_capturados}", (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1)
        else:
            cv2.putText(frame, f"Ultima prediccion: {frames_capturados} frames", (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1)
        
        if estado_dedos_actual is not None:
            dedos_texto = f"Dedos: {''.join([str(d) for d in estado_dedos_actual])}"
            cv2.putText(frame, dedos_texto, (10, 90), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 0), 2)
        
        if prediccion_actual:
            color_prediccion = (0, 255, 0) if confianza_actual > 0.7 else (0, 165, 255)
            texto_prediccion = f"Gesto: {prediccion_actual} ({confianza_actual:.2f})"
            cv2.putText(frame, texto_prediccion, (10, 120), cv2.FONT_HERSHEY_SIMPLEX, 0.7, color_prediccion, 2)
        
        if vector_actual is not None:
            info_vector = f"Vector: {len(vector_actual)} pixels"
            cv2.putText(frame, info_vector, (10, 150), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 0), 1)
        
        # Mostrar ventanas
        # cv2.imshow('Trayectoria Inferencia', pizarra_actual)
        cv2.imshow('Camara LSM - Inferencia', frame)
        
        # Manejo de teclas
        key = cv2.waitKey(1) & 0xFF
        if key == 27:  # ESC
            break
        elif key == ord('r') or key == ord('R'):
            # Reiniciar inferencia
            grabando_inferencia = False
            centroides_buffer = []
            estados_dedos_buffer = []
            prediccion_actual = ""
            estado_dedos_actual = None
            frames_capturados = 0
            print("Inferencia reiniciada - Esperando nueva seña")
        elif key == 32:  # ESPACIO - Forzar predicción
            if grabando_inferencia and len(centroides_buffer) >= 5:
                grabando_inferencia = False
                print("Predicción forzada por usuario")
                # Procesar datos actuales (código duplicado por simplicidad)
                centroides_estandarizados = estandarizar_frames(centroides_buffer, 40)
                estados_dedos_estandarizados = estandarizar_frames(estados_dedos_buffer, 40)
                centroides_medios = obtener_frames_medios(centroides_estandarizados, 10)
                estados_dedos_medios = obtener_frames_medios(estados_dedos_estandarizados, 10)
                pizarra_actual = crear_pizarron(centroides_medios, 'Trayectoria Inferencia')
                vector_binario, matriz_binaria = pizarron_a_vector_binario(pizarra_actual)
                secuencia_dedos_array = np.array(estados_dedos_medios)
                prediccion, confianza = predecir_gesto(secuencia_dedos_array, vector_binario, modelo)
                prediccion_actual = prediccion
                confianza_actual = confianza
                print(f"Predicción forzada: '{prediccion}' (confianza: {confianza:.2f})")
                vector_actual = vector_binario
                matriz_actual = matriz_binaria
                # cv2.imshow('Imagen Binaria', matriz_binaria)
        
        # Actualizar estado anterior
        mano_detectada_anteriormente = mano_actualmente_detectada

# Estadísticas finales
print(f"\nESTADÍSTICAS DE INFERENCIA:")
print(f"   Frames procesados: {len(frames_inferencia)}")
print(f"   Última predicción: '{prediccion_actual}' (confianza: {confianza_actual:.2f})")

print("Programa terminado")
cap.release()
cv2.destroyAllWindows()