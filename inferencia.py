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
modelo_knn = "knn_model.pkl"
prediccion_actual = ""
confianza_actual = 0.0

# Buffers para inferencia en tiempo real
trayectoria_izq_buffer = []
trayectoria_der_buffer = []
estados_dedos_izq_buffer = []
estados_dedos_der_buffer = []
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
    """Extrae centroides y estados de dedos de ambas manos"""
    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = hands.process(frame_rgb)
    
    centroides = []
    dedos_manos = []
    
    if results.multi_hand_landmarks:
        for hand_landmarks in results.multi_hand_landmarks:
            dedos, centroide_pos = detectar_dedos(hand_landmarks, frame.shape[1], frame.shape[0])
            centroides.append(centroide_pos)
            dedos_manos.append(dedos)
    
    return centroides, dedos_manos

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
        print("Ejecuta primero el entrenamiento")
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

def ajustar_dimensionalidad(caracteristicas, n_features_esperado):
    """Ajusta la dimensionalidad de las características al tamaño esperado"""
    if len(caracteristicas) == n_features_esperado:
        return caracteristicas.reshape(1, -1)
    
    elif len(caracteristicas) < n_features_esperado:
        # Rellenar con ceros
        print(f"Rellenando características: {len(caracteristicas)} -> {n_features_esperado}")
        caracteristicas_ajustadas = np.pad(caracteristicas, 
                                        (0, n_features_esperado - len(caracteristicas)), 
                                        'constant', constant_values=0)
    
    else:
        # Recortar
        print(f"Recortando características: {len(caracteristicas)} -> {n_features_esperado}")
        caracteristicas_ajustadas = caracteristicas[:n_features_esperado]
    
    return caracteristicas_ajustadas.reshape(1, -1)

def predecir_gesto(dedos_izq, dedos_der, vector_izq, vector_der, modelo):
    """Predice la clase del gesto usando el modelo KNN"""
    # Convertir a formato binario (0 y 1) - igual que en data.py
    vector_izq_bin = (vector_izq / 255.0).astype(np.float64)
    vector_der_bin = (vector_der / 255.0).astype(np.float64)
    
    # Concatenar características en el mismo orden que durante el entrenamiento
    caracteristicas = np.concatenate([
        dedos_izq,           # 5 características
        dedos_der,           # 5 características  
        vector_izq_bin,      # 400 características
        vector_der_bin       # 400 características
    ])
    
    # Ajustar al tamaño esperado por el modelo
    caracteristicas_ajustadas = ajustar_dimensionalidad(caracteristicas, modelo.n_features_in_)
    
    # Predecir
    prediccion = modelo.predict(caracteristicas_ajustadas)[0]
    probabilidades = modelo.predict_proba(caracteristicas_ajustadas)[0]
    
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

with mp_hands.Hands(model_complexity=1, max_num_hands=2, min_detection_confidence=0.7, min_tracking_confidence=0.5) as hands:
    ultima_deteccion = time.time()
    pizarra_izq_actual = crear_pizarron([], 'Esperando Izq')
    pizarra_der_actual = crear_pizarron([], 'Esperando Der')
    
    # Variables para control de inferencia
    ultima_prediccion = time.time()
    frames_capturados = 0
    estado_dedos_actual = None
    
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        
        frame = cv2.flip(frame, 1)
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = hands.process(frame_rgb)
        
        mano_actualmente_detectada = results.multi_hand_landmarks is not None
        
        # Lógica de detección de inicio/fin de seña
        if not grabando_inferencia and mano_actualmente_detectada:
            # Inicio de seña detectado
            grabando_inferencia = True
            inicio_grabacion = time.time()
            trayectoria_izq_buffer = []
            trayectoria_der_buffer = []
            estados_dedos_izq_buffer = []
            estados_dedos_der_buffer = []
            frames_capturados = 0
            print("Inicio de seña detectado - Capturando...")
        
        if grabando_inferencia:
            if mano_actualmente_detectada:
                ultima_deteccion = time.time()
                frames_capturados += 1
                
                # Extraer centroides y dedos del frame actual
                centroides, dedos_manos = extraer_centroide_y_dedos(frame, hands)
                
                # Procesar según número de manos detectadas
                if len(dedos_manos) == 1:
                    # Una mano - asumir derecha por defecto
                    dedos_der = dedos_manos[0]
                    dedos_izq = np.zeros(5, dtype=int)
                    trayectoria_der.append(centroides[0])
                    estados_dedos_der_buffer.append(dedos_der)
                    estado_dedos_actual = dedos_der
                elif len(dedos_manos) >= 2:
                    # Dos manos - tomar las primeras dos
                    dedos_izq, dedos_der = dedos_manos[0], dedos_manos[1]
                    trayectoria_izq_buffer.append(centroides[0])
                    trayectoria_der_buffer.append(centroides[1])
                    estados_dedos_izq_buffer.append(dedos_izq)
                    estados_dedos_der_buffer.append(dedos_der)
                    estado_dedos_actual = dedos_der  # Mostrar mano derecha por defecto
                
                estado_texto = f"GRABANDO: {frames_capturados} frames"
                color_estado = (0, 255, 0)
                
            else:
                # Si no hay mano detectada por 1 segundo, considerar que la seña terminó
                if time.time() - ultima_deteccion > 1.0:
                    grabando_inferencia = False
                    duracion_seña = time.time() - inicio_grabacion
                    print(f"Seña completada: {frames_capturados} frames ({duracion_seña:.1f}s)")
                    
                    # Realizar predicción con los datos capturados
                    if frames_capturados >= 5:  # Mínimo 5 frames para una seña válida
                        print("Procesando datos para inferencia...")
                        
                        # Obtener últimos estados de dedos
                        dedos_izq_final = estados_dedos_izq_buffer[-1] if estados_dedos_izq_buffer else np.zeros(5, dtype=int)
                        dedos_der_final = estados_dedos_der_buffer[-1] if estados_dedos_der_buffer else np.zeros(5, dtype=int)
                        
                        # Crear pizarrones
                        pizarra_izq = crear_pizarron(trayectoria_izq_buffer, 'Mano Izquierda')
                        pizarra_der = crear_pizarron(trayectoria_der_buffer, 'Mano Derecha')
                        
                        # Convertir a vectores binarios
                        vector_izq, matriz_izq = pizarron_a_vector_binario(pizarra_izq)
                        vector_der, matriz_der = pizarron_a_vector_binario(pizarra_der)
                        
                        # Predecir gesto
                        prediccion, confianza = predecir_gesto(dedos_izq_final, dedos_der_final, vector_izq, vector_der, modelo)
                        
                        prediccion_actual = prediccion
                        confianza_actual = confianza
                        
                        # Actualizar para mostrar
                        pizarra_izq_actual = pizarra_izq
                        pizarra_der_actual = pizarra_der
                        
                        print(f"Predicción: '{prediccion}' (confianza: {confianza:.2f})")
                        
                        ultima_prediccion = time.time()
                    else:
                        print("Seña demasiado corta para procesar (mínimo 5 frames)")
                    
                    estado_texto = f"PROCESADO"
                    color_estado = (255, 0, 0)
                else:
                    estado_texto = f"GRABANDO: {frames_capturados} frames (sin mano)"
                    color_estado = (0, 165, 255)
        
        else:
            if mano_actualmente_detectada:
                estado_texto = "LISTO - Moviendo mano..."
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
        cv2.putText(frame, f"Frames: {frames_capturados}", (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1)
        
        if estado_dedos_actual is not None:
            dedos_texto = f"Dedos: {''.join([str(d) for d in estado_dedos_actual])}"
            cv2.putText(frame, dedos_texto, (10, 90), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 0), 2)
        
        if prediccion_actual:
            color_prediccion = (0, 255, 0) if confianza_actual > 0.7 else (0, 165, 255)
            texto_prediccion = f"Gesto: {prediccion_actual} ({confianza_actual:.2f})"
            cv2.putText(frame, texto_prediccion, (10, 120), cv2.FONT_HERSHEY_SIMPLEX, 0.7, color_prediccion, 2)
        
        # Mostrar ventanas
        cv2.imshow('Trayectoria Izquierda', pizarra_izq_actual)
        cv2.imshow('Trayectoria Derecha', pizarra_der_actual)
        cv2.imshow('Camara LSM - Inferencia', frame)
        
        # Manejo de teclas
        key = cv2.waitKey(1) & 0xFF
        if key == 27:  # ESC
            break
        elif key == ord('r') or key == ord('R'):
            # Reiniciar inferencia
            grabando_inferencia = False
            trayectoria_izq_buffer = []
            trayectoria_der_buffer = []
            estados_dedos_izq_buffer = []
            estados_dedos_der_buffer = []
            prediccion_actual = ""
            estado_dedos_actual = None
            frames_capturados = 0
            pizarra_izq_actual = crear_pizarron([], 'Reiniciado Izq')
            pizarra_der_actual = crear_pizarron([], 'Reiniciado Der')
            print("Inferencia reiniciada - Esperando nueva seña")
        elif key == 32:  # ESPACIO - Forzar predicción
            if grabando_inferencia and frames_capturados >= 5:
                grabando_inferencia = False
                print("Predicción forzada por usuario")
                
                # Obtener últimos estados de dedos
                dedos_izq_final = estados_dedos_izq_buffer[-1] if estados_dedos_izq_buffer else np.zeros(5, dtype=int)
                dedos_der_final = estados_dedos_der_buffer[-1] if estados_dedos_der_buffer else np.zeros(5, dtype=int)
                
                # Crear pizarrones
                pizarra_izq = crear_pizarron(trayectoria_izq_buffer, 'Mano Izquierda')
                pizarra_der = crear_pizarron(trayectoria_der_buffer, 'Mano Derecha')
                
                # Convertir a vectores binarios
                vector_izq, matriz_izq = pizarron_a_vector_binario(pizarra_izq)
                vector_der, matriz_der = pizarron_a_vector_binario(pizarra_der)
                
                # Predecir gesto
                prediccion, confianza = predecir_gesto(dedos_izq_final, dedos_der_final, vector_izq, vector_der, modelo)
                
                prediccion_actual = prediccion
                confianza_actual = confianza
                
                # Actualizar para mostrar
                pizarra_izq_actual = pizarra_izq
                pizarra_der_actual = pizarra_der
                
                print(f"Predicción forzada: '{prediccion}' (confianza: {confianza:.2f})")

# Estadísticas finales
print(f"\nESTADÍSTICAS DE INFERENCIA:")
print(f"   Última predicción: '{prediccion_actual}' (confianza: {confianza_actual:.2f})")

print("Programa terminado")
cap.release()
cv2.destroyAllWindows()