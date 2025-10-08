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

# Variables globales
frames_video = []
grabando = False
mano_detectada_anteriormente = False
estado_dedos_actual = np.array([0, 0, 0, 0, 0])

# Modelos
modelo_dedos = None
modelo_trayectoria = None

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
    
    # Evitar división por cero
    if l1 * l3 == 0:
        angle = 0
    else:
        cos_angle = (l1**2 + l3**2 - l2**2) / (2 * l1 * l3)
        cos_angle = np.clip(cos_angle, -1, 1)
        angle = degrees(acos(cos_angle))
    
    dedo_pulgar = 1 if angle > 150 else 0
    
    # Detectar otros dedos (índice, medio, anular, meñique)
    if len(coordenadas_palma) > 0:
        nx, ny = centroide(coordenadas_palma)
        coordenadas_centroide = np.array([nx, ny])
        coordenadas_bases = np.array(coordenadas_bases)
        coordenadas_puntas = np.array(coordenadas_puntas)
        
        dis_centroid_puntas = np.linalg.norm(coordenadas_centroide - coordenadas_puntas, axis=1)
        dis_centroid_bases = np.linalg.norm(coordenadas_centroide - coordenadas_bases, axis=1)
        diferencia = dis_centroid_puntas - dis_centroid_bases
        
        dedos = (diferencia > 0).astype(int)
        dedos = np.append(dedo_pulgar, dedos)
    else:
        dedos = np.array([dedo_pulgar, 0, 0, 0, 0])
        nx, ny = 0, 0
    
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

def estandarizar_frames(frames, num_frames_deseado=20):
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

def obtener_frames_medios(centroides, num_frames=5):
    """Obtiene los frames del medio de la secuencia"""
    if len(centroides) <= num_frames:
        return centroides
    
    inicio = (len(centroides) - num_frames) // 2
    return centroides[inicio:inicio + num_frames]

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
    else:
        pizarron = 255 * np.ones((200, 200, 3), dtype=np.uint8)
        cv2.putText(pizarron, "Sin datos", (50, 100), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (100, 100, 100), 1)
    
    return pizarron

def pizarron_a_vector_binario(pizarron, tamano_salida=(20, 20), umbral=128):
    """Convierte el pizarrón a un vector binario (0 y 255)"""
    gris = cv2.cvtColor(pizarron, cv2.COLOR_BGR2GRAY)
    gris_redim = cv2.resize(gris, tamano_salida)
    _, binaria = cv2.threshold(gris_redim, umbral, 255, cv2.THRESH_BINARY)
    vector_binario = binaria.flatten()
    
    # Normalizar a 0 y 1 (como se hizo en el entrenamiento)
    vector_binario_normalizado = (vector_binario / 255.0).astype(np.float64)
    
    return vector_binario_normalizado, binaria

def cargar_modelos():
    """Carga ambos modelos entrenados"""
    global modelo_dedos, modelo_trayectoria
    
    try:
        modelo_dedos = joblib.load("modelo_dedos.pkl")
        modelo_trayectoria = joblib.load("modelo_trayectoria.pkl")
        print("✅ Modelos cargados exitosamente:")
        print(f"   - Modelo dedos: {type(modelo_dedos).__name__}")
        print(f"   - Modelo trayectoria: {type(modelo_trayectoria).__name__}")
        return True
    except Exception as e:
        print(f"❌ Error al cargar modelos: {e}")
        return False

def predecir_con_dedos(dedos):
    """Predice usando solo la configuración de dedos"""
    if modelo_dedos is None:
        return "Desconocido", 0
    
    # Asegurar que tenemos 5 dedos
    if len(dedos) != 5:
        dedos = np.zeros(5)
    
    # Preparar datos para el modelo de dedos
    X_dedos = np.array([dedos])
    
    # Predecir
    prediccion = modelo_dedos.predict(X_dedos)[0]
    probabilidades = modelo_dedos.predict_proba(X_dedos)[0]
    confianza = np.max(probabilidades)
    
    return prediccion, confianza

def predecir_con_trayectoria(vector_binario):
    """Predice usando solo la trayectoria"""
    if modelo_trayectoria is None:
        return "Desconocido", 0
    
    # Asegurar que tenemos 400 pixels
    if len(vector_binario) != 400:
        vector_binario = np.zeros(400)
    
    # Preparar datos para el modelo de trayectoria
    X_trayectoria = np.array([vector_binario])
    
    # Predecir
    prediccion = modelo_trayectoria.predict(X_trayectoria)[0]
    probabilidades = modelo_trayectoria.predict_proba(X_trayectoria)[0]
    confianza = np.max(probabilidades)
    
    return prediccion, confianza

def predecir_combinado(dedos, vector_binario):
    """Combina las predicciones de ambos modelos (ponderado por precisión)"""
    
    # Predecir con cada modelo
    pred_dedos, conf_dedos = predecir_con_dedos(dedos)
    pred_trayectoria, conf_trayectoria = predecir_con_trayectoria(vector_binario)
    
    print(f"🔍 Predicciones individuales:")
    print(f"   - Dedos: '{pred_dedos}' (confianza: {conf_dedos:.2%})")
    print(f"   - Trayectoria: '{pred_trayectoria}' (confianza: {conf_trayectoria:.2%})")
    
    # Estrategia de combinación: preferir dedos (mayor precisión)
    # Pero si la confianza de trayectoria es muy alta, considerarla
    if conf_dedos > 0.8:
        prediccion_final = pred_dedos
        confianza_final = conf_dedos
        metodo = "DEDOS (alta confianza)"
    elif conf_trayectoria > 0.9:
        prediccion_final = pred_trayectoria
        confianza_final = conf_trayectoria
        metodo = "TRAYECTORIA (muy alta confianza)"
    elif pred_dedos == pred_trayectoria:
        prediccion_final = pred_dedos
        confianza_final = (conf_dedos + conf_trayectoria) / 2
        metodo = "CONSENSO (ambos modelos)"
    else:
        # Preferir dedos por tener mejor precisión general
        prediccion_final = pred_dedos
        confianza_final = conf_dedos
        metodo = "DEDOS (mejor precisión)"
    
    return prediccion_final, confianza_final, metodo, pred_dedos, pred_trayectoria

# Cargar modelos al inicio
print("🤖 SISTEMA DE INFERENCIA LSM - MODELOS SEPARADOS")
print("=" * 50)

if not cargar_modelos():
    print("❌ No se pudieron cargar los modelos. Ejecuta model.py primero.")
    exit()

cap = cv2.VideoCapture(0)

print("\nCONTROLES:")
print("   ESC - Salir del programa")
print("   R - Reiniciar grabación actual")
print("   D - Mostrar/ocultar detalles de predicción")
print("\nESTRATEGIA: Se prioriza el modelo de dedos (81.82% precisión) sobre trayectoria (27.27%)")

with mp_hands.Hands(model_complexity=1, max_num_hands=1, min_detection_confidence=0.7) as hands:
    ultima_deteccion = time.time()
    pizarra_actual = crear_pizarron([], 'Esperando')
    vector_actual = None
    matriz_actual = None
    
    # Variables para la predicción
    ultima_prediccion = "Ninguna"
    ultima_confianza = 0
    mostrar_detalles = True
    metodo_ultimo = ""
    pred_dedos_ultima = ""
    pred_trayectoria_ultima = ""
    
    # Estructuras para almacenar datos durante la grabación
    centroides_trayectoria = []
    estados_dedos_trayectoria = []
    
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        
        frame = cv2.flip(frame, 1)
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = hands.process(frame_rgb)
        
        mano_actualmente_detectada = results.multi_hand_landmarks is not None
        
        # Lógica de inicio/fin de grabación
        if not grabando and mano_actualmente_detectada and not mano_detectada_anteriormente:
            grabando = True
            frames_video = []
            centroides_trayectoria = []
            estados_dedos_trayectoria = []
            print("\n🎥 ¡Mano detectada! Comenzando grabación...")
        
        if grabando:
            if mano_actualmente_detectada:
                ultima_deteccion = time.time()
                frames_video.append(frame.copy())
                
                # Extraer centroide y dedos del frame actual
                centroide_pos, dedos = extraer_centroide_y_dedos(frame, hands)
                if centroide_pos is not None and dedos is not None:
                    centroides_trayectoria.append(centroide_pos)
                    estados_dedos_trayectoria.append(dedos)
                    estado_dedos_actual = dedos
                
                estado_texto = f"GRABANDO: {len(frames_video)} frames"
                color_estado = (0, 255, 0)
            else:
                if time.time() - ultima_deteccion > 1.0:
                    grabando = False
                    print(f"⏹️  Grabación terminada. {len(frames_video)} frames capturados")
                    
                    if len(frames_video) > 0 and len(centroides_trayectoria) > 0:
                        print("🔄 Procesando datos para predicción...")
                        
                        # Estandarizar centroides
                        centroides_20 = estandarizar_frames(centroides_trayectoria, 20)
                        
                        # Obtener frames medios
                        centroides_medios = obtener_frames_medios(centroides_20, 5)
                        
                        # Crear pizarrón
                        pizarra_actual = crear_pizarron(centroides_medios, 'Trayectoria Media')
                        
                        # Convertir a vector binario
                        vector_binario, matriz_binaria = pizarron_a_vector_binario(pizarra_actual)
                        
                        # Obtener estado de dedos predominante
                        if len(estados_dedos_trayectoria) > 0:
                            dedos_predominantes = np.round(np.mean(estados_dedos_trayectoria, axis=0)).astype(int)
                            print(f"   Dedos predominantes: {dedos_predominantes}")
                        else:
                            dedos_predominantes = estado_dedos_actual
                        
                        # REALIZAR PREDICCIÓN COMBINADA
                        print("🤖 Realizando predicción combinada...")
                        prediccion, confianza, metodo, pred_dedos, pred_trayectoria = predecir_combinado(
                            dedos_predominantes, vector_binario
                        )
                        
                        ultima_prediccion = prediccion
                        ultima_confianza = confianza
                        metodo_ultimo = metodo
                        pred_dedos_ultima = pred_dedos
                        pred_trayectoria_ultima = pred_trayectoria
                        
                        print(f"\n🎯 PREDICCIÓN FINAL:")
                        print(f"   Gesto: '{prediccion}'")
                        print(f"   Confianza: {confianza:.2%}")
                        print(f"   Método: {metodo}")
                        
                        # Guardar para mostrar en ventanas
                        vector_actual = vector_binario
                        matriz_actual = matriz_binaria
                        
                        # Mostrar imagen binaria
                        cv2.imshow('Imagen Binaria', matriz_binaria)
                        
                    else:
                        print("❌ No hay datos suficientes para predicción")
                    
                    estado_texto = f"PREDICCIÓN COMPLETADA"
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
                if dedos_actuales is not None:
                    estado_dedos_actual = dedos_actuales
                
                # Mostrar centroide
                if centroide_pos:
                    cv2.circle(frame, centroide_pos, 4, color_estado, -1)
        
        # Mostrar información en pantalla
        cv2.putText(frame, estado_texto, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, color_estado, 2)
        cv2.putText(frame, f"Frames: {len(frames_video)}", (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1)
        
        # Mostrar predicción principal
        color_prediccion = (0, 255, 0) if ultima_confianza > 0.7 else (0, 165, 255) if ultima_confianza > 0.5 else (0, 0, 255)
        texto_prediccion = f"Pred: '{ultima_prediccion}' ({ultima_confianza:.1%})"
        cv2.putText(frame, texto_prediccion, (10, 90), cv2.FONT_HERSHEY_SIMPLEX, 0.7, color_prediccion, 2)
        
        # Mostrar detalles si están activados
        if mostrar_detalles and ultima_prediccion != "Ninguna":
            y_offset = 120
            # cv2.putText(frame, f"Metodo: {metodo_ultimo}", (10, y_offset), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 200, 0), 1)
            y_offset += 25
            # cv2.putText(frame, f"Dedos: '{pred_dedos_ultima}'", (10, y_offset), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 200, 200), 1)
            y_offset += 25
            # cv2.putText(frame, f"Trayec: '{pred_trayectoria_ultima}'", (10, y_offset), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 0, 200), 1)
        
        if estado_dedos_actual is not None:
            dedos_texto = f"Dedos: {''.join([str(d) for d in estado_dedos_actual])}"
            cv2.putText(frame, dedos_texto, (10, frame.shape[0] - 30), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 0), 2)
        
        # Mostrar información de modelos
        info_modelo = f"Modelos: Dedos(82%) + Trayec(27%)"
        cv2.putText(frame, info_modelo, (frame.shape[1] - 300, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 200, 200), 1)
        
        # Mostrar ventanas
        cv2.imshow('Trayectoria Media', pizarra_actual)
        cv2.imshow('Camara LSM - INFERENCIA', frame)
        
        # Manejo de teclas
        key = cv2.waitKey(1) & 0xFF
        if key == 27:  # ESC
            break
        elif key == ord('r') or key == ord('R'):
            # Reiniciar grabación
            grabando = False
            frames_video = []
            centroides_trayectoria = []
            estados_dedos_trayectoria = []
            ultima_prediccion = "Ninguna"
            ultima_confianza = 0
            pizarra_actual = crear_pizarron([], 'Reiniciado')
            print("🔄 Grabación reiniciada")
        elif key == ord('d') or key == ord('D'):
            mostrar_detalles = not mostrar_detalles
            print(f"🔍 Detalles de predicción: {'ACTIVADOS' if mostrar_detalles else 'DESACTIVADOS'}")
        
        # Actualizar estado anterior
        mano_detectada_anteriormente = mano_actualmente_detectada

print("✅ Sistema de inferencia terminado")
cap.release()
cv2.destroyAllWindows()