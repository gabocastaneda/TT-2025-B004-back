import mediapipe as mp
import cv2
import numpy as np
import time
import os
import pandas as pd
from math import acos, degrees
from collections import Counter

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
estado_dedos_actual = []

# Base de datos en memoria para KNN (se carga al inicio)
base_datos = []  # Lista de tuplas: (clase, dedos, trayectoria_vector)
k_vecinos = 5  # Número de vecinos para KNN

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
    """Convierte el pizarrón a un vector binario (0 y 1)"""
    gris = cv2.cvtColor(pizarron, cv2.COLOR_BGR2GRAY)
    gris_redim = cv2.resize(gris, tamano_salida)
    _, binaria = cv2.threshold(gris_redim, umbral, 255, cv2.THRESH_BINARY)
    vector_binario = binaria.flatten()
    
    # Normalizar a 0 y 1
    vector_binario_normalizado = (vector_binario / 255.0).astype(np.float64)
    
    return vector_binario_normalizado, binaria

def distancia_euclidiana(vector1, vector2):
    """Calcula la distancia euclidiana entre dos vectores"""
    return np.linalg.norm(vector1 - vector2)

def knn_predict(dedos_actual, trayectoria_actual, k=5):
    """Predice la clase usando KNN con distancia euclidiana"""
    if len(base_datos) == 0:
        return "Sin datos", 0.0, []
    
    # Combinar características: dedos + trayectoria
    caracteristicas_actual = np.concatenate([dedos_actual, trayectoria_actual])
    
    distancias = []
    
    for clase, dedos_db, trayectoria_db in base_datos:
        # Combinar características de la base de datos
        caracteristicas_db = np.concatenate([dedos_db, trayectoria_db])
        
        # Calcular distancia euclidiana
        dist = distancia_euclidiana(caracteristicas_actual, caracteristicas_db)
        distancias.append((dist, clase))
    
    # Ordenar por distancia (más cercanos primero)
    distancias.sort(key=lambda x: x[0])
    
    # Tomar los k vecinos más cercanos
    k_vecinos = distancias[:k]
    
    # Contar las clases de los k vecinos
    clases_vecinos = [clase for _, clase in k_vecinos]
    contador = Counter(clases_vecinos)
    
    # La clase más común
    clase_predicha = contador.most_common(1)[0][0]
    confianza = contador[clase_predicha] / k
    
    return clase_predicha, confianza, k_vecinos

def cargar_base_datos_desde_csv(archivo_csv="dataset_lsm.csv"):
    """Carga la base de datos desde un archivo CSV existente"""
    global base_datos
    
    if not os.path.exists(archivo_csv):
        print(f"⚠️  Archivo {archivo_csv} no encontrado.")
        print("   El sistema funcionará sin base de datos inicial.")
        return False
    
    try:
        df = pd.read_csv(archivo_csv)
        
        # Verificar que tenga las columnas necesarias
        columnas_requeridas = ['clase', 'dedo_pulgar', 'dedo_indice', 'dedo_medio', 'dedo_anular', 'dedo_menique']
        if not all(col in df.columns for col in columnas_requeridas):
            print("❌ El CSV no tiene el formato esperado")
            return False
        
        # Limpiar la base de datos actual
        base_datos = []
        
        # Procesar cada fila del CSV
        for _, fila in df.iterrows():
            clase = fila['clase']
            
            # Extraer dedos
            dedos = np.array([
                fila['dedo_pulgar'],
                fila['dedo_indice'], 
                fila['dedo_medio'],
                fila['dedo_anular'],
                fila['dedo_menique']
            ])
            
            # Extraer trayectoria (pixels)
            columnas_pixels = [col for col in df.columns if col.startswith('pixel_')]
            if len(columnas_pixels) > 0:
                trayectoria = fila[columnas_pixels].values.astype(np.float64)
            else:
                trayectoria = np.zeros(400)  # Vector por defecto
            
            base_datos.append((clase, dedos, trayectoria))
        
        print(f"✅ Base de datos cargada: {len(base_datos)} ejemplos de {len(df['clase'].unique())} clases")
        print(f"   Clases disponibles: {', '.join(df['clase'].unique())}")
        return True
        
    except Exception as e:
        print(f"❌ Error al cargar base de datos: {e}")
        return False

def mostrar_estadisticas():
    """Muestra estadísticas de la base de datos"""
    if len(base_datos) == 0:
        print("📊 Base de datos vacía")
        return
    
    clases = [clase for clase, _, _ in base_datos]
    contador = Counter(clases)
    
    print(f"\n📊 ESTADÍSTICAS BASE DE DATOS:")
    print(f"   Total de ejemplos: {len(base_datos)}")
    for clase, count in contador.items():
        print(f"   - '{clase}': {count} ejemplos")

# ============================================================================
# PROGRAMA PRINCIPAL
# ============================================================================

print("\n🎯 SISTEMA KNN - CLASIFICACIÓN EN TIEMPO REAL")
print("=" * 50)

# Cargar base de datos al inicio
cargar_base_datos_desde_csv()

cap = cv2.VideoCapture(0)

print("\nCONTROLES:")
print("   K - Cambiar número de vecinos (actual: 5)")
print("   E - Mostrar estadísticas de la base de datos")
print("   ESC - Salir del programa")
print("\nFUNCIONAMIENTO:")
print("   - El sistema detecta automáticamente cuando aparece una mano")
print("   - Graba la trayectoria y configuración de dedos")
print("   - Clasifica usando KNN con la base de datos cargada")
print("   - Muestra la predicción y confianza en tiempo real")

with mp_hands.Hands(model_complexity=1, max_num_hands=1, min_detection_confidence=0.7) as hands:
    ultima_deteccion = time.time()
    pizarra_actual = crear_pizarron([], 'Esperando')
    vector_actual = None
    matriz_actual = None
    
    # Variables para predicción
    ultima_prediccion = "Ninguna"
    ultima_confianza = 0.0
    
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
                        print("🔄 Procesando datos para KNN...")
                        
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
                        else:
                            dedos_predominantes = estado_dedos_actual
                        
                        # REALIZAR PREDICCIÓN KNN
                        if len(base_datos) > 0:
                            prediccion, confianza, vecinos = knn_predict(dedos_predominantes, vector_binario, k_vecinos)
                            
                            ultima_prediccion = prediccion
                            ultima_confianza = confianza
                            
                            print(f"\n🎯 PREDICCIÓN KNN (k={k_vecinos}):")
                            print(f"   Gesto: '{prediccion}'")
                            print(f"   Confianza: {confianza:.2%}")
                            print(f"   Vecinos más cercanos:")
                            for dist, clase in vecinos:
                                print(f"     - '{clase}' (distancia: {dist:.4f})")
                        else:
                            print("⚠️  Base de datos vacía. No se puede realizar predicción.")
                            ultima_prediccion = "Sin datos"
                            ultima_confianza = 0.0
                        
                        # Guardar para mostrar en ventanas
                        vector_actual = vector_binario
                        matriz_actual = matriz_binaria
                        
                        # Mostrar imagen binaria
                        cv2.imshow('Imagen Binaria', matriz_binaria)
                        
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
        
        # Mostrar predicción KNN
        if ultima_prediccion != "Ninguna":
            color_pred = (0, 255, 0) if ultima_confianza > 0.7 else (0, 165, 255) if ultima_confianza > 0.5 else (0, 0, 255)
            texto_pred = f"KNN: '{ultima_prediccion}' ({ultima_confianza:.1%})"
            cv2.putText(frame, texto_pred, (10, 90), cv2.FONT_HERSHEY_SIMPLEX, 0.7, color_pred, 2)
        
        if estado_dedos_actual is not None:
            dedos_texto = f"Dedos: {''.join([str(d) for d in estado_dedos_actual])}"
            cv2.putText(frame, dedos_texto, (10, 120), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 0), 2)
        
        # Mostrar información de la base de datos
        info_bd = f"BD: {len(base_datos)} ejemplos | k: {k_vecinos}"
        cv2.putText(frame, info_bd, (frame.shape[1] - 350, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 200, 200), 1)
        
        # Mostrar ventanas
        cv2.imshow('Trayectoria Media', pizarra_actual)
        cv2.imshow('Camara LSM - KNN', frame)
        
        # Manejo de teclas
        key = cv2.waitKey(1) & 0xFF
        if key == 27:  # ESC
            break
        elif key == ord('k') or key == ord('K'):
            try:
                nuevo_k = int(input(f"   Ingrese nuevo valor de k (actual: {k_vecinos}): "))
                if nuevo_k > 0 and nuevo_k <= len(base_datos):
                    k_vecinos = nuevo_k
                    print(f"   ✅ k cambiado a: {k_vecinos}")
                else:
                    print(f"   ❌ k debe estar entre 1 y {len(base_datos)}")
            except ValueError:
                print("   ❌ Ingrese un número válido")
        elif key == ord('e') or key == ord('E'):
            mostrar_estadisticas()
        
        # Actualizar estado anterior
        mano_detectada_anteriormente = mano_actualmente_detectada

# Mostrar estadísticas finales
mostrar_estadisticas()
print("\n✅ Programa terminado")
cap.release()
cv2.destroyAllWindows()