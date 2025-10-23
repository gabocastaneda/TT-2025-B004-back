import mediapipe as mp
import cv2
import numpy as np
import time
import os
import pandas as pd
from math import acos, degrees

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
estado_dedos_actual = None
archivo_csv = "dataset_lsm.csv"
clase_actual = ""  

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

def generar_encabezados():
    """Genera encabezados descriptivos para las columnas"""
    encabezados = ["clase", "dedos_izquierda", "dedos_derecha"]
    
    # Encabezados para los vectores de trayectoria (20x20 = 400 elementos cada uno)
    for i in range(400):
        encabezados.append(f"pixel_izq_{i:03d}")
    for i in range(400):
        encabezados.append(f"pixel_der_{i:03d}")
    
    return encabezados

def inicializar_csv():
    """Inicializa el archivo CSV con los encabezados"""
    if not os.path.exists(archivo_csv):
        encabezados = generar_encabezados()
        df = pd.DataFrame(columns=encabezados)
        df.to_csv(archivo_csv, index=False)
        print(f"Archivo CSV creado: {archivo_csv}")
        print(f"Estructura: Clase + dedos_izq + dedos_der + 400px_izq + 400px_der = 802 columnas")
    else:
        print(f"Archivo CSV existente: {archivo_csv}")
        print(f"Se agregarán nuevos registros al final")

def guardar_en_csv(clase, dedos_izq, dedos_der, vector_izq, vector_der):
    """Guarda los datos en el archivo CSV"""
    # Convertir a formato binario (0 y 1)
    vector_izq_bin = (vector_izq / 255.0).astype(np.float64)
    vector_der_bin = (vector_der / 255.0).astype(np.float64)

    # Convertir arrays a strings para almacenar
    secuencia_izq = str(dedos_izq.tolist())
    secuencia_der = str(dedos_der.tolist())

    # Crear fila de datos
    datos = [clase, secuencia_izq, secuencia_der] + vector_izq_bin.tolist() + vector_der_bin.tolist()

    # Generar encabezados
    encabezados = ["clase", "dedos_izquierda", "dedos_derecha"] + \
                [f"pixel_izq_{i:03d}" for i in range(len(vector_izq_bin))] + \
                [f"pixel_der_{i:03d}" for i in range(len(vector_der_bin))]

    # Crear archivo si no existe
    if not os.path.exists(archivo_csv):
        pd.DataFrame(columns=encabezados).to_csv(archivo_csv, index=False)

    # Leer y agregar nueva fila
    df = pd.read_csv(archivo_csv)
    nueva_fila = pd.DataFrame([datos], columns=encabezados)
    df = pd.concat([df, nueva_fila], ignore_index=True)
    df.to_csv(archivo_csv, index=False)
    
    print(f"Datos guardados: Clase '{clase}'")

def cambiar_clase():
    """Permite cambiar la clase actual mediante input"""
    global clase_actual
    print(f"\nCLASE ACTUAL: '{clase_actual}'")
    nueva_clase = input("   Ingrese nueva clase (o Enter para mantener actual): ").strip()
    if nueva_clase:
        clase_actual = nueva_clase
        print(f"Nueva clase establecida: '{clase_actual}'")
    else:
        print(f"Manteniendo clase: '{clase_actual}'")
    return clase_actual

# Inicializar sistema
inicializar_csv()
cap = cv2.VideoCapture(0)

print("\nCONTROLES:")
print("   C - Cambiar clase actual")
print("   G - Guardar gesto actual en CSV")
print("   ESC - Salir del programa")
print(f"\nCLASE INICIAL: '{clase_actual}'")

with mp_hands.Hands(model_complexity=1, max_num_hands=2, min_detection_confidence=0.7, min_tracking_confidence=0.5) as hands:
    ultima_deteccion = time.time()
    pizarra_izq_actual = crear_pizarron([], 'Esperando Izq')
    pizarra_der_actual = crear_pizarron([], 'Esperando Der')
    
    # Estructuras para almacenar datos durante la grabación
    trayectoria_izq = []
    trayectoria_der = []
    estados_dedos_izq = []
    estados_dedos_der = []
    
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
            trayectoria_izq = []
            trayectoria_der = []
            estados_dedos_izq = []
            estados_dedos_der = []
            print("¡Mano detectada! Comenzando grabación...")
        
        if grabando:
            if mano_actualmente_detectada:
                ultima_deteccion = time.time()
                frames_video.append(frame.copy())
                
                # Extraer centroide y dedos del frame actual
                centroides, dedos_manos = extraer_centroide_y_dedos(frame, hands)

                # Procesar según número de manos detectadas
                if len(dedos_manos) == 1:
                    # Una mano - asumir derecha por defecto
                    dedos_der = dedos_manos[0]
                    dedos_izq = np.zeros(5, dtype=int)
                    trayectoria_der.append(centroides[0])
                    estados_dedos_der.append(dedos_der)
                elif len(dedos_manos) >= 2:
                    # Dos manos - tomar las primeras dos
                    dedos_izq, dedos_der = dedos_manos[0], dedos_manos[1]
                    trayectoria_izq.append(centroides[0])
                    trayectoria_der.append(centroides[1])
                    estados_dedos_izq.append(dedos_izq)
                    estados_dedos_der.append(dedos_der)
                
                estado_texto = f"GRABANDO: {len(frames_video)} frames"
                color_estado = (0, 255, 0)
            else:
                if time.time() - ultima_deteccion > 1.0:
                    grabando = False
                    print(f"Grabación terminada. {len(frames_video)} frames capturados")
                    
                    if len(frames_video) > 0:
                        print("Procesando datos...")
                        
                        # Obtener estados de dedos medios (últimos frames)
                        num_frames_medios = min(5, len(estados_dedos_izq), len(estados_dedos_der))
                        
                        if num_frames_medios > 0:
                            # Tomar los últimos estados de dedos
                            dedos_izq_final = estados_dedos_izq[-1] if estados_dedos_izq else np.zeros(5, dtype=int)
                            dedos_der_final = estados_dedos_der[-1] if estados_dedos_der else np.zeros(5, dtype=int)
                            
                            # Crear pizarrones
                            pizarra_izq = crear_pizarron(trayectoria_izq, 'Mano Izquierda')
                            pizarra_der = crear_pizarron(trayectoria_der, 'Mano Derecha')
                            
                            # Convertir a vectores binarios
                            vector_izq, matriz_izq = pizarron_a_vector_binario(pizarra_izq)
                            vector_der, matriz_der = pizarron_a_vector_binario(pizarra_der)
                            
                            # Guardar en CSV
                            guardar_en_csv(clase_actual, dedos_izq_final, dedos_der_final, vector_izq, vector_der)
                            
                            # Actualizar para mostrar
                            pizarra_izq_actual = pizarra_izq
                            pizarra_der_actual = pizarra_der
                            
                            print(f"✅ Gesto procesado y guardado")
                        
                    estado_texto = f"PROCESADO"
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
                estado_dedos_actual = dedos_actuales
                
                # Mostrar centroide
                if centroide_pos:
                    cv2.circle(frame, centroide_pos, 4, color_estado, -1)
        
        # Mostrar información en pantalla
        cv2.putText(frame, estado_texto, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, color_estado, 2)
        cv2.putText(frame, f"Frames: {len(frames_video)}", (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1)
        cv2.putText(frame, f"Clase: {clase_actual}", (10, 90), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 0), 2)
        
        if estado_dedos_actual is not None:
            dedos_texto = f"Dedos: {''.join([str(d) for d in estado_dedos_actual])}"
            cv2.putText(frame, dedos_texto, (10, 120), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 0), 2)
        
        # Mostrar ventanas
        cv2.imshow('Trayectoria Izquierda', pizarra_izq_actual)
        cv2.imshow('Trayectoria Derecha', pizarra_der_actual)
        cv2.imshow('Camara LSM', frame)
        
        # Manejo de teclas
        key = cv2.waitKey(1) & 0xFF
        if key == 27:  # ESC
            break
        elif key == ord('c') or key == ord('C'):
            cambiar_clase()
        elif key == ord('g') or key == ord('G'):
            if estado_dedos_actual is not None:
                # Guardado manual con datos actuales
                dedos_manual = estado_dedos_actual
                pizarra_manual = crear_pizarron([], 'Manual')
                vector_manual, _ = pizarron_a_vector_binario(pizarra_manual)
                guardar_en_csv(clase_actual, np.zeros(5), dedos_manual, vector_manual, vector_manual)
                print(f"✅ Gesto manual guardado - Clase: '{clase_actual}'")
        
        # Actualizar estado anterior
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