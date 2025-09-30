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
estado_dedos_actual = []

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

def visualizar_vector_consola(vector_binario, tamano_original=(20, 20)):
    """Visualiza el vector binario en la consola"""
    matriz = vector_binario.reshape(tamano_original)
    
    print("\n" + "="*60)
    print("VECTOR BINARIO DEL GESTO")
    print("="*60)
    
    print(f"\nRepresentación visual ({tamano_original[0]}x{tamano_original[1]}):")
    print("┌" + "─" * (tamano_original[1] * 2) + "┐")
    for i in range(tamano_original[0]):
        fila_str = "│"
        for j in range(tamano_original[1]):
            if matriz[i, j] == 255:
                fila_str += "██"
            else:
                fila_str += "  "
        fila_str += "│"
        print(fila_str)
    print("└" + "─" * (tamano_original[1] * 2) + "┘")
    
    return matriz

def mostrar_dedos_consola(dedos):
    """Muestra el estado de los dedos en consola"""
    nombres_dedos = ["Pulgar", "Índice", "Medio", "Anular", "Meñique"]
    print(f"\n✋ ESTADO DE DEDOS: {''.join([str(d) for d in dedos])}")
    for i, (nombre, estado) in enumerate(zip(nombres_dedos, dedos)):
        print(f"   {nombre}: {estado}")

def generar_encabezados(tamano_vector=400):
    """Genera encabezados descriptivos para las columnas"""
    encabezados = []
    
    # Encabezados para el vector de trayectoria (20x20 = 400 elementos)
    for i in range(tamano_vector):
        fila = i // 20
        columna = i % 20
        encabezados.append(f"pixel_{fila:02d}_{columna:02d}")
    
    # Encabezados para los dedos
    nombres_dedos = ["pulgar", "indice", "medio", "anular", "menique"]
    for dedo in nombres_dedos:
        encabezados.append(f"dedo_{dedo}")
    
    return encabezados

def exportar_datos_completos(vector_binario, dedos, nombre_archivo="datos_gesto.csv"):
    """Exporta vector y estado de dedos a CSV con encabezados"""
    # Normalizar vector a 0 y 1
    vector_normalizado = (vector_binario / 255).astype(int)
    
    # Combinar vector y dedos
    datos_completos = np.append(vector_normalizado, dedos)
    
    # Generar encabezados
    encabezados = generar_encabezados(len(vector_normalizado))
    
    # Crear DataFrame con encabezados
    df = pd.DataFrame([datos_completos], columns=encabezados)
    
    # Guardar como CSV
    df.to_csv(nombre_archivo, index=False)
    
    print(f"\nDatos exportados a: {nombre_archivo}")
    print(f"Dimensión total: {len(datos_completos)} elementos")
    print(f"   - Trayectoria (pixels): {len(vector_normalizado)} elementos")
    print(f"   - Dedos: {len(dedos)} elementos")
    
    # Mostrar estructura del archivo
    print(f"\nESTRUCTURA DEL ARCHIVO CSV:")
    print(f"   Columnas 1-{len(vector_normalizado)}: Pixels de la trayectoria (20x20)")
    print(f"   Columnas {len(vector_normalizado)+1}-{len(datos_completos)}: Estado de los dedos")
    
    return datos_completos, encabezados

def mostrar_estructura_datos(vector_binario, dedos, encabezados):
    """Muestra la estructura completa de los datos"""
    print("\n" + "="*60)
    print("ESTRUCTURA COMPLETA DE DATOS")
    print("="*60)
    
    print(f"\nSECCIÓN 1: TRAYECTORIA (PIXELS)")
    print(f"   Columnas: 1 a {len(vector_binario)}")
    print(f"   Formato: {int(np.sqrt(len(vector_binario)))}x{int(np.sqrt(len(vector_binario)))} pixels")
    print(f"   Rango de valores: 0 (blanco) o 1 (negro)")
    
    print(f"\nSECCIÓN 2: CONFIGURACIÓN DE DEDOS")
    print(f"   Columnas: {len(vector_binario)+1} a {len(vector_binario)+len(dedos)}")
    print(f"   Dedos: {list(zip(['Pulgar', 'Índice', 'Medio', 'Anular', 'Meñique'], dedos))}")
    print(f"   Valores: 0 (cerrado) o 1 (abierto)")
    
    print(f"\nEJEMPLO DE DATOS:")
    print(f"   Primera columna: {encabezados[0]} = {vector_binario[0] // 255}")
    print(f"   Última columna: {encabezados[-1]} = {dedos[-1]}")
    print(f"   Total de características: {len(vector_binario) + len(dedos)}")

# Configuración principal
cap = cv2.VideoCapture(0)
print("Iniciando sistema de captura LSM...")
print("La grabación comenzará cuando detecte tu mano")
print("Se detendrá cuando no detecte mano por 1 segundo")

with mp_hands.Hands(model_complexity=1, max_num_hands=1, min_detection_confidence=0.7) as hands:
    ultima_deteccion = time.time()
    pizarra_actual = crear_pizarron([], 'Esperando')
    vector_actual = None
    matriz_actual = None
    datos_completos_actual = None
    encabezados_actual = None
    
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
            print("¡Mano detectada! Comenzando grabación...")
        
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
                    print(f"Grabación terminada. {len(frames_video)} frames capturados")
                    
                    if len(frames_video) > 0 and len(centroides_trayectoria) > 0:
                        print("Procesando datos...")
                        
                        # Estandarizar centroides
                        centroides_20 = estandarizar_frames(centroides_trayectoria, 20)
                        print(f"Centroides estandarizados: {len(centroides_20)} frames")
                        
                        # Obtener frames medios
                        centroides_medios = obtener_frames_medios(centroides_20, 5)
                        print(f"Frames medios seleccionados: {len(centroides_medios)}")
                        
                        # Crear pizarrón
                        pizarra_actual = crear_pizarron(centroides_medios, 'Trayectoria Media')
                        
                        # Convertir a vector binario
                        print("\nConvirtiendo a vector binario...")
                        vector_binario, matriz_binaria = pizarron_a_vector_binario(pizarra_actual)
                        
                        # Obtener estado de dedos predominante
                        if len(estados_dedos_trayectoria) > 0:
                            dedos_predominantes = np.round(np.mean(estados_dedos_trayectoria, axis=0)).astype(int)
                        else:
                            dedos_predominantes = estado_dedos_actual
                        
                        # VISUALIZAR EN CONSOLA
                        print("\n" + "="*60)
                        print("DATOS COMPLETOS DEL GESTO CAPTURADO")
                        print("="*60)
                        
                        # Mostrar vector
                        visualizar_vector_consola(vector_binario)
                        
                        # Mostrar dedos
                        mostrar_dedos_consola(dedos_predominantes)
                        
                        # Exportar datos completos con encabezados
                        datos_completos, encabezados = exportar_datos_completos(vector_binario, dedos_predominantes)
                        
                        # Mostrar estructura completa
                        mostrar_estructura_datos(vector_binario, dedos_predominantes, encabezados)
                        
                        # Guardar para mostrar en ventanas
                        vector_actual = vector_binario
                        matriz_actual = matriz_binaria
                        datos_completos_actual = datos_completos
                        encabezados_actual = encabezados
                        
                        # Mostrar imagen binaria
                        cv2.imshow('Imagen Binaria', matriz_binaria)
                        
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
                
                # Mostrar estado de dedos en pantalla
                dedos_texto = f"Dedos: {''.join([str(d) for d in dedos_actuales])}"
                cv2.putText(frame, dedos_texto, (10, 120), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 0), 2)
        
        # Actualizar estado anterior
        mano_detectada_anteriormente = mano_actualmente_detectada
        
        # Mostrar información en pantalla
        cv2.putText(frame, estado_texto, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, color_estado, 2)
        cv2.putText(frame, f"Frames: {len(frames_video)}", (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1)
        
        if vector_actual is not None:
            info_vector = f"Vector: {vector_actual.shape}"
            cv2.putText(frame, info_vector, (10, 90), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 0), 1)
        
        # Mostrar ventanas
        cv2.imshow('Trayectoria Media', pizarra_actual)
        cv2.imshow('Camara LSM', frame)
        
        if cv2.waitKey(1) & 0xFF == 27:
            break

# Limpieza final
import shutil
if os.path.exists('temp_frames'):
    shutil.rmtree('temp_frames')

cap.release()
cv2.destroyAllWindows()