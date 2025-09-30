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
archivo_csv = "dataset_lsm.csv"
clase_actual = "Hola"  

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

def generar_encabezados(tamano_vector=400):
    """Genera encabezados descriptivos para las columnas"""
    encabezados = ["clase"]
    
    # Encabezados para los dedos
    nombres_dedos = ["pulgar", "indice", "medio", "anular", "menique"]
    for dedo in nombres_dedos:
        encabezados.append(f"dedo_{dedo}")
    
    # Encabezados para el vector de trayectoria (20x20 = 400 elementos)
    for i in range(tamano_vector):
        fila = i // 20
        columna = i % 20
        encabezados.append(f"pixel_{fila:02d}_{columna:02d}")
    
    return encabezados

def inicializar_csv():
    """Inicializa el archivo CSV con los encabezados"""
    if not os.path.exists(archivo_csv):
        encabezados = generar_encabezados()
        df = pd.DataFrame(columns=encabezados)
        df.to_csv(archivo_csv, index=False)
        print(f"📁 Archivo CSV creado: {archivo_csv}")
        print(f"📊 Estructura: Clase + 5 dedos + 400 pixels = 406 columnas")
    else:
        print(f"📁 Archivo CSV existente: {archivo_csv}")
        print(f"📝 Se agregarán nuevos registros al final")

def guardar_en_csv(clase, dedos, vector_binario):
    """Guarda un nuevo registro en el CSV"""
    # Normalizar vector a 0 y 1
    vector_normalizado = (vector_binario / 255).astype(int)
    
    # Combinar datos en el orden: clase, dedos, trayectoria
    datos_completos = [clase] + dedos.tolist() + vector_normalizado.tolist()
    
    # Leer CSV existente
    if os.path.exists(archivo_csv):
        df = pd.read_csv(archivo_csv)
    else:
        encabezados = generar_encabezados()
        df = pd.DataFrame(columns=encabezados)
    
    # Agregar nueva fila
    nueva_fila = pd.DataFrame([datos_completos], columns=df.columns)
    df = pd.concat([df, nueva_fila], ignore_index=True)
    
    # Guardar CSV
    df.to_csv(archivo_csv, index=False)
    
    return datos_completos

def mostrar_estructura_datos(clase, dedos, vector_binario):
    """Muestra la estructura completa de los datos"""
    print("\n" + "="*60)
    print("🏗️  ESTRUCTURA COMPLETA DE DATOS")
    print("="*60)
    
    print(f"\n📊 SECCIÓN 1: CLASE")
    print(f"   Columna: 1")
    print(f"   Valor: '{clase}'")
    
    print(f"\n📊 SECCIÓN 2: CONFIGURACIÓN DE DEDOS")
    print(f"   Columnas: 2 a 6")
    print(f"   Dedos: {list(zip(['Pulgar', 'Índice', 'Medio', 'Anular', 'Meñique'], dedos))}")
    print(f"   Valores: 0 (cerrado) o 1 (abierto)")
    
    print(f"\n📊 SECCIÓN 3: TRAYECTORIA (PIXELS)")
    print(f"   Columnas: 7 a 406")
    print(f"   Formato: 20x20 pixels")
    print(f"   Rango de valores: 0 (blanco) o 1 (negro)")
    
    print(f"\n🔍 EJEMPLO DE DATOS:")
    print(f"   Primera columna (clase): '{clase}'")
    print(f"   Columna 2 (dedo_pulgar): {dedos[0]}")
    print(f"   Última columna (pixel_19_19): {vector_binario[-1] // 255}")
    print(f"   Total de características: {1 + len(dedos) + len(vector_binario)}")

def cambiar_clase():
    """Permite cambiar la clase actual mediante input"""
    global clase_actual
    print(f"\n📝 CLASE ACTUAL: '{clase_actual}'")
    nueva_clase = input("   Ingrese nueva clase (o Enter para mantener actual): ").strip()
    if nueva_clase:
        clase_actual = nueva_clase
        print(f"   ✅ Nueva clase establecida: '{clase_actual}'")
    else:
        print(f"   🔄 Manteniendo clase: '{clase_actual}'")
    return clase_actual

# Inicializar sistema
inicializar_csv()
cap = cv2.VideoCapture(0)

print("\n🎮 CONTROLES:")
print("   C - Cambiar clase actual")
print("   G - Guardar gesto actual en CSV")
print("   ESC - Salir del programa")
print(f"\n📝 CLASE INICIAL: '{clase_actual}'")

with mp_hands.Hands(model_complexity=1, max_num_hands=1, min_detection_confidence=0.7) as hands:
    ultima_deteccion = time.time()
    pizarra_actual = crear_pizarron([], 'Esperando')
    vector_actual = None
    matriz_actual = None
    
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
            print("🟢 ¡Mano detectada! Comenzando grabación...")
        
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
                    print(f"🔴 Grabación terminada. {len(frames_video)} frames capturados")
                    
                    if len(frames_video) > 0 and len(centroides_trayectoria) > 0:
                        print("🔄 Procesando datos...")
                        
                        # Estandarizar centroides
                        centroides_20 = estandarizar_frames(centroides_trayectoria, 20)
                        print(f"📊 Centroides estandarizados: {len(centroides_20)} frames")
                        
                        # Obtener frames medios
                        centroides_medios = obtener_frames_medios(centroides_20, 5)
                        print(f"🎯 Frames medios seleccionados: {len(centroides_medios)}")
                        
                        # Crear pizarrón
                        pizarra_actual = crear_pizarron(centroides_medios, 'Trayectoria Media')
                        
                        # Convertir a vector binario
                        vector_binario, matriz_binaria = pizarron_a_vector_binario(pizarra_actual)
                        
                        # Obtener estado de dedos predominante
                        if len(estados_dedos_trayectoria) > 0:
                            dedos_predominantes = np.round(np.mean(estados_dedos_trayectoria, axis=0)).astype(int)
                        else:
                            dedos_predominantes = estado_dedos_actual
                        
                        # Guardar en CSV automáticamente
                        datos_guardados = guardar_en_csv(clase_actual, dedos_predominantes, vector_binario)
                        
                        print(f"\n💾 Gesto guardado en CSV:")
                        print(f"   Clase: '{clase_actual}'")
                        print(f"   Dedos: {dedos_predominantes}")
                        print(f"   Trayectoria: {len(vector_binario)} pixels")
                        
                        # Mostrar estructura
                        mostrar_estructura_datos(clase_actual, dedos_predominantes, vector_binario)
                        
                        # Guardar para mostrar en ventanas
                        vector_actual = vector_binario
                        matriz_actual = matriz_binaria
                        
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
        
        # Mostrar información en pantalla
        cv2.putText(frame, estado_texto, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, color_estado, 2)
        cv2.putText(frame, f"Frames: {len(frames_video)}", (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1)
        cv2.putText(frame, f"Clase: {clase_actual}", (10, 90), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 0), 2)
        
        if estado_dedos_actual is not None:
            dedos_texto = f"Dedos: {''.join([str(d) for d in estado_dedos_actual])}"
            cv2.putText(frame, dedos_texto, (10, 120), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 0), 2)
        
        if vector_actual is not None:
            info_vector = f"Vector: {vector_actual.shape}"
            cv2.putText(frame, info_vector, (10, 150), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 0), 1)
        
        # Mostrar ventanas
        cv2.imshow('Trayectoria Media', pizarra_actual)
        cv2.imshow('Camara LSM', frame)
        
        # Manejo de teclas
        key = cv2.waitKey(1) & 0xFF
        if key == 27:  # ESC
            break
        elif key == ord('c') or key == ord('C'):
            cambiar_clase()
        elif key == ord('g') or key == ord('G'):
            if vector_actual is not None and estado_dedos_actual is not None:
                datos_guardados = guardar_en_csv(clase_actual, estado_dedos_actual, vector_actual)
                print(f"💾 Gesto guardado manualmente - Clase: '{clase_actual}'")
        
        # Actualizar estado anterior
        mano_detectada_anteriormente = mano_actualmente_detectada

# Estadísticas finales
if os.path.exists(archivo_csv):
    df = pd.read_csv(archivo_csv)
    print(f"\n📈 ESTADÍSTICAS FINALES:")
    print(f"   Total de gestos guardados: {len(df)}")
    print(f"   Clases registradas: {df['clase'].unique().tolist()}")
    for clase in df['clase'].unique():
        count = len(df[df['clase'] == clase])
        print(f"     - '{clase}': {count} gestos")

print("👋 Programa terminado")
cap.release()
cv2.destroyAllWindows()