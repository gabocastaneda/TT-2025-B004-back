import mediapipe as mp
import cv2
import numpy as np
import time
import os

mp_drawing = mp.solutions.drawing_utils
mp_drawing_styles = mp.solutions.drawing_styles
mp_hands = mp.solutions.hands

puntos_palma = [0, 1, 2, 5, 9, 13, 17]
frames_video = []
grabando = False
mano_detectada_anteriormente = False

def centroide(lista_coordenadas):
    coordenadas = np.array(lista_coordenadas)
    centroid = np.mean(coordenadas, axis=0)
    centroid = int(centroid[0]), int(centroid[1])
    return centroid

def extraer_centroide_frame(frame, hands):
    """Extrae el centroide de un frame si se detecta mano"""
    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = hands.process(frame_rgb)
    
    if results.multi_hand_landmarks:
        coordenadas_palma = []
        for handLms in results.multi_hand_landmarks:
            for i in puntos_palma:
                x = int(handLms.landmark[i].x * frame.shape[1])
                y = int(handLms.landmark[i].y * frame.shape[0])
                coordenadas_palma.append([x, y])
        
        if coordenadas_palma:
            return centroide(coordenadas_palma)
    return None

def estandarizar_frames(frames, num_frames_deseado=20):
    """Estandariza los frames a un número fijo usando interpolación"""
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
    """
    Convierte el pizarrón a un vector binario (0 y 255)
    Usamos 20x20 para que sea más fácil visualizar en consola
    """
    # Convertir a escala de grises
    gris = cv2.cvtColor(pizarron, cv2.COLOR_BGR2GRAY)
    
    # Redimensionar al tamaño deseado (más pequeño para consola)
    gris_redim = cv2.resize(gris, tamano_salida)
    
    # Aplicar umbral para binarizar
    _, binaria = cv2.threshold(gris_redim, umbral, 255, cv2.THRESH_BINARY)
    
    # Convertir a vector 1D
    vector_binario = binaria.flatten()
    
    return vector_binario, binaria

def visualizar_vector_consola(vector_binario, tamano_original=(20, 20), caracter_activo='█', caracter_inactivo=' '):
    """
    Visualiza el vector binario en la consola usando caracteres ASCII
    """
    # Reformar a matriz 2D
    matriz = vector_binario.reshape(tamano_original)
    
    print("\n" + "="*60)
    print("VISUALIZACIÓN DEL VECTOR EN CONSOLA")
    print("="*60)
    
    # Mostrar matriz visual
    print(f"\nRepresentación visual ({tamano_original[0]}x{tamano_original[1]}):")
    print("┌" + "─" * (tamano_original[1] * 2) + "┐")
    for i in range(tamano_original[0]):
        fila_str = "│"
        for j in range(tamano_original[1]):
            if matriz[i, j] == 255:
                fila_str += caracter_activo * 2
            else:
                fila_str += caracter_inactivo * 2
        fila_str += "│"
        print(fila_str)
    print("└" + "─" * (tamano_original[1] * 2) + "┘")
    
    # Mostrar estadísticas
    total_elementos = len(vector_binario)
    elementos_activos = np.sum(vector_binario == 255)
    elementos_inactivos = np.sum(vector_binario == 0)
    
    print(f"\nEstadísticas del vector:")
    print(f"   Tamaño total: {total_elementos} elementos")
    print(f"   Píxeles activos (255): {elementos_activos} ({elementos_activos/total_elementos*100:.1f}%)")
    print(f"   Píxeles inactivos (0): {elementos_inactivos} ({elementos_inactivos/total_elementos*100:.1f}%)")
    
    return matriz

def exportar_vector_csv(vector_binario, nombre_archivo="vector_gesto.csv"):
    """
    Exporta el vector a un archivo CSV
    """
    # Normalizar a 0 y 1 para mayor legibilidad
    vector_normalizado = (vector_binario / 255).astype(int)
    
    # Guardar como CSV
    np.savetxt(nombre_archivo, vector_normalizado.reshape(1, -1), delimiter=',', fmt='%d')
    print(f"\n💾 Vector exportado a: {nombre_archivo}")

# Crear carpeta para guardar frames temporales si no existe
if not os.path.exists('temp_frames'):
    os.makedirs('temp_frames')

cap = cv2.VideoCapture(0)
print("Iniciando detección de mano...")
print("La grabación comenzará cuando detecte tu mano")
print("Se detendrá cuando no detecte mano por 1 segundo")

with mp_hands.Hands(model_complexity=1, max_num_hands=1, min_detection_confidence=0.7) as hands:
    ultima_deteccion = time.time()
    pizarra_actual = crear_pizarron([], 'Esperando')
    vector_actual = None
    matriz_actual = None
    
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
            print("¡Mano detectada! Comenzando grabación...")
        
        if grabando:
            if mano_actualmente_detectada:
                ultima_deteccion = time.time()
                frames_video.append(frame.copy())
                estado_texto = f"GRABANDO: {len(frames_video)} frames"
                color_estado = (0, 255, 0)
            else:
                if time.time() - ultima_deteccion > 1.0:
                    grabando = False
                    print(f"Grabación terminada. {len(frames_video)} frames capturados")
                    
                    if len(frames_video) > 0:
                        print("Procesando frames...")
                        
                        centroides_todos = []
                        for frame_vid in frames_video:
                            centroide_frame = extraer_centroide_frame(frame_vid, hands)
                            if centroide_frame is not None:
                                centroides_todos.append(centroide_frame)
                        
                        centroides_20 = estandarizar_frames(centroides_todos, 20)
                        print(f" Centroides estandarizados: {len(centroides_20)} frames")
                        
                        centroides_medios = obtener_frames_medios(centroides_20, 5)
                        print(f"Frames medios seleccionados: {len(centroides_medios)}")
                        
                        pizarra_actual = crear_pizarron(centroides_medios, 'Trayectoria Media')
                        
                        # CONVERTIR PIZARRÓN A VECTOR BINARIO
                        print("\nConvirtiendo pizarrón a vector binario...")
                        vector_binario, matriz_binaria = pizarron_a_vector_binario(pizarra_actual, tamano_salida=(20, 20))
                        
                        # VISUALIZAR EN CONSOLA
                        matriz_visual = visualizar_vector_consola(vector_binario, tamano_original=(20, 20))
                        
                        # Exportar a CSV
                        exportar_vector_csv(vector_binario)
                        
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
        
        # Dibujar landmarks si hay mano detectada
        if results.multi_hand_landmarks:
            for hand_landmarks in results.multi_hand_landmarks:
                mp_drawing.draw_landmarks(
                    frame,
                    hand_landmarks,
                    mp_hands.HAND_CONNECTIONS,
                    mp_drawing_styles.get_default_hand_landmarks_style(),
                    mp_drawing_styles.get_default_hand_connections_style())
                
                coordenadas_palma = []
                for i in puntos_palma:
                    x = int(hand_landmarks.landmark[i].x * frame.shape[1])
                    y = int(hand_landmarks.landmark[i].y * frame.shape[0])
                    coordenadas_palma.append([x, y])
                
                if coordenadas_palma:
                    nx, ny = centroide(coordenadas_palma)
                    cv2.circle(frame, (nx, ny), 4, color_estado, -1)
        
        # Actualizar estado anterior
        mano_detectada_anteriormente = mano_actualmente_detectada
        
        # Mostrar información en pantalla
        cv2.putText(frame, estado_texto, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, color_estado, 2)
        cv2.putText(frame, f"Frames: {len(frames_video)}", (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1)
        
        # Mostrar información del vector si está disponible
        if vector_actual is not None:
            info_vector = f"Vector: {vector_actual.shape}"
            cv2.putText(frame, info_vector, (10, 90), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 0), 1)
        
        # Mostrar pizarra y ventanas
        cv2.imshow('Trayectoria Media', pizarra_actual)
        cv2.imshow('Camara LSM', frame)
        
        if cv2.waitKey(1) & 0xFF == 27:
            break

# Limpiar archivos temporales
import shutil
if os.path.exists('temp_frames'):
    shutil.rmtree('temp_frames')

cap.release()
cv2.destroyAllWindows()