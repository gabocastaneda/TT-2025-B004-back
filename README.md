# Sistema de Captura y Registro de Gestos Manuales

Este proyecto permite capturar gestos de manos en tiempo real, extraer su trayectoria y el estado de los dedos, y almacenar estos datos estructurados en un archivo CSV, utilizando MediaPipe y OpenCV en Python.

## Descripción

El sistema detecta una mano desde la cámara, extrae la posición del centroide y el estado de los cinco dedos (abierto/cerrado), genera una representación binaria de la trayectoria y registra toda la información junto con la clase del gesto en un CSV. Está diseñado para construir datasets de gestos, útiles para entrenamiento de modelos de clasificación de lenguaje de señas o reconocimiento de gestos.

## Características

- Seguimiento en tiempo real de landmarks con MediaPipe.
- Extracción de trayectoria del centroide y estado de los dedos en cada frame.
- Visualización de trayectorias y estados en la interfaz.
- Estandarización de los datos y almacenamiento en vectores binarios.
- Compatible con entrenamiento de modelos de machine learning (KNN).
- Compatible con inferencia en tiempo real desde cámara.

## Entrenamiento del Modelo

El entrenamiento se realiza mediante **K-Nearest Neighbors (KNN)** utilizando los datos registrados previamente.  
El script `model.py` se encarga de leer el archivo CSV, procesar los vectores de características (estado de los dedos y trayectoria binaria), entrenar el modelo y guardar el resultado.

**Pasos:**

1. Asegúrate de tener un archivo `dataset_lsm.csv` generado con capturas previas.
2. Ejecuta el script de entrenamiento:
   ```bash
   python model.py
   ```
3. Durante la ejecución:
   - Se cargan los datos desde CSV.
   - Se separan características (X) y etiquetas (y).
   - Se divide el dataset en entrenamiento y prueba (80/20) si hay suficientes muestras.
   - Se entrena un modelo KNN con los vecinos definidos (por defecto 11).
   - Se genera un reporte de precisión, matriz de confusión y se guarda el modelo en `knn_model.pkl`.

**Salida esperada:**

- Reporte de clasificación con métricas de rendimiento.
- Visualización de matriz de confusión.
- Archivo `knn_model.pkl` con el modelo entrenado listo para usar en inferencia.

## Inferencia en Tiempo Real

El script `inferencia.py` implementa un sistema de predicción en tiempo real utilizando el modelo entrenado (`knn_model.pkl`). Captura los gestos desde la cámara y predice su clase automáticamente.

**Flujo principal:**

1. Se carga el modelo KNN preentrenado.
2. Se inicia la cámara (MediaPipe Hands).
3. Cuando una mano es detectada:
   - Se extraen los landmarks.
   - Se calculan el centroide y el estado de los dedos.
   - Se estandarizan los frames a 30 muestras.
   - Se genera una imagen binaria de trayectoria.
   - Se combinan todas las características y se realiza la predicción con KNN.

**Controles de teclado:**

- **R:** Reiniciar la inferencia.
- **Espacio:** Forzar una predicción manual.
- **Esc:** Salir del programa.

**Salida en consola y ventana:**

- Predicción del gesto y nivel de confianza.
- Visualización de trayectoria, estados de dedos y resultados en pantalla.

El sistema espera al menos 5 frames válidos antes de realizar una predicción.  
Si la confianza es baja (<0.7), el color cambia para indicar incertidumbre.
