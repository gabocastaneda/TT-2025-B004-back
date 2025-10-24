# Lenguaje de Señas Mexicanas (LSM) - Sistema de Reconocimiento

## Descripción

Este proyecto implementa un sistema de reconocimiento de gestos para Lenguaje de Señas Mexicanas (LSM) utilizando visión por computadora y aprendizaje automático. El sistema captura gestos de manos a través de una cámara web, procesa la información de trayectoria y estado de los dedos, y utiliza un modelo KNN para clasificar los gestos en tiempo real.

## Características

- **Captura de datos**: Detecta y registra el estado de los dedos (abierto/cerrado) y la trayectoria del centroide de la mano.
- **Procesamiento de video**: Utiliza MediaPipe para detectar landmarks de manos y OpenCV para procesar imágenes.
- **Entrenamiento**: Entrena un modelo K-Nearest Neighbors (KNN) con datos almacenados en un archivo CSV.
- **Inferencia en tiempo real**: Clasifica gestos en tiempo real con confianza calculada.
- **Interfaz visual**: Muestra la cámara en vivo, la trayectoria de la mano y el estado del reconocimiento.

## Requisitos

- Python 3.10.11
- Cámara web funcional
- Sistema operativo: Windows, macOS o Linux

## Instalación

1. Clona o descarga el repositorio.
2. Instala las dependencias:
   ```bash
   pip install -r requirements.txt
   ```
3. Asegúrate de tener una cámara web conectada.

## Estructura del proyecto

- `data.py`: Captura gestos, procesa datos (trayectoria y estado de dedos) y los guarda en un archivo CSV (`dataset_lsm.csv`).
- `model.py`: Entrena un modelo KNN con los datos del CSV y genera una matriz de confusión para evaluación.
- `inferencia.py`: Realiza predicciones en tiempo real utilizando el modelo entrenado.
- `dataset_lsm.csv`: Almacena los datos capturados (clase, secuencia de dedos y vector binario de trayectoria).
- `knn_gestos_model.pkl`: Archivo donde se guarda el modelo KNN entrenado.

## Uso

1. **Captura de datos**:

   - Ejecuta `data.py`:
     ```bash
     python data.py
     ```
   - Controles:
     - `C`: Cambiar la clase actual (etiqueta del gesto).
     - `G`: Guardar el gesto actual en el CSV (manualmente).
     - `ESC`: Salir del programa.
   - Realiza gestos frente a la cámara. El sistema detecta automáticamente el inicio y fin de los gestos y los guarda en el CSV.

2. **Entrenamiento del modelo**:

   - Ejecuta `model.py` para entrenar el modelo KNN:
     ```bash
     python model.py
     ```
   - El modelo se guarda como `knn_gestos_model.pkl`.
   - Se muestra una matriz de confusión y métricas de evaluación si hay suficientes datos.

3. **Inferencia en tiempo real**:
   - Ejecuta `inferencia.py`:
     ```bash
     python inferencia.py
     ```
   - Controles:
     - `R`: Reiniciar la captura para una nueva predicción.
     - `ESPACIO`: Forzar una predicción con los datos actuales.
     - `ESC`: Salir del programa.
   - El sistema muestra la predicción del gesto y su confianza en tiempo real.

## Flujo de trabajo

1. **Captura**: Usa `data.py` para grabar gestos y etiquetarlos, generando un dataset en `dataset_lsm.csv`.
2. **Entrenamiento**: Usa `model.py` para entrenar el modelo KNN con los datos capturados.
3. **Inferencia**: Usa `inferencia.py` para clasificar gestos en tiempo real con el modelo entrenado.

## Limitaciones

- Requiere buena iluminación y fondo limpio para una detección óptima.
- La precisión depende de la cantidad y calidad de los datos en `dataset_lsm.csv`.

## Contribuciones

Las contribuciones son bienvenidas. Por favor, crea un _issue_ o _pull request_ en el repositorio para sugerencias o mejoras.
