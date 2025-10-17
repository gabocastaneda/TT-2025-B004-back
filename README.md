# Lenguaje de Señas Mexicanas (LSM) - Sistema de Reconocimiento

## Descripción

Este proyecto implementa un sistema de reconocimiento de gestos para Lenguaje de Señas Mexicanas (LSM) utilizando visión por computadora y aprendizaje automático. El sistema captura gestos de manos a través de una cámara web, procesa la información de trayectoria y estado de los dedos, y utiliza un modelo KNN para clasificar los gestos en tiempo real.

## Características

- **Captura de Datos**: Detecta y registra el estado de los dedos (abierto/cerrado) y la trayectoria del centroide de la mano.
- **Procesamiento de Video**: Utiliza MediaPipe para detectar landmarks de manos y OpenCV para procesar imágenes.
- **Entrenamiento**: Entrena un modelo K-Nearest Neighbors (KNN) con datos almacenados en un archivo CSV.
- **Inferencia en Tiempo Real**: Clasifica gestos en tiempo real con confianza calculada.
- **Interfaz Visual**: Muestra la cámara en vivo, la trayectoria de la mano y el estado del reconocimiento.

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

## Estructura del Proyecto

- `data.py`: Captura gestos, procesa datos (trayectoria y estado de dedos) y los guarda en un archivo CSV (`dataset_lsm.csv`).
- `model.py`: Entrena un modelo KNN con los datos del CSV y genera una matriz de confusión para evaluación.
- `inferencia.py`: Realiza predicciones en tiempo real utilizando el modelo entrenado.
- `dataset_lsm.csv`: Almacena los datos capturados (clase, secuencia de dedos y vector binario de trayectoria).
- `knn_model.pkl`: Archivo donde se guarda el modelo KNN entrenado.

## Uso

1. **Captura de Datos**:

   - Ejecuta `data.py`:
     ```bash
     python data.py
     ```
   - Controles:
     - `C`: Cambiar la clase actual (etiqueta del gesto).
     - `G`: Guardar el gesto actual en el CSV (manualmente).
     - `ESC`: Salir del programa.
   - Realiza gestos frente a la cámara. El sistema detecta automáticamente el inicio y fin de los gestos y los guarda en el CSV.

2. **Entrenamiento del Modelo**:

   - Ejecuta `model.py` para entrenar el modelo KNN:
     ```bash
     python model.py
     ```
   - El modelo se guarda como `knn_model.pkl`.
   - Se muestra una matriz de confusión y métricas de evaluación si hay suficientes datos.

3. **Inferencia en Tiempo Real**:
   - Ejecuta `inferencia.py`:
     ```bash
     python inferencia.py
     ```
   - Controles:
     - `R`: Reiniciar la captura para una nueva predicción.
     - `ESPACIO`: Forzar una predicción con los datos actuales.
     - `ESC`: Salir del programa.
   - El sistema muestra la predicción del gesto y su confianza en tiempo real.

## Flujo de Trabajo

1. **Captura**: Usa `data.py` para grabar gestos y etiquetarlos, generando un dataset en `dataset_lsm.csv`.
2. **Entrenamiento**: Usa `model.py` para entrenar el modelo KNN con los datos capturados.
3. **Inferencia**: Usa `inferencia.py` para clasificar gestos en tiempo real con el modelo entrenado.

## Detalles Técnicos

- **Detección de Manos**: Utiliza MediaPipe Hands para detectar landmarks de manos con alta precisión (mínimo 0.95 de confianza).
- **Características**:
  - Estado de los 5 dedos (0 = cerrado, 1 = abierto) en 5 frames centrales (25 valores).
  - Trayectoria de la mano convertida a un vector binario de 400 píxeles (20x20).
- **Modelo**: KNN con 11 vecinos (configurable en `model.py`).
- **Estandarización**: Los datos se estandarizan a 30 frames, y se seleccionan los 30 frames centrales para la inferencia.
- **Visualización**: Muestra la trayectoria de la mano en un pizarrón de 200x200 píxeles y el estado de los dedos en tiempo real.

## Limitaciones

- Requiere buena iluminación y fondo limpio para una detección óptima.
- Solo detecta una mano a la vez (`max_num_hands=1`), hasta el momento.
- La precisión depende de la cantidad y calidad de los datos en `dataset_lsm.csv`.
- El modelo KNN puede no ser óptimo para datasets muy grandes o gestos complejos.

## Contribuciones

Las contribuciones son bienvenidas. Por favor, crea un _issue_ o _pull request_ en el repositorio para sugerencias o mejoras.

## Licencia

Este proyecto está bajo la licencia MIT. Consulta el archivo `LICENSE` para más detalles.
