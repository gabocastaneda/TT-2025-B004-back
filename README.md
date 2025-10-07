# Sistema de Captura y Registro de Gestos Manuales

Este proyecto permite capturar gestos de manos en tiempo real, extraer su trayectoria y el estado de los dedos, y almacenar estos datos estructurados en un archivo CSV, utilizando MediaPipe y OpenCV en Python.

## Descripción

El sistema detecta una mano desde la cámara, extrae la posición del centroide y el estado de los cinco dedos (abierto/cerrado), genera una representación binaria de la trayectoria y registra toda la información junto con la clase del gesto en un CSV. Está diseñado para construir datasets de gestos, útiles para entrenamiento de modelos de clasificación de lenguaje de señas o reconocimiento de gestos.

## Características

- Seguimiento en tiempo real de landmarks con MediaPipe.
- Extracción de trayectoria del centroide y estado de los dedos en cada frame.
- Visualización de las trayectorias y estados en la interfaz.
- Estandarización de los datos y almacenamiento como vectores binarios (trayectoria) más clase y dedos.
- Interfaz por consola para cambiar la clase, guardar gestos y visualizar información relevante.
- Compatible con Python 3.x.

## Instalación

1. Clona este repositorio:
   ```
   git clone https://github.com/tuusuario/tu-repo.git
   cd tu-repo
   ```
2. Instala las dependencias necesarias:
   ```
   pip install opencv-python mediapipe numpy pandas
   ```

## Uso

1. Ejecuta el script principal:
   ```
   python data.py
   ```
2. Siga las instrucciones en consola:
   - C: Cambiar clase actual
   - G: Guardar gesto actual en el CSV
   - ESC: Salir del programa

Los datos se guardan en el archivo `datasetlsm.csv`. Al finalizar, se muestran estadísticas de los gestos registrados.

## Aplicaciones

- Creación de datasets personalizados para reconocimiento automático de gestos/manuales.
- Proyectos de aprendizaje profundo y visión artificial enfocados en el lenguaje de señas mexicano (LSM).
- Pruebas rápidas y prototipado de clasificadores de gestos manuales con trayectorias y estados digitales.

## Créditos

Desarrollado con OpenCV y MediaPipe, inspirado por soluciones y tutoriales de visión artificial y reconocimiento de lenguaje de señas en Python.
