import pandas as pd
import numpy as np
import ast
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import joblib
import os
import seaborn as sns
import matplotlib.pyplot as plt

def entrenar_modelo_knn(archivo_csv="dataset_lsm.csv", modelo_salida="knn_model.pkl", n_vecinos=11):
    """
    Entrena un modelo KNN con los datos del CSV y lo guarda en un archivo
    """
    print("ENTRENANDO MODELO KNN...")
    
    if not os.path.exists(archivo_csv):
        print(f"No existe el archivo CSV: {archivo_csv}")
        return None
    
    # Cargar datos
    df = pd.read_csv(archivo_csv)
    print(f"Datos cargados: {len(df)} muestras")
    print(f"Clases: {df['clase'].unique().tolist()}")
    
    if len(df) < n_vecinos:
        print(f"No hay suficientes datos. Se necesitan al menos {n_vecinos} muestras")
        return None
    
    # Preparar características (X) y etiquetas (y)
    X = []
    y = df['clase'].values
    
    for _, fila in df.iterrows():
        secuencia_dedos = ast.literal_eval(fila['secuencia_dedos_centrales'])
        vector_dedos = np.array(secuencia_dedos).flatten()
        
        pixel_cols = [col for col in df.columns if col.startswith('pixel_')]
        vector_trayectoria = fila[pixel_cols].values
        
        caracteristicas = np.concatenate([vector_dedos, vector_trayectoria])
        X.append(caracteristicas)
    
    X = np.array(X)
    
    print(f"Características: {X.shape[1]} dimensiones")
    print(f"Muestras: {X.shape[0]}")
    
    # Dividir datos para evaluación
    if len(df) > 10:
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0, stratify=y, shuffle=True) 
        
        # Entrenar modelo KNN
        knn = KNeighborsClassifier(n_neighbors=n_vecinos)
        knn.fit(X_train, y_train)
        
        # Evaluar modelo
        y_pred = knn.predict(X_test)
        accuracy = accuracy_score(y_test, y_pred)
        
        print(f"Modelo KNN entrenado exitosamente")
        print(f"Precisión en test: {accuracy:.3f}")
        print(f"Vecinos (k): {n_vecinos}")
        print(f"Distribución de clases:")
        
        clases_unicas, conteos = np.unique(y, return_counts=True)
        for clase, conteo in zip(clases_unicas, conteos):
            print(f"   - {clase}: {conteo} muestras")
        
        # Reporte de clasificación detallado
        print("\nReporte de clasificación:")
        print(classification_report(y_test, y_pred))
        
        # Matriz de confusión
        cm = confusion_matrix(y_test, y_pred, labels=clases_unicas)
        plt.figure(figsize=(10, 8))
        sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", xticklabels=clases_unicas, yticklabels=clases_unicas)
        plt.title("Matriz de Confusión - KNN")
        plt.xlabel("Predicción")
        plt.ylabel("Real")
        plt.tight_layout()
        plt.show()
        plt.savefig()
        
    else:
        # Entrenar con todos los datos si hay pocas muestras
        knn = KNeighborsClassifier(n_neighbors=n_vecinos)
        knn.fit(X, y)
        print(f"Modelo KNN entrenado con {len(X)} muestras (sin división train/test)")
    
    # Guardar modelo
    joblib.dump(knn, modelo_salida)
    print(f"Modelo guardado en: {modelo_salida}")
    
    return knn

def verificar_modelo(modelo_path="knn_model.pkl"):
    """Verifica que el modelo se haya guardado correctamente"""
    if os.path.exists(modelo_path):
        modelo = joblib.load(modelo_path)
        print(f"Modelo verificado: {type(modelo).__name__}")
        print(f"Clases: {modelo.classes_.tolist()}")
        print(f"Número de características: {modelo.n_features_in_}")
        return True
    else:
        print("No se pudo encontrar el modelo guardado")
        return False

if __name__ == "__main__":
    # Entrenar modelo
    modelo = entrenar_modelo_knn(n_vecinos=11)
    
    # Verificar que se guardó correctamente
    if modelo is not None:
        verificar_modelo()
