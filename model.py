import pandas as pd
import numpy as np
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report
from sklearn.preprocessing import LabelEncoder
import joblib
import ast

N_NEIGHBORS = 11

def preparar_datos(csv_path="dataset_lsm.csv"):
    """Prepara los datos para el entrenamiento"""
    print("Cargando datos...")
    df = pd.read_csv(csv_path)
    
    # Verificar estructura
    print(f"Datos cargados: {len(df)} registros")
    print(f"Clases: {df['clase'].unique()}")
    
    # Preparar características (X)
    # Extraer características de píxeles (columnas 3 en adelante)
    X_pixels = df.iloc[:, 3:].values
    
    # Extraer y procesar secuencias de dedos
    secuencias_left = []
    secuencias_right = []
    
    for idx in range(len(df)):
        # Procesar secuencia izquierda
        sec_left_str = df.iloc[idx]['secuencia_dedos_centrales_left']
        try:
            if isinstance(sec_left_str, str):
                sec_left = ast.literal_eval(sec_left_str)
            else:
                sec_left = sec_left_str
            # Aplanar la secuencia (5 frames × 5 dedos = 25 características)
            secuencias_left.append(np.array(sec_left).flatten())
        except:
            secuencias_left.append(np.zeros(25))
        
        # Procesar secuencia derecha
        sec_right_str = df.iloc[idx]['secuencia_dedos_centrales_right']
        try:
            if isinstance(sec_right_str, str):
                sec_right = ast.literal_eval(sec_right_str)
            else:
                sec_right = sec_right_str
            secuencias_right.append(np.array(sec_right).flatten())
        except:
            secuencias_right.append(np.zeros(25))
    
    # Convertir a arrays
    secuencias_left = np.array(secuencias_left)
    secuencias_right = np.array(secuencias_right)
    
    # Combinar todas las características
    X_combined = np.hstack([secuencias_left, secuencias_right, X_pixels])
    
    # Preparar etiquetas (y)
    y = df['clase'].values
    
    print(f"Características combinadas: {X_combined.shape}")
    print(f"Etiquetas: {y.shape}")
    
    return X_combined, y

def entrenar_modelo_knn(X, y, n_neighbors=N_NEIGHBORS, test_size=0.2, random_state=0, shuffle=True):
    """Entrena el modelo KNN"""
    print("\nEntrenando modelo KNN...")
    
    # Codificar etiquetas
    le = LabelEncoder()
    y_encoded = le.fit_transform(y)
    
    # Dividir datos
    X_train, X_test, y_train, y_test = train_test_split(
        X, y_encoded, test_size=test_size, random_state=random_state, stratify=y_encoded
    )
    
    print(f"Training set: {X_train.shape}")
    print(f"Test set: {X_test.shape}")
    
    # Entrenar KNN
    knn = KNeighborsClassifier(n_neighbors=N_NEIGHBORS, weights='distance')
    knn.fit(X_train, y_train)
    
    # Evaluar
    y_pred = knn.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    
    print(f"\nPrecisión del modelo: {accuracy:.4f}")
    print("\nReporte de clasificación:")
    print(classification_report(y_test, y_pred, target_names=le.classes_))
    
    return knn, le, X_train, X_test, y_train, y_test

def guardar_modelo(model, label_encoder, nombre_archivo="knn_gestos_model.pkl"):
    """Guarda el modelo entrenado"""
    modelo_guardar = {
        'model': model,
        'label_encoder': label_encoder,
        'classes': label_encoder.classes_
    }
    
    joblib.dump(modelo_guardar, nombre_archivo)
    print(f"\nModelo guardado como: {nombre_archivo}")
    print(f"Clases guardadas: {list(label_encoder.classes_)}")

def main():
    try:
        # Preparar datos
        X, y = preparar_datos()
        
        if len(X) == 0:
            print("No hay datos para entrenar")
            return
        
        # Entrenar modelo
        knn, le, X_train, X_test, y_train, y_test = entrenar_modelo_knn(X, y, n_neighbors=N_NEIGHBORS)
        
        # Guardar modelo
        guardar_modelo(knn, le)
        
        print("\nEntrenamiento completado exitosamente\n")
        
    except Exception as e:
        print(f"Error durante el entrenamiento: {e}\n")
        raise

if __name__ == "__main__":
    main()