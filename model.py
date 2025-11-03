import pandas as pd
import numpy as np
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.preprocessing import LabelEncoder, StandardScaler
import joblib
import ast
import seaborn as sns
import matplotlib.pyplot as plt
import os

# Parámetros base
N_NEIGHBORS = 7
TEST_SIZE = 0.3
RANDOM_STATE = 0

# Crear directorio para matrices si no existe
MATRICES_DIR = "matrices_confusion"
if not os.path.exists(MATRICES_DIR):
    os.makedirs(MATRICES_DIR)

def preparar_datos(csv_path="dataset_lsm.csv"):
    print("Cargando datos...")
    df = pd.read_csv(csv_path)

    print(f"Datos cargados: {len(df)} registros")
    print(f"Clases: {df['clase'].unique()}")

    # Extraer características de píxeles
    X_pixels = df.iloc[:, 3:].values

    # Procesar secuencias de dedos
    secuencias_left, secuencias_right = [], []

    for idx in range(len(df)):
        # Secuencia izquierda
        try:
            sec_left = df.iloc[idx]["secuencia_dedos_centrales_left"]
            if isinstance(sec_left, str):
                sec_left = ast.literal_eval(sec_left)
            secuencias_left.append(np.array(sec_left).flatten())
        except Exception:
            secuencias_left.append(np.zeros(25))

        # Secuencia derecha
        try:
            sec_right = df.iloc[idx]["secuencia_dedos_centrales_right"]
            if isinstance(sec_right, str):
                sec_right = ast.literal_eval(sec_right)
            secuencias_right.append(np.array(sec_right).flatten())
        except Exception:
            secuencias_right.append(np.zeros(25))

    secuencias_left = np.array(secuencias_left)
    secuencias_right = np.array(secuencias_right)

    # Combinar todas las características
    X_combined = np.hstack([secuencias_left, secuencias_right, X_pixels])

    # Estandarizar datos
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X_combined)

    y = df["clase"].values

    print(f"Características combinadas: {X_combined.shape}")
    print(f"Etiquetas: {y.shape}")

    return X_scaled, y, scaler

def plot_confusion_matrix(y_true, y_pred, classes, title="Matriz de Confusión", save_path=None):
    cm = confusion_matrix(y_true, y_pred)
    plt.figure(figsize=(10, 8))
    sns.heatmap(
        cm,
        annot=True,
        fmt="d",
        cmap="Blues",
        xticklabels=classes,
        yticklabels=classes,
    )
    plt.title(title, fontsize=16, fontweight="bold")
    plt.xlabel("Predicción")
    plt.ylabel("Valor real")
    plt.xticks(rotation=45)
    plt.yticks(rotation=0)
    plt.tight_layout()
    
    # Guardar la imagen si se proporciona un path
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Matriz de confusión guardada en: {save_path}")
    
    plt.show()

def entrenar_modelos(X, y):
    le = LabelEncoder()
    y_encoded = le.fit_transform(y)

    X_train, X_test, y_train, y_test = train_test_split(X, y_encoded, test_size=TEST_SIZE, random_state=RANDOM_STATE, stratify=y_encoded, shuffle=True)

    modelos = {
        "KNN": KNeighborsClassifier(n_neighbors=N_NEIGHBORS, weights="distance"),
        "RandomForest": RandomForestClassifier(n_estimators=200, random_state=RANDOM_STATE),
        "SVM": SVC(kernel="rbf", C=10, gamma="scale", probability=True, random_state=RANDOM_STATE),
        "LogisticRegression": LogisticRegression()
    }

    resultados = {}

    for nombre, modelo in modelos.items():
        print(f"\n- 6Entrenando modelo: {nombre}...")
        modelo.fit(X_train, y_train)
        y_pred = modelo.predict(X_test)

        acc = accuracy_score(y_test, y_pred)
        print(f"Precisión: {acc:.4f}")
        print("Reporte de clasificación:")
        print(classification_report(y_test, y_pred, target_names=le.classes_))

        resultados[nombre] = {
            "modelo": modelo,
            "accuracy": acc,
            "predicciones": y_pred,
        }

        # Matriz de confusión con guardado automático
        nombre_archivo = f"{MATRICES_DIR}/matriz_confusion_{nombre}.png"
        plot_confusion_matrix(
            y_test, y_pred, le.classes_, 
            title=f"Matriz de Confusión - {nombre}",
            save_path=nombre_archivo
        )

    # Escoger mejor modelo
    mejor_nombre = max(resultados, key=lambda k: resultados[k]["accuracy"])
    mejor_modelo = resultados[mejor_nombre]["modelo"]
    mejor_acc = resultados[mejor_nombre]["accuracy"]

    print(f"\nMejor modelo: {mejor_nombre} (Precisión = {mejor_acc:.4f})")

    return mejor_modelo, le, mejor_nombre

def guardar_modelo(model, label_encoder, scaler, nombre_modelo, archivo="modelo_final.pkl"):
    modelo_guardar = {
        "model": model,
        "label_encoder": label_encoder,
        "scaler": scaler,
        "classes": label_encoder.classes_,
        "nombre_modelo": nombre_modelo,
    }

    joblib.dump(modelo_guardar, archivo)
    print(f"\nModelo guardado como: {archivo}")
    print(f"Tipo de modelo: {nombre_modelo}")
    print(f"Clases: {list(label_encoder.classes_)}")

def main():
    try:
        X, y, scaler = preparar_datos()
        if len(X) == 0:
            print("No hay datos para entrenar")
            return

        mejor_modelo, le, nombre_modelo = entrenar_modelos(X, y)
        guardar_modelo(mejor_modelo, le, scaler, nombre_modelo)

        print("Entrenamiento completado exitosamente")
        print(f"Matrices de confusión guardadas en la carpeta: {MATRICES_DIR}")

    except Exception as e:
        print(f"Error durante el entrenamiento: {e}")
        raise


if __name__ == "__main__":
    main()