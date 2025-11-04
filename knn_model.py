import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import confusion_matrix, accuracy_score, classification_report
import ast
import matplotlib.pyplot as plt
import seaborn as sns
from collections import Counter
import os

CSV_PATH = 'dataset_lsm.csv'
KNN_DIR = 'knn_matrices'

if not os.path.exists(KNN_DIR):
    os.makedirs(KNN_DIR)

# -- Implemantación manual de KNN --
class KNN:
    def __init__(self, k):
        self.k = k

    def fit(self, X, y):
        self.X_train = X
        self.y_train = y

    def predict(self, X):
        return np.array([self._predecir(x) for x in X])

    def _predecir(self, x):
        distancias = np.linalg.norm(self.X_train - x, axis=1)
        vecinos_idx = np.argsort(distancias)[:self.k]
        etiquetas_vecinas = self.y_train[vecinos_idx]
        etiqueta_mas_comun = Counter(etiquetas_vecinas).most_common(1)[0][0]
        return etiqueta_mas_comun

def preparar_datos(csv_path):
    print('Cargando datos')
    df = pd.read_csv(csv_path)

    print(f'Datos cargados: {len(df)} registros')
    print(f"Clases: {df['clase'].unique()}")

    # Extraer caracteristicias de pixeles
    X_pixels = df.iloc[:, 3:].values

    # Procesar secuencias de dedos
    secuencias_left, secuencias_right = [], []

    for idx in range(len(df)):
        # Secuencia izquierda
        try:
            sec_left = df.iloc[idx]['secuencia_dedos_centrales_left']
            if isinstance(sec_left, str):
                sec_left = ast.literal_eval(sec_left)
            secuencias_left.append(np.array(sec_left).flatten())
        except Exception:
            secuencias_left.append(np.zeros(25))

        # Secuencia derecha
        try:
            sec_right = df.iloc[idx]['secuencia_dedos_centrales_right']
            if isinstance(sec_right, str):
                sec_right = ast.literal_eval(sec_right)
            secuencias_right.append(np.array(sec_right).flatten())
        except Exception:
            secuencias_right.append(np.zeros(25))

    secuencias_left = np.array(secuencias_left)
    secuencias_right = np.array(secuencias_right)

    # Combinar todas las caracteristicas
    X_combined = np.hstack([secuencias_left, secuencias_right, X_pixels])

    # Estandarizar los datos
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X_combined)

    y = df['clase'].values

    print(f'Caracteristicas totales: {X_combined.shape}')
    print(f'Etiquetas: {y.shape}')

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

if __name__ == "__main__":
    X, y, scaler = preparar_datos(CSV_PATH)
    resultados = []

    for k in range(5, 21, 2):
        modelo = KNN(k=k)  
        modelo.fit(X, y)
        y_pred = modelo.predict(X)
        acc = accuracy_score(y, y_pred)
        print(f'k={k} | Exactitud: {acc:.4f}')
        resultados.append((k, acc))

    # Imprime el mejor k encontrado
    mejor_k, mejor_acc = max(resultados, key=lambda x: x[1])
    print(f"\nMejor k: {mejor_k} con exactitud {mejor_acc:.4f}")

    # Si quieres ver el reporte y matriz de confusión para el mejor k:
    modelo = KNN(k=mejor_k)
    modelo.fit(X, y)
    y_pred = modelo.predict(X)

    print('Reporte de clasificación: ')
    print(classification_report(y, y_pred))
    plot_confusion_matrix(y, y_pred, classes=np.unique(y))
