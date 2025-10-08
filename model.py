import pandas as pd
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import joblib

# Cargar dataset
df = pd.read_csv("dataset_lsm.csv")

# --- Modelo 1: Dedos ---
X_dedos = df[["dedo_pulgar", "dedo_indice", "dedo_medio", "dedo_anular", "dedo_menique"]]
y = df["clase"]

knn_dedos = KNeighborsClassifier(n_neighbors=5)
knn_dedos.fit(X_dedos, y)
joblib.dump(knn_dedos, "modelo_dedos.pkl")
print("✅ Modelo de dedos guardado como 'modelo_dedos.pkl'")

# --- Modelo 2: Trayectorias ---
pixel_cols = [c for c in df.columns if c.startswith("pixel_")]
X_pixeles = df[pixel_cols]

knn_trayectoria = KNeighborsClassifier(n_neighbors=5)
knn_trayectoria.fit(X_pixeles, y)
joblib.dump(knn_trayectoria, "modelo_trayectoria.pkl")
print("✅ Modelo de trayectoria guardado como 'modelo_trayectoria.pkl'")

# --- Evaluar modelo de dedos ---
X_train_d, X_test_d, y_train_d, y_test_d = train_test_split(X_dedos, y, test_size=0.2, random_state=42, stratify=y)
knn_dedos = KNeighborsClassifier(n_neighbors=1)
knn_dedos.fit(X_train_d, y_train_d)
y_pred_d = knn_dedos.predict(X_test_d)
acc_dedos = accuracy_score(y_test_d, y_pred_d)
print(f"📏 Precisión del modelo de dedos: {acc_dedos*100:.2f}%")

# --- Evaluar modelo de trayectoria ---
X_train_t, X_test_t, y_train_t, y_test_t = train_test_split(X_pixeles, y, test_size=0.2, random_state=42, stratify=y)
knn_trayectoria = KNeighborsClassifier(n_neighbors=3)
knn_trayectoria.fit(X_train_t, y_train_t)
y_pred_t = knn_trayectoria.predict(X_test_t)
acc_trayectoria = accuracy_score(y_test_t, y_pred_t)
print(f"🧭 Precisión del modelo de trayectoria: {acc_trayectoria*100:.2f}%")

# Guardar ambos modelos después de entrenar y evaluar
joblib.dump(knn_dedos, "modelo_dedos.pkl")
joblib.dump(knn_trayectoria, "modelo_trayectoria.pkl")
print("✅ Modelos guardados correctamente.")
