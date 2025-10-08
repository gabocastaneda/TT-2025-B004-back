import pandas as pd
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.ensemble import VotingClassifier
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import joblib
import numpy as np

print("="*70)
print("ENTRENAMIENTO DE MODELOS LSM - VERSIÓN ESCALABLE")
print("="*70)

# Cargar dataset
try:
    df = pd.read_csv("dataset_lsm.csv")
    print(f"\n✅ Dataset cargado: {len(df)} muestras")
except:
    print("\n❌ Error: No se pudo cargar dataset_lsm.csv")
    print("   Ejecuta data.py primero para generar el dataset")
    exit()

print(f"Clases: {sorted(df['clase'].unique())}")
print(f"Total de clases: {len(df['clase'].unique())}")

# Verificar distribución
distribucion = df['clase'].value_counts().sort_index()
print(f"\nDistribución de muestras por clase:")
print(distribucion)

if len(distribucion.unique()) > 1:
    print(f"\n⚠️ ADVERTENCIA: Dataset desbalanceado")
    print(f"   Mínimo: {distribucion.min()} muestras")
    print(f"   Máximo: {distribucion.max()} muestras")
    print("   Se recomienda balancear el dataset")

# Preparar datos
X_dedos = df[["dedo_pulgar", "dedo_indice", "dedo_medio", "dedo_anular", "dedo_menique"]]
pixel_cols = [c for c in df.columns if c.startswith("pixel_")]
X_pixeles = df[pixel_cols]
y = df["clase"]

print(f"\nCaracterísticas:")
print(f"   - Dedos: {X_dedos.shape[1]} features")
print(f"   - Píxeles: {X_pixeles.shape[1]} features")

# CRÍTICO: Verificar y normalizar si es necesario
valores_max = X_pixeles.max().max()
valores_min = X_pixeles.min().min()

print(f"\nRango de valores en píxeles: [{valores_min}, {valores_max}]")

if valores_max > 1.5:  # Si hay valores > 1.5, están en 0-255
    print("⚠️  Píxeles en rango 0-255, normalizando a 0-1...")
    X_pixeles = X_pixeles / 255.0
    print("✅ Normalización completada")
else:
    print("✅ Píxeles ya normalizados (0-1)")

# === MODELO 1: DEDOS ===
print("\n" + "="*70)
print("1️⃣ ENTRENAMIENTO MODELO DE DEDOS")
print("="*70)

X_train_d, X_test_d, y_train_d, y_test_d = train_test_split(
    X_dedos, y, test_size=0.2, random_state=42, stratify=y
)

print(f"\nConjunto de entrenamiento: {len(X_train_d)} muestras")
print(f"Conjunto de prueba: {len(X_test_d)} muestras")

# Búsqueda de mejores hiperparámetros
print("\nBuscando mejores hiperparámetros...")

param_grid_dedos = {
    'n_neighbors': [1, 3, 5, 7],
    'weights': ['uniform', 'distance'],
    'metric': ['euclidean', 'manhattan']
}

grid_dedos = GridSearchCV(
    KNeighborsClassifier(),
    param_grid_dedos,
    cv=5,
    scoring='accuracy',
    n_jobs=-1,
    verbose=1
)

grid_dedos.fit(X_train_d, y_train_d)

print(f"\n✅ Mejores parámetros encontrados:")
for param, valor in grid_dedos.best_params_.items():
    print(f"   - {param}: {valor}")

print(f"\nPrecisión en validación cruzada: {grid_dedos.best_score_*100:.2f}%")

# Evaluar en test
knn_dedos = grid_dedos.best_estimator_
y_pred_d = knn_dedos.predict(X_test_d)
acc_dedos = accuracy_score(y_test_d, y_pred_d)

print(f"Precisión en conjunto de prueba: {acc_dedos*100:.2f}%")

print("\nReporte de clasificación (Dedos):")
print(classification_report(y_test_d, y_pred_d, zero_division=0))

# === MODELO 2: TRAYECTORIAS ===
print("\n" + "="*70)
print("2️⃣ ENTRENAMIENTO MODELO DE TRAYECTORIAS")
print("="*70)

X_train_t, X_test_t, y_train_t, y_test_t = train_test_split(
    X_pixeles, y, test_size=0.2, random_state=42, stratify=y
)

print(f"\nConjunto de entrenamiento: {len(X_train_t)} muestras")
print(f"Conjunto de prueba: {len(X_test_t)} muestras")

# KNN para trayectorias
print("\nEntrenando KNN para trayectorias...")

param_grid_tray = {
    'n_neighbors': [1, 3, 5, 7, 9],
    'weights': ['uniform', 'distance'],
    'metric': ['euclidean', 'manhattan']
}

grid_tray = GridSearchCV(
    KNeighborsClassifier(),
    param_grid_tray,
    cv=5,
    scoring='accuracy',
    n_jobs=-1,
    verbose=1
)

grid_tray.fit(X_train_t, y_train_t)

print(f"\n✅ Mejores parámetros KNN:")
for param, valor in grid_tray.best_params_.items():
    print(f"   - {param}: {valor}")

print(f"\nPrecisión KNN en validación cruzada: {grid_tray.best_score_*100:.2f}%")

knn_trayectoria = grid_tray.best_estimator_

# SVM para trayectorias
print("\nEntrenando SVM para trayectorias...")

svm_trayectoria = SVC(
    kernel='rbf',
    C=1.0,
    gamma='scale',
    probability=True,
    random_state=42
)
svm_trayectoria.fit(X_train_t, y_train_t)

y_pred_svm = svm_trayectoria.predict(X_test_t)
acc_svm = accuracy_score(y_test_t, y_pred_svm)
print(f"Precisión SVM en test: {acc_svm*100:.2f}%")

# Ensemble: Combinar KNN + SVM
print("\nCreando ensemble KNN + SVM...")

ensemble_trayectoria = VotingClassifier(
    estimators=[
        ('knn', knn_trayectoria),
        ('svm', svm_trayectoria)
    ],
    voting='soft',  # Usar probabilidades
    weights=[2, 1]  # Dar más peso a KNN
)

ensemble_trayectoria.fit(X_train_t, y_train_t)

y_pred_t = ensemble_trayectoria.predict(X_test_t)
acc_trayectoria = accuracy_score(y_test_t, y_pred_t)

print(f"\n✅ Precisión del ensemble en test: {acc_trayectoria*100:.2f}%")

print("\nReporte de clasificación (Trayectorias - Ensemble):")
print(classification_report(y_test_t, y_pred_t, zero_division=0))

# === ANÁLISIS DE CONFLICTOS ===
print("\n" + "="*70)
print("3️⃣ ANÁLISIS DE CONFLICTOS ENTRE CLASES")
print("="*70)

# Detectar configuraciones de dedos compartidas
print("\nDetectando configuraciones de dedos compartidas...")

dedos_por_clase = {}
for clase in df['clase'].unique():
    mask = df['clase'] == clase
    dedos_unicos = X_dedos[mask].drop_duplicates()
    dedos_por_clase[clase] = [tuple(row) for _, row in dedos_unicos.iterrows()]

# Encontrar conflictos
conflictos = {}
for clase1, dedos1 in dedos_por_clase.items():
    for clase2, dedos2 in dedos_por_clase.items():
        if clase1 < clase2:  # Evitar duplicados
            interseccion = set(dedos1) & set(dedos2)
            if interseccion:
                conflictos[f"{clase1}-{clase2}"] = list(interseccion)

if conflictos:
    print("\n⚠️  Configuraciones de dedos compartidas detectadas:")
    for par, configs in conflictos.items():
        print(f"\n   {par}:")
        for config in configs:
            config_str = ''.join(map(str, [int(x) for x in config]))
            print(f"      Dedos: [{config_str}]")
    
    # Estas clases DEBEN usar trayectorias para diferenciarse
    clases_ambiguas = set()
    for par in conflictos.keys():
        c1, c2 = par.split('-')
        clases_ambiguas.add(c1)
        clases_ambiguas.add(c2)
    
    print(f"\n   ⚠️  Clases ambiguas (requieren trayectoria para diferenciarse):")
    print(f"      {sorted(clases_ambiguas)}")
else:
    clases_ambiguas = set()
    print("\n✅ No hay conflictos: cada configuración de dedos es única por clase")

# Matriz de confusión
print("\n" + "="*70)
print("4️⃣ MATRICES DE CONFUSIÓN")
print("="*70)

print("\nModelo de Dedos:")
cm_dedos = confusion_matrix(y_test_d, y_pred_d)
print(cm_dedos)

print("\nModelo de Trayectorias:")
cm_tray = confusion_matrix(y_test_t, y_pred_t)
print(cm_tray)

# Identificar errores comunes
print("\n" + "="*70)
print("5️⃣ ANÁLISIS DE ERRORES")
print("="*70)

# Errores del modelo de dedos
errores_dedos = y_test_d != y_pred_d
if errores_dedos.any():
    print("\nErrores comunes del modelo de DEDOS:")
    for i, (real, pred) in enumerate(zip(y_test_d[errores_dedos], y_pred_d[errores_dedos])):
        print(f"   Real: '{real}' → Predicho: '{pred}'")
else:
    print("\n✅ Modelo de dedos sin errores en test")

# Errores del modelo de trayectorias
errores_tray = y_test_t != y_pred_t
if errores_tray.any():
    print("\nErrores comunes del modelo de TRAYECTORIAS:")
    for i, (real, pred) in enumerate(zip(y_test_t[errores_tray], y_pred_t[errores_tray])):
        print(f"   Real: '{real}' → Predicho: '{pred}'")
else:
    print("\n✅ Modelo de trayectorias sin errores en test")

# === GUARDAR MODELOS Y METADATA ===
print("\n" + "="*70)
print("6️⃣ GUARDANDO MODELOS")
print("="*70)

joblib.dump(knn_dedos, "modelo_dedos.pkl")
joblib.dump(ensemble_trayectoria, "modelo_trayectoria.pkl")

metadata = {
    'acc_dedos': acc_dedos,
    'acc_trayectoria': acc_trayectoria,
    'acc_svm': acc_svm,
    'params_dedos': grid_dedos.best_params_,
    'params_trayectoria': grid_tray.best_params_,
    'clases': sorted(y.unique().tolist()),
    'clases_ambiguas': sorted(clases_ambiguas),
    'n_muestras': len(df),
    'n_features_dedos': X_dedos.shape[1],
    'n_features_pixeles': X_pixeles.shape[1],
    'pixel_normalizado': True
}

joblib.dump(metadata, "metadata_modelos.pkl")

print("\n✅ Archivos guardados:")
print("   - modelo_dedos.pkl")
print("   - modelo_trayectoria.pkl")
print("   - metadata_modelos.pkl")

# === RESUMEN FINAL ===
print("\n" + "="*70)
print("📊 RESUMEN FINAL")
print("="*70)

print(f"\n🎯 Rendimiento de Modelos:")
print(f"   - Dedos: {acc_dedos*100:.2f}%")
print(f"   - Trayectorias (KNN solo): {grid_tray.best_score_*100:.2f}%")
print(f"   - Trayectorias (SVM solo): {acc_svm*100:.2f}%")
print(f"   - Trayectorias (Ensemble): {acc_trayectoria*100:.2f}%")

print(f"\n📋 Información del Dataset:")
print(f"   - Total de muestras: {len(df)}")
print(f"   - Total de clases: {len(df['clase'].unique())}")
print(f"   - Clases: {sorted(df['clase'].unique())}")

if clases_ambiguas:
    print(f"\n⚠️  Clases con conflictos de dedos: {len(clases_ambiguas)}")
    print(f"   {sorted(clases_ambiguas)}")
    print("   Estas clases se diferenciarán principalmente por TRAYECTORIA")

print("\n" + "="*70)
print("✅ ENTRENAMIENTO COMPLETADO")
print("="*70)

print("\n💡 Próximos pasos:")
print("   1. Ejecutar inferencia_completa.py para probar el sistema")
print("   2. Si hay errores, revisar el diagnóstico con diagnostico_dataset.py")
print("   3. Considerar agregar más muestras para clases problemáticas")

if acc_dedos < 0.7 or acc_trayectoria < 0.5:
    print("\n⚠️  ADVERTENCIA: Precisión baja detectada")
    print("   Recomendaciones:")
    print("   - Aumentar número de muestras por clase (mínimo 30)")
    print("   - Verificar calidad de las grabaciones")
    print("   - Asegurar movimientos consistentes y distintivos")
    print("   - Ejecutar diagnostico_dataset.py para análisis detallado")