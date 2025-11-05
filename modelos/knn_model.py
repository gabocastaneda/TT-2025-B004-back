from collections import Counter
import numpy as np

class KNN:
    def __init__(self, k):
        self.k = k

    def fit(self, X, y):
        self.X_train = X
        self.y_train = y

    def predict(self, X):
        return np.array([self._predecir(x) for x in X])
    
    def predict_proba(self, X):
        predictions = []
        for x in X:
            distancias = np.linalg.norm(self.X_train - x, axis=1)
            vecinos_idx = np.argsort(distancias)[:self.k]
            etiquetas_vecinas = self.y_train[vecinos_idx]
            counter = Counter(etiquetas_vecinas)
            total_vecinos = len(etiquetas_vecinas)
            probas = [counter.get(clase, 0) / total_vecinos for clase in np.unique(self.y_train)]
            predictions.append(probas)
        
        return np.array(predictions)

    def _predecir(self, x):
        distancias = np.linalg.norm(self.X_train - x, axis=1)
        vecinos_idx = np.argsort(distancias)[:self.k]
        etiquetas_vecinas = self.y_train[vecinos_idx]
        etiqueta_mas_comun = Counter(etiquetas_vecinas).most_common(1)[0][0]
        return etiqueta_mas_comun
