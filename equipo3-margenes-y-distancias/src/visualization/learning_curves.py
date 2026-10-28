# src/visualization/learning_curves.py
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import learning_curve

def plot_learning_curve(estimator, X, y, title="Curvas de Aprendizaje", cv=5, scoring='accuracy', n_jobs=None):
    """
    Grafica curvas de aprendizaje (entrenamiento vs validación).
    """
    train_sizes, train_scores, test_scores = learning_curve(
        estimator, X, y, cv=cv, scoring=scoring, n_jobs=n_jobs,
        train_sizes=np.linspace(0.1, 1.0, 5)
    )
    train_mean = np.mean(train_scores, axis=1)
    train_std = np.std(train_scores, axis=1)
    test_mean = np.mean(test_scores, axis=1)
    test_std = np.std(test_scores, axis=1)
    plt.figure(figsize=(7, 5))
    plt.plot(train_sizes, train_mean, 'o-', color='r', label='Entrenamiento')
    plt.plot(train_sizes, test_mean, 'o-', color='g', label='Validación Cruzada')
    plt.fill_between(train_sizes, train_mean - train_std, train_mean + train_std, alpha=0.1, color='r')
    plt.fill_between(train_sizes, test_mean - test_std, test_mean + test_std, alpha=0.1, color='g')
    plt.title(title)
    plt.xlabel('Tamaño del conjunto de entrenamiento')
    plt.ylabel(scoring)
    plt.legend(loc='best')
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    return plt.gcf()
