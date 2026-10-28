# src/data/loaders.py
import numpy as np
from sklearn.datasets import make_moons, make_circles, make_blobs
from src.utils.config import RANDOM_STATE

def load_classification_margins(n_samples=400, noise=0.25, random_state=RANDOM_STATE):
    """
    Genera dataset sintético para visualización de fronteras de decisión (SVC, k-NN).
    Recomendado: make_moons (no linealmente separable) para resaltar márgenes/distancias.
    """
    X, y = make_moons(n_samples=n_samples, noise=noise, random_state=random_state)
    return X, y

def load_classification_circles(n_samples=400, noise=0.1, factor=0.5, random_state=RANDOM_STATE):
    """Dataset make_circles para evaluar kernels no lineales."""
    X, y = make_circles(n_samples=n_samples, noise=noise, factor=factor, random_state=random_state)
    return X, y

def load_classification_blobs(n_samples=400, centers=2, cluster_std=1.2, random_state=RANDOM_STATE):
    """Dataset make_blobs (útil para comparar fronteras)."""
    X, y = make_blobs(n_samples=n_samples, centers=centers, cluster_std=cluster_std, random_state=random_state)
    return X, y

def load_regression_svr(n_samples=200, noise=0.2, random_state=RANDOM_STATE):
    """
    Genera datos sintéticos no lineales para SVR: y = sin(2*pi*x1) + 0.5*x2^2 + ruido
    Se generan 2 features para permitir visualización 2D de superficie.
    """
    rng = np.random.RandomState(random_state)
    X = rng.uniform(-1.0, 1.0, size=(n_samples, 2))
    y = np.sin(2 * np.pi * X[:, 0]) + 0.5 * (X[:, 1] ** 2) + rng.normal(0, noise, size=n_samples)
    return X, y

def load_regression_curve_1d(n_samples=150, noise=0.15, random_state=RANDOM_STATE):
    """Versión 1D para visualizar curva SVR vs puntos."""
    rng = np.random.RandomState(random_state)
    X = np.linspace(-2, 2, n_samples).reshape(-1, 1)
    y = np.sin(X).ravel() + rng.normal(0, noise, size=n_samples)
    return X, y
