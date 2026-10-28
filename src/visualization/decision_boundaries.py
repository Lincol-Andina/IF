# src/visualization/decision_boundaries.py
import numpy as np
import matplotlib.pyplot as plt
from src.utils.config import MESH_STEP

def plot_decision_boundary_2d(clf, X, y, scaler=None, title="Frontera de Decisión 2D", h=MESH_STEP):
    """
    Grafica frontera de decisión para clasificadores 2D.
    Si se proporciona scaler, se entrena sobre X escalado pero se visualiza en malla escalada
    coherente con el espacio de entrada del clasificador.
    """
    X = np.array(X)
    y = np.array(y)
    x_min, x_max = X[:, 0].min() - 0.5, X[:, 0].max() + 0.5
    y_min, y_max = X[:, 1].min() - 0.5, X[:, 1].max() + 0.5
    xx, yy = np.meshgrid(np.arange(x_min, x_max, h),
                         np.arange(y_min, y_max, h))
    grid = np.c_[xx.ravel(), yy.ravel()]
    if scaler is not None:
        grid_s = scaler.transform(grid)
        Z = clf.predict(grid_s)
    else:
        Z = clf.predict(grid)
    Z = Z.reshape(xx.shape)
    plt.figure(figsize=(7, 5))
    plt.contourf(xx, yy, Z, alpha=0.35, cmap=plt.cm.coolwarm)
    scatter = plt.scatter(X[:, 0], X[:, 1], c=y, cmap=plt.cm.coolwarm, edgecolors='k', s=40)
    plt.xlabel('X1')
    plt.ylabel('X2')
    plt.title(title)
    plt.colorbar(scatter)
    plt.tight_layout()
    return plt.gcf()

def plot_regression_surface_2d(reg, X, y, scaler=None, title="Superficie de Regresión SVR (2D)", h=MESH_STEP):
    """
    Grafica superficie de regresión para SVR con 2 features.
    """
    X = np.array(X)
    y = np.array(y)
    x_min, x_max = X[:, 0].min() - 0.5, X[:, 0].max() + 0.5
    y_min, y_max = X[:, 1].min() - 0.5, X[:, 1].max() + 0.5
    xx, yy = np.meshgrid(np.arange(x_min, x_max, h),
                         np.arange(y_min, y_max, h))
    grid = np.c_[xx.ravel(), yy.ravel()]
    if scaler is not None:
        grid_s = scaler.transform(grid)
        Z = reg.predict(grid_s)
    else:
        Z = reg.predict(grid)
    Z = Z.reshape(xx.shape)
    fig = plt.figure(figsize=(8, 6))
    ax = fig.add_subplot(111, projection='3d')
    ax.plot_surface(xx, yy, Z, cmap=plt.cm.viridis, alpha=0.7)
    ax.scatter(X[:, 0], X[:, 1], y, c='r', s=30, edgecolors='k')
    ax.set_xlabel('X1')
    ax.set_ylabel('X2')
    ax.set_zlabel('y')
    ax.set_title(title)
    plt.tight_layout()
    return fig
