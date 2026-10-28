# src/visualization/roc_curves.py
import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve, auc

def plot_roc_curve(y_true, y_proba, title="Curva ROC-AUC"):
    """
    Grafica curva ROC para clasificación binaria.
    """
    if y_proba is None:
        raise ValueError("Se requiere y_proba para graficar ROC-AUC")
    if hasattr(y_proba, 'shape') and len(y_proba.shape) == 2 and y_proba.shape[1] == 2:
        y_score = y_proba[:, 1]
    else:
        y_score = y_proba
    fpr, tpr, _ = roc_curve(y_true, y_score)
    roc_auc = auc(fpr, tpr)
    plt.figure(figsize=(6, 5))
    plt.plot(fpr, tpr, color='darkorange', lw=2, label=f'ROC curve (AUC = {roc_auc:.4f})')
    plt.plot([0, 1], [0, 1], color='navy', lw=1, linestyle='--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('Tasa de Falsos Positivos (FPR)')
    plt.ylabel('Tasa de Verdaderos Positivos (TPR)')
    plt.title(title)
    plt.legend(loc="lower right")
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    return plt.gcf()
