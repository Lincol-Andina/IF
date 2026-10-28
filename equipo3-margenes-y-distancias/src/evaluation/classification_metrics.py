# src/evaluation/classification_metrics.py
import numpy as np
from sklearn.metrics import (
    confusion_matrix, classification_report,
    accuracy_score, precision_score, recall_score, f1_score,
    roc_curve, roc_auc_score
)

class ClassificationEvaluator:
    def __init__(self, y_true, y_pred, y_proba=None):
        self.y_true = y_true
        self.y_pred = y_pred
        self.y_proba = y_proba

    def confusion_matrix(self):
        return confusion_matrix(self.y_true, self.y_pred)

    def report(self):
        return classification_report(self.y_true, self.y_pred, digits=4)

    def scores(self, average='binary'):
        acc = accuracy_score(self.y_true, self.y_pred)
        prec = precision_score(self.y_true, self.y_pred, average=average, zero_division=0)
        rec = recall_score(self.y_true, self.y_pred, average=average, zero_division=0)
        f1 = f1_score(self.y_true, self.y_pred, average=average, zero_division=0)
        return {'accuracy': acc, 'precision': prec, 'recall': rec, 'f1': f1}

    def roc_auc(self, average='macro'):
        if self.y_proba is None:
            return None
        try:
            if self.y_proba.shape[1] == 2:
                return roc_auc_score(self.y_true, self.y_proba[:, 1])
            else:
                return roc_auc_score(self.y_true, self.y_proba, multi_class='ovr', average=average)
        except Exception:
            return None

    def roc_curve_data(self):
        if self.y_proba is None:
            return None, None, None
        try:
            if hasattr(self.y_proba, 'shape') and len(self.y_proba.shape) == 2 and self.y_proba.shape[1] == 2:
                fpr, tpr, thresholds = roc_curve(self.y_true, self.y_proba[:, 1])
                return fpr, tpr, thresholds
        except Exception:
            pass
        return None, None, None
