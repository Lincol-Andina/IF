# src/evaluation/regression_metrics.py
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import numpy as np

class RegressionEvaluator:
    def __init__(self, y_true, y_pred):
        self.y_true = y_true
        self.y_pred = y_pred

    def rmse(self):
        mse = mean_squared_error(self.y_true, self.y_pred)
        return np.sqrt(mse)

    def mae(self):
        return mean_absolute_error(self.y_true, self.y_pred)

    def r2(self):
        return r2_score(self.y_true, self.y_pred)

    def all(self):
        return {
            'RMSE': self.rmse(),
            'MAE': self.mae(),
            'R2': self.r2()
        }
