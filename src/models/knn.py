# src/models/knn.py
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import GridSearchCV
from src.utils.config import RANDOM_STATE

class KNNModel:
    def __init__(self, random_state=RANDOM_STATE):
        self.random_state = random_state
        self.best_estimator_ = None
        self.grid_search = None
        self.best_params_ = None
        self.best_score_ = None

    def get_param_grid(self):
        return {
            'n_neighbors': [1, 3, 5, 7, 9, 11],
            'weights': ['uniform', 'distance'],
            'metric': ['euclidean', 'manhattan', 'minkowski'],
            'p': [1, 2]
        }

    def train(self, X_train_s, y_train, cv=5, scoring='accuracy'):
        knn = KNeighborsClassifier()
        self.grid_search = GridSearchCV(knn, self.get_param_grid(), cv=cv, scoring=scoring, n_jobs=-1)
        self.grid_search.fit(X_train_s, y_train)
        self.best_estimator_ = self.grid_search.best_estimator_
        self.best_params_ = self.grid_search.best_params_
        self.best_score_ = self.grid_search.best_score_
        return self.best_estimator_

    def predict(self, X_s):
        return self.best_estimator_.predict(X_s)

    def predict_proba(self, X_s):
        return self.best_estimator_.predict_proba(X_s)
