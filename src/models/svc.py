# src/models/svc.py
from sklearn.svm import SVC
from sklearn.model_selection import GridSearchCV
from src.utils.config import RANDOM_STATE

class SVCModel:
    def __init__(self, random_state=RANDOM_STATE):
        self.random_state = random_state
        self.best_estimator_ = None
        self.grid_search = None
        self.best_params_ = None
        self.best_score_ = None

    def get_param_grid(self):
        return {
            'C': [0.1, 1, 10, 100],
            'gamma': ['scale', 'auto', 0.01, 0.1, 1],
            'kernel': ['rbf', 'linear']
        }

    def train(self, X_train_s, y_train, cv=5, scoring='accuracy'):
        svc = SVC(probability=True, random_state=self.random_state)
        self.grid_search = GridSearchCV(svc, self.get_param_grid(), cv=cv, scoring=scoring, n_jobs=-1)
        self.grid_search.fit(X_train_s, y_train)
        self.best_estimator_ = self.grid_search.best_estimator_
        self.best_params_ = self.grid_search.best_params_
        self.best_score_ = self.grid_search.best_score_
        return self.best_estimator_

    def predict(self, X_s):
        return self.best_estimator_.predict(X_s)

    def predict_proba(self, X_s):
        return self.best_estimator_.predict_proba(X_s)
