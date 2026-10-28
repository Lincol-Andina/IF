# src/data/preprocessing.py
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from src.utils.config import TEST_SIZE, RANDOM_STATE

class DataPreprocessor:
    """
    Preprocesamiento robusto para modelos basados en márgenes y distancias.
    - División train/test
    - Escalado obligatorio con StandardScaler (fit solo en train)
    """
    def __init__(self, test_size=TEST_SIZE, random_state=RANDOM_STATE, stratify=None):
        self.test_size = test_size
        self.random_state = random_state
        self.stratify = stratify
        self.scaler = StandardScaler()
        self.X_train = None
        self.X_test = None
        self.y_train = None
        self.y_test = None
        self.X_train_s = None
        self.X_test_s = None

    def split_and_scale(self, X, y):
        if self.stratify is not None:
            self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(
                X, y, test_size=self.test_size, random_state=self.random_state, stratify=self.stratify
            )
        else:
            self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(
                X, y, test_size=self.test_size, random_state=self.random_state
            )
        # Escalado obligatorio
        self.X_train_s = self.scaler.fit_transform(self.X_train)
        self.X_test_s = self.scaler.transform(self.X_test)
        return self.X_train_s, self.X_test_s, self.y_train, self.y_test, self.scaler

    def get_data(self):
        return {
            'X_train': self.X_train, 'X_test': self.X_test,
            'y_train': self.y_train, 'y_test': self.y_test,
            'X_train_s': self.X_train_s, 'X_test_s': self.X_test_s,
            'scaler': self.scaler
        }
