# src/utils/config.py
import os
import random
import numpy as np

# Semillas para reproducibilidad
RANDOM_STATE = 42

# Rutas del proyecto
BASE_DIR = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
DATA_RAW_DIR = os.path.join(BASE_DIR, "data", "raw")
DATA_PROCESSED_DIR = os.path.join(BASE_DIR, "data", "processed")
RESULTS_DIR = os.path.join(BASE_DIR, "results")
FIGURES_DIR = os.path.join(RESULTS_DIR, "figures")
METRICS_DIR = os.path.join(RESULTS_DIR, "metrics")
DOCS_DIR = os.path.join(BASE_DIR, "docs")
SCRIPTS_DIR = os.path.join(BASE_DIR, "scripts")
SRC_DIR = os.path.join(BASE_DIR, "src")

# División train/test
TEST_SIZE = 0.2

# Configuración para visualización de fronteras de decisión
MESH_STEP = 0.02

def set_seed(seed: int = RANDOM_STATE):
    """Fija semillas para reproducibilidad."""
    random.seed(seed)
    np.random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
