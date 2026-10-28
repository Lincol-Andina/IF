# src/utils/helpers.py
import os
import json
import pandas as pd
from src.utils.config import FIGURES_DIR, METRICS_DIR

def ensure_dirs():
    """Crea directorios necesarios si no existen."""
    os.makedirs(FIGURES_DIR, exist_ok=True)
    os.makedirs(METRICS_DIR, exist_ok=True)

def save_metrics_csv(df: pd.DataFrame, filename: str):
    """Guarda métricas en formato CSV."""
    ensure_dirs()
    path = os.path.join(METRICS_DIR, filename)
    df.to_csv(path, index=False)
    print(f"[OK] Métricas guardadas: {path}")
    return path

def save_metrics_json(data: dict, filename: str):
    """Guarda métricas en formato JSON."""
    ensure_dirs()
    path = os.path.join(METRICS_DIR, filename)
    with open(path, 'w', encoding='utf-8') as f:
        json.dump(data, f, indent=4, ensure_ascii=False)
    print(f"[OK] Métricas guardadas: {path}")
    return path

def save_figure(fig, filename: str):
    """Guarda figura con alta resolución."""
    ensure_dirs()
    path = os.path.join(FIGURES_DIR, filename)
    fig.savefig(path, dpi=300, bbox_inches='tight')
    print(f"[OK] Figura guardada: {path}")
    return path
