# Equipo 3 | Familia 2: Modelos Basados en Márgenes y Distancias (Instancias)

Proyecto correspondiente a la actividad de Investigación de Inteligencia Artificial. Se desarrollan los tres modelos asignados: **SVC, SVR y k-NN**.

## Integrantes
- Equipo 3

## Objetivo
Desarrollar, simular y documentar la familia de **Modelos Basados en Márgenes y Distancias (Instancias)** cumpliendo con los tres pilares solicitados: fundamentos matemáticos y teóricos, simulación y experimentación con código fuente, y aplicaciones y casos de uso. Además, se incluye una **Simulación Interactiva** con Streamlit.

## Estructura del Proyecto
```text
equipo3-margenes-y-distancias/
├── app.py                  # Simulación Interactiva con Streamlit
├── docs/                   # Documentación por pilares
├── scripts/                # Script unificado de experimentación
├── src/                    # Código modular (data, models, evaluation, visualization, utils)
├── notebooks/              # Notebooks exploratorios
├── results/                # Figuras y métricas generadas
└── requirements.txt
```

## Instalación
```bash
# Crear entorno virtual (recomendado)
python -m venv venv
# Windows
venv\Scripts\activate
# Linux/Mac
# source venv/bin/activate

# Instalar dependencias
pip install -r requirements.txt
```

## 1) Simulación y Experimentación (Código Fuente)
Para ejecutar todos los experimentos (SVC, SVR, k-NN), optimización con GridSearchCV, evaluación y generación de figuras:

```bash
python scripts/run_all_experiments.py
```

Los resultados se guardarán en:
- `results/metrics/` (CSV/JSON con métricas y mejores hiperparámetros)
- `results/figures/` (fronteras de decisión, curvas ROC, etc.)

## 2) Simulación Interactiva (Streamlit) - Obligatorio
La interfaz interactiva permite seleccionar SVC, SVR o k-NN, ajustar hiperparámetros en tiempo real y visualizar la frontera de decisión / comportamiento del modelo en 2D.

```bash
streamlit run app.py
```

Al ejecutar, se abrirá en `http://localhost:8501`.

## 3) Documentación
Consultar los documentos en `docs/`:
- `01_fundamentos_teoricos.md` - Pilar 1 (Formulación, optimización, algoritmo)
- `02_simulacion_experimentacion.md` - Pilar 2 (Metodología, hiperparámetros, métricas)
- `03_aplicaciones_casos_uso.md` - Pilar 3 (Casos de uso reales)
- `04_conclusiones_discusion.md` - Conclusiones y comparación SVM vs k-NN

## Notas Metodológicas
- **Escalado obligatorio:** Se utiliza `StandardScaler` para todos los modelos (fundamental para modelos basados en distancias y márgenes).
- **Reproducibilidad:** Se emplean semillas (`random_state`) en generación de datos, división train/test y modelos.
- **Búsqueda de hiperparámetros:** `GridSearchCV` con validación cruzada.
- **Visualización 2D:** Fronteras de decisión generadas para análisis intuitivo.
