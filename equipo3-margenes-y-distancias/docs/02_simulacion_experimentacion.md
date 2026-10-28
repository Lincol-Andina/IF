# docs/02_simulacion_experimentacion.md

# Pilar 2: Simulación y Experimentación / Código Fuente

**Equipo 3 | Familia 2: Modelos Basados en Márgenes y Distancias (Instancias)**

## 2.1. Metodología General
Se sigue un flujo reproducible: generación de datos sintéticos -> preprocesamiento -> optimización de hiperparámetros -> entrenamiento -> evaluación -> visualización.

## 2.2. Preprocesamiento
- **Datos sintéticos:** `make_moons` (clasificación, no linealmente separable) y función no lineal para SVR.
- **División:** `train_test_split` con `test_size=0.2`, `random_state=42`.
- **Escalado obligatorio:** `StandardScaler`. Se ajusta (`fit`) únicamente con datos de entrenamiento y se aplica (`transform`) a entrenamiento y prueba (evita *data leakage*).
- **Requisito:** Fundamental para SVC, SVR y, especialmente, k-NN por ser basado en distancias.

## 2.3. Optimización de Hiperparámetros (GridSearchCV)
Se utiliza `GridSearchCV` con validación cruzada `cv=5`.

| Modelo | Espacio de búsqueda | Métrica de scoring |
| :-- | :-- | :-- |
| **SVC** | `C`: [0.1,1,10,100], `gamma`: ['scale','auto',0.01,0.1,1], `kernel`: ['rbf','linear'] | `accuracy` |
| **SVR** | `C`: [0.1,1,10,100], `epsilon`: [0.01,0.1,0.5,1.0], `gamma`: ['scale','auto',0.01,0.1,1], `kernel`: ['rbf','linear'] | `neg_mean_squared_error` |
| **k-NN** | `n_neighbors`: [1,3,5,7,9,11], `weights`: ['uniform','distance'], `metric`: ['euclidean','manhattan','minkowski'], `p`: [1,2] | `accuracy` |

## 2.4. Evaluación de Desempeño

### Clasificación (SVC y k-NN)
- **Matriz de confusión:** Verdaderos/Falsos Positivos y Negativos.
- **Precision:** $TP/(TP+FP)$
- **Recall (Sensibilidad):** $TP/(TP+FN)$
- **F1-Score:** $2*(Precision*Recall)/(Precision+Recall)$
- **ROC-AUC:** Área bajo la curva ROC, con `predict_proba` habilitado en SVC.

### Regresión (SVR)
- **RMSE (Root Mean Squared Error):** $\sqrt{\frac{1}{n}\sum (y_i - \hat{y}_i)^2}$
- **MAE (Mean Absolute Error):** $\frac{1}{n}\sum |y_i - \hat{y}_i|$
- **R² (Coeficiente de Determinación):** $1 - \frac{\sum (y_i-\hat{y}_i)^2}{\sum (y_i-\bar{y})^2}$

## 2.5. Visualización
- **Fronteras de decisión en 2D:** Para SVC y k-NN sobre datos sintéticos 2D, mostrando regiones de clasificación.
- **Superficie de regresión en 2D:** Para SVR, visualización de la función estimada.
- **Curvas ROC:** Para evaluar capacidad discriminativa (SVC y k-NN).
- **Curvas de aprendizaje:** Análisis sesgo-varianza.

## 2.6. Ejecución
```bash
# Instalar dependencias
pip install -r requirements.txt

# Ejecutar experimentos completos
python scripts/run_all_experiments.py

# Ejecutar simulación interactiva
streamlit run app.py
```
