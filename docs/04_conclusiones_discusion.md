# docs/04_conclusiones_discusion.md

# Conclusiones y Discusión

**Equipo 3 | Familia 2: Modelos Basados en Márgenes y Distancias (Instancias)**

## Comparación Práctica: SVM (SVC/SVR) vs k-NN

| Aspecto | **SVC/SVR (Basado en Márgenes)** | **k-NN (Basado en Distancias/Instancias)** |
| :-- | :-- | :-- |
| **Tipo** | Paramétrico/geométrico (construye frontera/tubo) | No paramétrico, basado en instancias (lazy learning) |
| **Entrenamiento** | Más costoso (optimización QP / GridSearch). Requiere buscar solución óptima. | Muy barato (solo almacena datos). |
| **Inferencia** | Eficiente si hay pocos vectores de soporte. | Costosa en datasets grandes (calcula distancias a todo el set). |
| **Fronteras de decisión** | Suaves/controladas por margen. Con RBF generan fronteras flexibles pero con regularización vía C/gamma. | Irregulares y adaptadas localmente. Muy flexibles, propensas a sobreajuste con k pequeño. |
| **Sensibilidad a escala** | **Crítica** - Requiere escalado. | **Crítica** - Requiere escalado (distancias dominadas por features con mayor rango). |
| **Sensibilidad a outliers** | Controlable con C, epsilon y elección de kernel. | Mayor con k pequeño; mejora con k mayor o `weights='distance'`. |
| **Interpretabilidad** | Media. Se entienden margen/vectores de soporte, pero kernel RBF es menos intuitivo. | **Alta**. Muy intuitivo: decisión basada en vecinos más cercanos. |
| **Generalización** | Buena con buen ajuste (margen máximo tiende a favorecer generalización). | Buena con k óptimo y escalado correcto; sensible a ruido/localidad. |
| **Manejo de no linealidad** | Excelente con kernel trick (RBF/polinomial). | Excelente localmente sin kernels explícitos. |
| **Datos desequilibrados** | Se puede mitigar con `class_weight`. | Requiere atención (votación puede favorecer clase mayoritaria). |

## Observaciones de los Experimentos
- **Escalado obligatorio:** Confirmado como necesario para obtener resultados coherentes con márgenes y distancias.
- **Optimización con GridSearchCV:** Permite encontrar equilibrio entre sesgo/varianza (especialmente en k-NN al elegir k; en SVM al balancear C/gamma).
- **Visualización 2D:** Las fronteras de decisión permiten comparar claramente el comportamiento: SVM tiende a generar fronteras más "globales" con margen, k-NN más "fragmentadas"/locales.
- **SVR ε-insensitivo:** El tubo ε es útil para ignorar variaciones pequeñas, enfocando el modelo en predicciones con errores relevantes.

## Conclusión General
Los modelos de la **Familia 2** son potentes cuando se aplican correctamente. **SVM** destaca por su fundamento teórico (margen óptimo), buen rendimiento con datos bien escalados y eficiencia en inferencia con pocos vectores de soporte. **k-NN** destaca por su simplicidad, interpretabilidad y efectividad en problemas con estructura local. La elección entre ellos debe considerar: tamaño del dataset, necesidad de interpretabilidad, costo computacional en inferencia, presencia de ruido/outliers y linealidad/no linealidad del problema.

**Recomendación práctica:** Para problemas donde se necesita explicar la decisión caso por caso, **k-NN** es preferible. Para problemas con mayor dimensionalidad o necesidad de frontera compacta/generalizable, **SVC/SVR** suelen ser más adecuados, siempre con preprocesamiento (escalado) riguroso.
