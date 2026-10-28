# docs/03_aplicaciones_casos_uso.md

# Pilar 3: Aplicaciones y Casos de Uso

**Equipo 3 | Familia 2: Modelos Basados en Márgenes y Distancias (Instancias)**

## 3.1. SVC (Máquinas de Vectores de Soporte para Clasificación)

**Caso de estudio real: Detección de cáncer de mama con el dataset Wisconsin Diagnostic Breast Cancer (WDBC)**

- **Problema:** Clasificar tumores mamarios como **benignos** o **malignos** utilizando características numéricas extraídas de imágenes digitalizadas de aspirados con aguja fina (FNA). Estas características incluyen radio, textura, perímetro, área, suavidad, entre otras.
- **Dataset real:** *Breast Cancer Wisconsin (Diagnostic) Dataset* (UCI Machine Learning Repository), ampliamente utilizado en investigación biomédica para validar clasificadores.
- **Por qué es idóneo:** El SVC con kernel RBF es ideal para este problema, ya que permite encontrar un hiperplano de margen máximo en un espacio de alta dimensionalidad, separando eficazmente clases que presentan cierto grado de solapamiento. Además, solo los vectores de soporte participan en la definición de la frontera, lo cual resulta ventajoso para la inferencia.
- **Sensibilidad a outliers:** Moderada-Alta si el parámetro $C$ es muy elevado (penaliza fuertemente los errores, reduciendo el margen). Con valores moderados de $C$, el modelo es más tolerante a puntos atípicos.
- **Interpretabilidad:** Media. Aunque menos interpretable que un árbol de decisión, los vectores de soporte pueden analizarse para entender qué instancias son críticas para la clasificación. El uso de kernel dificulta la visualización directa del hiperplano.
- **Escalabilidad:** Entrenamiento con kernel RBF puede ser costoso para conjuntos de datos con decenas de miles de muestras ($O(n^2)$ a $O(n^3)$). Sin embargo, para el dataset WDBC (569 muestras) es completamente viable.
- **Costo computacional en inferencia:** Bajo-moderado. Solo se requiere evaluar el kernel con los vectores de soporte, por lo que, si el número de SV es reducido, la predicción es rápida.

## 3.2. SVR (Máquinas de Vectores de Soporte para Regresión)

**Caso de estudio real: Predicción de precios de viviendas con el dataset Ames Housing**

- **Problema:** Estimar el precio de venta de viviendas residenciales en Ames, Iowa (EE. UU.), utilizando variables estructurales, de ubicación, materiales, año de construcción, superficie habitable, entre otras.
- **Dataset real:** *Ames Housing Dataset* (Dean De Cock, 2011), ampliamente empleado en competiciones y en investigación para tareas de regresión inmobiliaria.
- **Por qué es idóneo:** SVR con función de pérdida $\epsilon$-insensitiva es adecuado para este problema, ya que permite ignorar pequeñas desviaciones dentro del tubo $\epsilon$ y concentrarse en errores significativos. Con kernel RBF es capaz de modelar relaciones no lineales complejas entre las características y el precio, sin imponer una forma funcional rígida.
- **Sensibilidad a outliers:** Moderada. El tubo $\epsilon$ brinda cierto grado de robustez, pero viviendas con precios extremadamente atípicos (mansiones de lujo) pueden influir si el parámetro $C$ es alto. Un ajuste adecuado de $\epsilon, C$ y $\gamma$ es fundamental.
- **Interpretabilidad:** Media-Baja. Al utilizar kernel no lineal, la relación entre predictores y precio no es directamente interpretable. Se requiere análisis auxiliar (ej. análisis de vectores de soporte) para obtener mayor explicabilidad.
- **Escalabilidad:** Moderada-Baja para grandes volúmenes de datos. El entrenamiento con kernel es computacionalmente costoso conforme crece $n$. Requiere obligatoriamente escalado de características (`StandardScaler`).
- **Costo computacional en inferencia:** Depende directamente del número de vectores de soporte. En datasets inmobiliarios medianos, suele ser aceptable, aunque mayor que modelos puramente lineales.

## 3.3. k-NN (k-Vecinos Más Cercanos)

**Caso de estudio real: Sistema de recomendación de películas basado en filtrado colaborativo con el dataset MovieLens**

- **Problema:** Recomendar películas a un usuario activo basándose en las preferencias de otros usuarios con gustos similares. Dado un usuario $u$, se buscan los $k$ usuarios más parecidos según sus calificaciones históricas y se recomiendan aquellas películas que estos vecinos hayan calificado positivamente y que $u$ aún no haya visto.
- **Dataset real:** *MovieLens* (GroupLens Research, University of Minnesota). Es uno de los datasets más utilizados y representativos para sistemas de recomendación basados en similitud.
- **Por qué es idóneo:** k-NN es naturalmente aplicable al filtrado colaborativo, ya que su lógica es exactamente la de "vecindad": *"Usuarios con historiales de calificación similares tienden a tener preferencias similares"*. Al utilizar métricas de distancia (Euclidiana, Coseno o Manhattan) sobre los vectores de calificación, el modelo identifica usuarios vecinos y genera recomendaciones sin necesidad de un modelo paramétrico complejo.
- **Sensibilidad a outliers:** Alta. Un usuario con calificaciones muy atípicas puede convertirse en vecino erróneo si $k$ es pequeño. Para mitigar esto, se recomienda utilizar $k$ adecuado (valores intermedios), ponderación por distancia ($weights='distance'$) o filtrar usuarios con muy pocas calificaciones.
- **Interpretabilidad:** **Alta**. Es uno de los modelos más explicables: se puede justificar explícitamente la recomendación indicando *"te recomendamos X porque usuarios similares a ti también la calificaron altamente"*. Esto es fundamental para sistemas de recomendación confiables.
- **Escalabilidad:** Entrenamiento: muy eficiente ($O(1)$, solo se almacena la matriz de usuarios-películas). Inferencia: costosa, ya que requiere calcular similitud/distancia entre el usuario objetivo y todos los usuarios del sistema ($O(nd)$). Para grandes versiones de MovieLens (millones de calificaciones) se requiere indexación (ej. KD-Tree, Ball-Tree) o técnicas de reducción dimensional.
- **Costo computacional en inferencia:** Elevado en sistemas con muchos usuarios. En versiones pequeñas de MovieLens es viable para fines académicos; en producción se optimiza con aproximaciones (Nearest Neighbor Search aproximado).

## Justificación Global: Familia de Márgenes y Distancias

- **Márgenes (SVM: SVC/SVR):** Buscan una frontera con **máximo margen** entre clases o un **tubo $\epsilon$-insensitivo** óptimo para regresión, lo que favorece la generalización. Son especialmente efectivos cuando existe una separación clara o cuando se requiere una frontera compacta con buen desempeño en inferencia (gracias a los vectores de soporte).
- **Distancias (k-NN):** Basados en **proximidad**. Funcionan bajo el supuesto de continuidad local: instancias cercanas comparten etiqueta o valor. Son ideales cuando la relación es local y no global, y destacan por su **alta interpretabilidad**.
- **Consideración común:** Ambos modelos son **altamente sensibles a la escala** de las características, por lo que el uso de **`StandardScaler` es obligatorio**. Además, la elección de métrica (SVM con kernel, k-NN con distancia) es determinante para su desempeño. 
- **Comparativo práctico:** k-NN es más interpretable y flexible sin entrenamiento costoso, pero con mayor costo en inferencia. SVC/SVR ofrecen mayor eficiencia en predicción y fronteras más generales, a cambio de menor interpretabilidad con kernels y mayor costo computacional en entrenamiento.
