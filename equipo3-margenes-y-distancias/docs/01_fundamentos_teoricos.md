# docs/01_fundamentos_teoricos.md

# Pilar 1: Fundamentos Matemáticos y Teóricos

**Equipo 3 | Familia 2: Modelos Basados en Márgenes y Distancias (Instancias)**

## 1.1. Máquinas de Vectores de Soporte para Clasificación (SVC)

### Formulación del modelo
Para un problema de clasificación binaria con conjunto de entrenamiento $\{(x_i, y_i)\}_{i=1}^{n}$, $x_i \in \mathbb{R}^d$, $y_i \in \{+1, -1\}$, SVM busca el hiperplano separador óptimo:

$$
h(x) = w^T x + b = 0
$$

La función de decisión es:

$$
\hat{y} = \operatorname{sign}(w^T x + b)
$$

### Criterio de optimización (margen máximo)
El objetivo es maximizar el margen geométrico $\frac{2}{\|w\|}$, equivalente a minimizar $\frac{1}{2}\|w\|^2$. Para datos no separables linealmente se introducen variables de holgura $\xi_i \geq 0$:

$$
\min_{w,b,\xi} \frac{1}{2}\|w\|^2 + C \sum_{i=1}^{n} \xi_i
$$

Sujeto a:

$$
y_i (w^T x_i + b) \geq 1 - \xi_i, \quad \xi_i \geq 0, \quad i=1,\ldots,n
$$

Donde $C > 0$ es el parámetro de regularización (trade-off entre margen y clasificación errónea).

### Algoritmo de aprendizaje (Programación Cuadrática / Multiplicadores de Lagrange - KKT)
Formulación dual (vía multiplicadores $\alpha_i \geq 0$):

$$
\max_{\alpha} \sum_{i=1}^{n} \alpha_i - \frac{1}{2}\sum_{i=1}^{n}\sum_{j=1}^{n} \alpha_i \alpha_j y_i y_j (x_i^T x_j)
$$

Sujeto a $0 \leq \alpha_i \leq C$, $\sum \alpha_i y_i = 0$. Mediante el **kernel trick**, $K(x_i,x_j)=\langle \phi(x_i),\phi(x_j)\rangle$ permite fronteras no lineales:

$$
f(x)=\sum_{i \in SV} \alpha_i y_i K(x_i,x) + b
$$

Las condiciones **KKT** garantizan la optimalidad. La solución se obtiene mediante **Programación Cuadrática (QP)**.

## 1.2. Máquinas de Vectores de Soporte para Regresión (SVR)

### Formulación del modelo
SVR busca una función $f(x)=w^T x + b$ que se desvíe a lo sumo $\epsilon$ de los valores reales, manteniendo la función lo más plana posible.

### Criterio de optimización (pérdida ε-insensitiva)
Se minimiza:

$$
\min_{w,b,\xi_i,\xi_i^*} \frac{1}{2}\|w\|^2 + C \sum_{i=1}^{n} (\xi_i + \xi_i^*)
$$

Sujeto a:

$$
y_i - (w^T x_i + b) \leq \epsilon + \xi_i, \quad
(w^T x_i + b) - y_i \leq \epsilon + \xi_i^*, \quad
\xi_i, \xi_i^* \geq 0
$$

La **función de pérdida ε-insensitiva** es $L_\epsilon(y,f(x))=\max(0, |y-f(x)|-\epsilon)$.

### Algoritmo de aprendizaje
Dual con kernel:

$$
\max_{\alpha,\alpha^*} -\epsilon \sum_{i=1}^{n}(\alpha_i+\alpha_i^*) + \sum_{i=1}^{n} y_i(\alpha_i - \alpha_i^*) - \frac{1}{2}\sum_{i,j}(\alpha_i-\alpha_i^*)(\alpha_j-\alpha_j^*)K(x_i,x_j)
$$

Restricciones $0 \leq \alpha_i, \alpha_i^* \leq C$, $\sum(\alpha_i-\alpha_i^*)=0$. Predicción:

$$
f(x)=\sum_{i=1}^{n}(\alpha_i - \alpha_i^*)K(x_i,x)+b
$$

Solo los puntos fuera del tubo $|y_i - f(x_i)| \geq \epsilon$ (vectores de soporte) contribuyen al modelo.

## 1.3. k-Vecinos Más Cercanos (k-NN)

### Formulación del modelo (basado en instancias)
k-NN es un **método perezoso (lazy)**: no construye un modelo paramétrico explícito en entrenamiento, almacena el conjunto $\{(x_i,y_i)\}$. Para una consulta $x_q$:

1. Calcular distancias $d(x_q, x_i)$ para $i=1,\ldots,n$.
2. Seleccionar el conjunto $N_k(x_q)$ con los $k$ vecinos más cercanos.
3. Predecir por votación mayoritaria (clasificación) o promedio (regresión).

### Criterio de decisión
**Clasificación:**

$$
\hat{y}(x_q) = \arg\max_{c} \sum_{i \in N_k(x_q)} w_i \, \delta(y_i, c), \quad \delta: \text{indicador}
$$

Con pesos $w_i=1/k$ (uniforme) o $w_i = 1/d(x_q,x_i)$ (ponderado por distancia).

**Regresión:**

$$
\hat{y}(x_q)=\frac{1}{k}\sum_{i \in N_k(x_q)} y_i \quad \text{o} \quad \hat{y}(x_q)=\frac{\sum w_i y_i}{\sum w_i}
$$

### Métricas de distancia comunes
- Euclidiana ($L_2$): $d(x,x')=\sqrt{\sum_j (x_j - x'_j)^2}$
- Manhattan ($L_1$): $d(x,x')=\sum_j |x_j - x'_j|$
- Minkowski ($L_p$): $d(x,x')=(\sum_j |x_j - x'_j|^p)^{1/p}$

### Algoritmo de aprendizaje
- **Fase de entrenamiento:** Almacenamiento del dataset (costo $O(1)$).
- **Fase de inferencia:** Búsqueda de vecinos más cercanos (brute force $O(nd)$, o estructuras como KD-Tree/Ball-Tree $O(\log n)$ en bajas dimensiones).
- **Requisito crítico:** **Escalado de características** (StandardScaler/MinMaxScaler), ya que es sensible a escalas.
