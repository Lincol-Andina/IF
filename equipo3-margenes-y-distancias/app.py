# app.py - Simulación Interactiva con Streamlit | Equipo 3 | Familia 2: Márgenes y Distancias
import os
import sys
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

import numpy as np
import matplotlib.pyplot as plt
import streamlit as st
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC, SVR
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import train_test_split
from src.data.loaders import load_classification_margins, load_regression_svr
from src.utils.config import RANDOM_STATE, MESH_STEP

st.set_page_config(page_title="Simulación Interactiva - Equipo 3 | SVC, SVR, k-NN", layout="wide")

st.title("Simulación Interactiva: Modelos Basados en Márgenes y Distancias (Instancias)")
st.caption("Equipo 3 | SVC (Clasificación) | SVR (Regresión) | k-NN (Clasificación)")

# Sidebar
st.sidebar.header("Configuración")
modelo = st.sidebar.selectbox("Seleccionar Modelo", ["SVC", "SVR", "k-NN"])

st.sidebar.subheader("Generación de Datos Sintéticos")
tipo_datos = "clasificacion" if modelo in ["SVC", "k-NN"] else "regresion"
n_samples = st.sidebar.slider("Número de muestras", 100, 600, 400, step=50)
noise = st.sidebar.slider("Ruido", 0.0, 0.5, 0.25, step=0.01)

@st.cache_data
def get_data_clf(n, ns):
    X, y = load_classification_margins(n_samples=n, noise=ns, random_state=RANDOM_STATE)
    return X, y

@st.cache_data
def get_data_reg(n, ns):
    X, y = load_regression_svr(n_samples=n, noise=ns, random_state=RANDOM_STATE)
    return X, y

if modelo in ["SVC", "k-NN"]:
    X, y = get_data_clf(n_samples, noise)
    st.sidebar.success("Datos: make_moons (Clasificación)")
else:
    X, y = get_data_reg(n_samples, noise)
    st.sidebar.success("Datos: Sintético no lineal (Regresión)")

# División y escalado
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=RANDOM_STATE)
scaler = StandardScaler()
X_train_s = scaler.fit_transform(X_train)
X_test_s = scaler.transform(X_test)

# Hiperparámetros dinámicos
st.sidebar.subheader("Hiperparámetros")
if modelo == "SVC":
    C = st.sidebar.slider("C (Margen)", 0.01, 100.0, 1.0, step=0.01)
    gamma = st.sidebar.slider("gamma (Kernel RBF)", 0.001, 10.0, 0.1, step=0.001)
    kernel = st.sidebar.selectbox("Kernel", ["rbf", "linear"])
    clf = SVC(C=C, gamma=gamma, kernel=kernel, probability=True, random_state=RANDOM_STATE)
    clf.fit(X_train_s, y_train)
    y_pred = clf.predict(X_test_s)
    acc = np.mean(y_pred == y_test)
    st.sidebar.metric("Exactitud (Test)", f"{acc:.4f}")
elif modelo == "SVR":
    C = st.sidebar.slider("C", 0.01, 100.0, 10.0, step=0.01)
    epsilon = st.sidebar.slider("epsilon (Tubo ε-insensitivo)", 0.01, 1.0, 0.1, step=0.01)
    gamma = st.sidebar.slider("gamma (Kernel RBF)", 0.001, 10.0, 0.1, step=0.001)
    kernel = st.sidebar.selectbox("Kernel", ["rbf", "linear"])
    reg = SVR(C=C, epsilon=epsilon, gamma=gamma, kernel=kernel)
    reg.fit(X_train_s, y_train)
    y_pred_r = reg.predict(X_test_s)
    from sklearn.metrics import mean_squared_error, r2_score
    rmse = np.sqrt(mean_squared_error(y_test, y_pred_r))
    r2 = r2_score(y_test, y_pred_r)
    st.sidebar.metric("RMSE (Test)", f"{rmse:.4f}")
    st.sidebar.metric("R² (Test)", f"{r2:.4f}")
elif modelo == "k-NN":
    k = st.sidebar.slider("Número de vecinos k", 1, 25, 5, step=1)
    metric = st.sidebar.selectbox("Métrica de distancia", ["euclidean", "manhattan", "minkowski"])
    p = 2
    if metric == "minkowski":
        p = st.sidebar.slider("p (Minkowski)", 1, 3, 2, step=1)
    weights = st.sidebar.selectbox("Pesos", ["uniform", "distance"])
    knn = KNeighborsClassifier(n_neighbors=k, weights=weights, metric=metric, p=p)
    knn.fit(X_train_s, y_train)
    y_pred = knn.predict(X_test_s)
    acc = np.mean(y_pred == y_test)
    st.sidebar.metric("Exactitud (Test)", f"{acc:.4f}")

# Visualización
col1, col2 = st.columns([2, 1])
with col1:
    st.subheader(f"Visualización en Tiempo Real - {modelo}")
    fig, ax = plt.subplots(figsize=(7, 5))
    if modelo in ["SVC", "k-NN"]:
        X_all_s = scaler.transform(X)
        x_min, x_max = X[:, 0].min() - 0.5, X[:, 0].max() + 0.5
        y_min, y_max = X[:, 1].min() - 0.5, X[:, 1].max() + 0.5
        xx, yy = np.meshgrid(np.arange(x_min, x_max, MESH_STEP),
                             np.arange(y_min, y_max, MESH_STEP))
        grid = np.c_[xx.ravel(), yy.ravel()]
        grid_s = scaler.transform(grid)
        if modelo == "SVC":
            Z = clf.predict(grid_s)
        else:
            Z = knn.predict(grid_s)
        Z = Z.reshape(xx.shape)
        ax.contourf(xx, yy, Z, alpha=0.35, cmap=plt.cm.coolwarm)
        scatter = ax.scatter(X_train[:, 0], X_train[:, 1], c=y_train, cmap=plt.cm.coolwarm, edgecolors='k', s=40, label='Entrenamiento')
        ax.scatter(X_test[:, 0], X_test[:, 1], c=y_test, cmap=plt.cm.coolwarm, marker='s', edgecolors='k', s=40, label='Prueba')
        ax.set_title(f"Frontera de Decisión 2D - {modelo}")
        ax.legend()
    else:  # SVR
        x_min, x_max = X[:, 0].min() - 0.5, X[:, 0].max() + 0.5
        y_min, y_max = X[:, 1].min() - 0.5, X[:, 1].max() + 0.5
        xx, yy = np.meshgrid(np.arange(x_min, x_max, MESH_STEP),
                             np.arange(y_min, y_max, MESH_STEP))
        grid = np.c_[xx.ravel(), yy.ravel()]
        grid_s = scaler.transform(grid)
        Z = reg.predict(grid_s)
        Z = Z.reshape(xx.shape)
        c = ax.contourf(xx, yy, Z, alpha=0.6, cmap=plt.cm.viridis)
        fig.colorbar(c, ax=ax, label='y predicho')
        ax.scatter(X_train[:, 0], X_train[:, 1], c=y_train, cmap=plt.cm.coolwarm, edgecolors='k', s=40, label='Entrenamiento')
        ax.scatter(X_test[:, 0], X_test[:, 1], c=y_test, cmap=plt.cm.coolwarm, marker='s', edgecolors='k', s=40, label='Prueba')
        ax.set_title("SVR - Superficie de Regresión 2D")
        ax.legend()
    ax.set_xlabel('X1')
    ax.set_ylabel('X2')
    plt.tight_layout()
    st.pyplot(fig)

with col2:
    st.subheader("Información del Modelo")
    st.markdown(f"**Modelo seleccionado:** {modelo}")
    st.markdown(f"**Muestras totales:** {len(X)}")
    st.markdown(f"**Train/Test:** {len(X_train)}/{len(X_test)}")
    st.markdown("**Preprocesamiento:** StandardScaler (obligatorio)")
    if modelo == "SVC":
        st.json({"C": C, "gamma": gamma, "kernel": kernel})
    elif modelo == "SVR":
        st.json({"C": C, "epsilon": epsilon, "gamma": gamma, "kernel": kernel})
    else:
        st.json({"k": k, "weights": weights, "metric": metric, "p": p})
    st.info("Interactivo en tiempo real. Ajusta los sliders para observar cambios en la frontera/superficie.")

st.markdown("---")
st.caption("Desarrollado por Equipo 3 | Actividad de Investigación - Inteligencia Artificial | Familia 2: Márgenes y Distancias")
