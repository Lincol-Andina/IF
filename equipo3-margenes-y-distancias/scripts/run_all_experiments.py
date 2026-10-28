# scripts/run_all_experiments.py
import os
import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import matplotlib.pyplot as plt
from src.utils.config import set_seed, FIGURES_DIR, METRICS_DIR
from src.utils.helpers import ensure_dirs, save_metrics_csv, save_metrics_json, save_figure
from src.data.loaders import load_classification_margins, load_regression_svr
from src.data.preprocessing import DataPreprocessor
from src.models.svc import SVCModel
from src.models.svr import SVRModel
from src.models.knn import KNNModel
from src.evaluation.classification_metrics import ClassificationEvaluator
from src.evaluation.regression_metrics import RegressionEvaluator
from src.visualization.decision_boundaries import plot_decision_boundary_2d, plot_regression_surface_2d
from src.visualization.roc_curves import plot_roc_curve
from src.visualization.learning_curves import plot_learning_curve
import pandas as pd

def main():
    print("=" * 70)
    print("EQUIPO 3 - FAMILIA 2: MODELOS BASADOS EN MÁRGENES Y DISTANCIAS (INSTANCIAS)")
    print("SVC | SVR | k-NN - Simulación y Experimentación")
    print("=" * 70)
    set_seed()
    ensure_dirs()

    # ==================== CLASIFICACIÓN (SVC y k-NN) ====================
    print("\n[1/3] CARGANDO DATOS SINTÉTICOS PARA CLASIFICACIÓN (make_moons)")
    X_clf, y_clf = load_classification_margins(n_samples=400, noise=0.25)
    pre_clf = DataPreprocessor(stratify=y_clf)
    Xtr_s, Xte_s, ytr, yte, scaler_clf = pre_clf.split_and_scale(X_clf, y_clf)
    print(f"Train: {Xtr_s.shape}, Test: {Xte_s.shape}")

    # SVC
    print("\n[2/3] ENTRENANDO Y OPTIMIZANDO SVC (GridSearchCV)")
    svc_m = SVCModel()
    svc_best = svc_m.train(Xtr_s, ytr, cv=5, scoring='accuracy')
    print(f"Mejores parámetros SVC: {svc_m.best_params_}")
    print(f"Mejor score CV (accuracy): {svc_m.best_score_:.4f}")
    ypred_svc = svc_best.predict(Xte_s)
    yprob_svc = svc_best.predict_proba(Xte_s)
    ev_svc = ClassificationEvaluator(yte, ypred_svc, yprob_svc)
    scores_svc = ev_svc.scores(average='binary')
    auc_svc = ev_svc.roc_auc()
    print("\n--- SVC - Métricas en Test ---")
    print(ev_svc.report())
    print(f"ROC-AUC: {auc_svc:.4f}" if auc_svc is not None else "ROC-AUC: N/A")
    # Guardar
    df_svc = pd.DataFrame([{'Modelo': 'SVC', **scores_svc, 'ROC-AUC': auc_svc, **svc_m.best_params_}])
    save_metrics_csv(df_svc, 'metrics_svc.csv')
    save_metrics_json({'mejores_parametros': svc_m.best_params_, 'scores_test': {**scores_svc, 'ROC-AUC': auc_svc}}, 'metrics_svc.json')
    # Figuras
    fig_db_svc = plot_decision_boundary_2d(svc_best, scaler_clf.transform(X_clf), y_clf, scaler=None, title='SVC - Frontera de Decisión 2D (Kernel RBF/Linear optimizado)')
    save_figure(fig_db_svc, 'frontera_decision_svc.png')
    plt.close(fig_db_svc)
    fig_roc_svc = plot_roc_curve(yte, yprob_svc, title='SVC - Curva ROC-AUC')
    save_figure(fig_roc_svc, 'roc_svc.png')
    plt.close(fig_roc_svc)
    fig_lc_svc = plot_learning_curve(svc_best, Xtr_s, ytr, title='SVC - Curvas de Aprendizaje', cv=5, scoring='accuracy')
    save_figure(fig_lc_svc, 'learning_curve_svc.png')
    plt.close(fig_lc_svc)

    # k-NN
    print("\n[3/3] ENTRENANDO Y OPTIMIZANDO k-NN (GridSearchCV)")
    knn_m = KNNModel()
    knn_best = knn_m.train(Xtr_s, ytr, cv=5, scoring='accuracy')
    print(f"Mejores parámetros k-NN: {knn_m.best_params_}")
    print(f"Mejor score CV (accuracy): {knn_m.best_score_:.4f}")
    ypred_knn = knn_best.predict(Xte_s)
    yprob_knn = knn_best.predict_proba(Xte_s)
    ev_knn = ClassificationEvaluator(yte, ypred_knn, yprob_knn)
    scores_knn = ev_knn.scores(average='binary')
    auc_knn = ev_knn.roc_auc()
    print("\n--- k-NN - Métricas en Test ---")
    print(ev_knn.report())
    print(f"ROC-AUC: {auc_knn:.4f}" if auc_knn is not None else "ROC-AUC: N/A")
    df_knn = pd.DataFrame([{'Modelo': 'k-NN', **scores_knn, 'ROC-AUC': auc_knn, **knn_m.best_params_}])
    save_metrics_csv(df_knn, 'metrics_knn.csv')
    save_metrics_json({'mejores_parametros': knn_m.best_params_, 'scores_test': {**scores_knn, 'ROC-AUC': auc_knn}}, 'metrics_knn.json')
    fig_db_knn = plot_decision_boundary_2d(knn_best, scaler_clf.transform(X_clf), y_clf, scaler=None, title='k-NN - Frontera de Decisión 2D')
    save_figure(fig_db_knn, 'frontera_decision_knn.png')
    plt.close(fig_db_knn)
    fig_roc_knn = plot_roc_curve(yte, yprob_knn, title='k-NN - Curva ROC-AUC')
    save_figure(fig_roc_knn, 'roc_knn.png')
    plt.close(fig_roc_knn)
    fig_lc_knn = plot_learning_curve(knn_best, Xtr_s, ytr, title='k-NN - Curvas de Aprendizaje', cv=5, scoring='accuracy')
    save_figure(fig_lc_knn, 'learning_curve_knn.png')
    plt.close(fig_lc_knn)

    # ==================== REGRESIÓN (SVR) ====================
    print("\n" + "=" * 70)
    print("REGRESIÓN - SVR")
    print("=" * 70)
    X_reg, y_reg = load_regression_svr(n_samples=200, noise=0.2)
    pre_reg = DataPreprocessor()
    Xtr_r_s, Xte_r_s, ytr_r, yte_r, scaler_reg = pre_reg.split_and_scale(X_reg, y_reg)
    print(f"Train: {Xtr_r_s.shape}, Test: {Xte_r_s.shape}")
    svr_m = SVRModel()
    svr_best = svr_m.train(Xtr_r_s, ytr_r, cv=5, scoring='neg_mean_squared_error')
    print(f"Mejores parámetros SVR: {svr_m.best_params_}")
    ypred_svr = svr_best.predict(Xte_r_s)
    ev_svr = RegressionEvaluator(yte_r, ypred_svr)
    svr_scores = ev_svr.all()
    print("\n--- SVR - Métricas en Test ---")
    for k,v in svr_scores.items():
        print(f"{k}: {v:.4f}")
    df_svr = pd.DataFrame([{'Modelo': 'SVR', **svr_scores, **svr_m.best_params_}])
    save_metrics_csv(df_svr, 'metrics_svr.csv')
    save_metrics_json({'mejores_parametros': svr_m.best_params_, 'scores_test': svr_scores}, 'metrics_svr.json')
    fig_surf_svr = plot_regression_surface_2d(svr_best, scaler_reg.transform(X_reg), y_reg, scaler=None, title='SVR - Superficie de Regresión 2D')
    save_figure(fig_surf_svr, 'superficie_regresion_svr.png')
    plt.close(fig_surf_svr)

    print("\n" + "=" * 70)
    print("EXPERIMENTOS COMPLETADOS")
    print(f"Figuras guardadas en: {FIGURES_DIR}")
    print(f"Métricas guardadas en: {METRICS_DIR}")
    print("=" * 70)

if __name__ == '__main__':
    main()
