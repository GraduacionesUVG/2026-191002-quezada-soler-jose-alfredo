"""
Compara 6 clasificadores sobre el dataset v2 con el mismo protocolo que
evaluar_modelo_v2.py: CV repetida 5x5, CV agrupada por hostname, umbral
propio por modelo (split 85/15 agrupado, maximiza F1), validacion externa
con McNemar contra Random Forest y latencia de inferencia en proceso.

Modelos: Random Forest, Logistic Regression, HistGradientBoosting, SVM RBF,
KNN (k=15) y MLP (64, 32). Mismas 15 features, sin ajuste de hiperparametros.

Los resultados se escriben con prefijo "comparacion_" en evaluation/resultados/.

Uso:
    cd evaluation
    python comparar_modelos.py
"""

import csv
import json
import os
import statistics
import sys
import time

import joblib
import numpy as np

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from scipy.stats import binomtest
from sklearn.calibration import CalibratedClassifierCV
from sklearn.ensemble import HistGradientBoostingClassifier, RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (accuracy_score, auc, confusion_matrix, f1_score,
                              precision_score, recall_score, roc_auc_score, roc_curve)
from sklearn.model_selection import (GroupShuffleSplit, RepeatedStratifiedKFold,
                                      StratifiedGroupKFold, StratifiedKFold,
                                      cross_val_predict, cross_validate, train_test_split)
from sklearn.neighbors import KNeighborsClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC

import evaluar_modelo as v1
from evaluar_modelo_v2 import cargar_dataset_v2

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
PHISHING_AI_DIR = v1.PHISHING_AI_DIR
RESULTS_DIR = v1.RESULTS_DIR
DATASET_V2 = os.path.join(v1.DATA_DIR, "dataset_entrenamiento_v2.csv")
EXTERNAL_CSV_V2 = os.path.join(BASE_DIR, "validacion_externa_v2.csv")
MODELS_DIR = os.path.join(PHISHING_AI_DIR, "modelos_comparacion")
UMBRALES_JSON = os.path.join(RESULTS_DIR, "comparacion_umbrales.json")

extract_features = v1.extract_features
RANDOM_STATE = v1.RANDOM_STATE
N_SPLITS, N_REPEATS, SCORING = v1.N_SPLITS, v1.N_REPEATS, v1.SCORING
N_JOBS = -1  # folds en paralelo; los resultados no cambian (semillas fijas)

MODEL_LABELS = {
    "random_forest": "Random Forest",
    "logistic_regression": "Logistic Regression",
    "hist_gradient_boosting": "HistGradientBoosting",
    "svm_rbf": "SVM (RBF)",
    "knn": "KNN (k=15)",
    "mlp": "MLP (64, 32)",
}
MODEL_COLORS = {
    "random_forest": "#2a78d6",
    "logistic_regression": "#eb6834",
    "hist_gradient_boosting": "#2f9e44",
    "svm_rbf": "#9c36b5",
    "knn": "#868e96",
    "mlp": "#d6336c",
}


def construir_modelos() -> dict:
    """RF y LR identicos a produccion (evaluar_modelo.construir_modelos);
    los otros 4 con hiperparametros fijados a priori, sin busqueda."""
    modelos = dict(v1.construir_modelos())
    modelos["hist_gradient_boosting"] = HistGradientBoostingClassifier(
        random_state=RANDOM_STATE, class_weight="balanced")
    modelos["svm_rbf"] = Pipeline([
        ("sc", StandardScaler()),
        # Platt scaling (equivalente a SVC(probability=True), que esta
        # deprecado en scikit-learn 1.9) para tener predict_proba y poder
        # aplicar umbral y ROC igual que al resto.
        ("clf", CalibratedClassifierCV(
            SVC(kernel="rbf", C=1.0, gamma="scale", class_weight="balanced",
                random_state=RANDOM_STATE),
            method="sigmoid", cv=5, ensemble=False)),
    ])
    modelos["knn"] = Pipeline([
        ("sc", StandardScaler()),
        ("clf", KNeighborsClassifier(n_neighbors=15)),
    ])
    modelos["mlp"] = Pipeline([
        ("sc", StandardScaler()),
        ("clf", MLPClassifier(hidden_layer_sizes=(64, 32), early_stopping=True,
                              max_iter=500, random_state=RANDOM_STATE)),
    ])
    return modelos


def resumen_cv(res: dict) -> dict:
    fila = {}
    for m in SCORING:
        vals = res[f"test_{m}"]
        fila[f"{m}_media"] = round(float(vals.mean()), 4)
        fila[f"{m}_std"] = round(float(vals.std()), 4)
    fila["fit_time_medio_s"] = round(float(res["fit_time"].mean()), 2)
    return fila


def imprimir(fila: dict):
    print("    " + "  ".join(f"{m}={fila[f'{m}_media']:.4f}(+-{fila[f'{m}_std']:.4f})"
                             for m in SCORING))


def guardar_matriz(path_csv: str, cm):
    v1.guardar_tabla_csv(path_csv, [
        {"": "Real Legitimo", "Pred Legitimo": cm[0, 0], "Pred Phishing": cm[0, 1]},
        {"": "Real Phishing", "Pred Legitimo": cm[1, 0], "Pred Phishing": cm[1, 1]}])


def graficar_roc(path_png: str, curvas: dict, titulo: str):
    fig, ax = plt.subplots(figsize=(5.5, 5))
    for clave, (fpr, tpr, a) in sorted(curvas.items(), key=lambda kv: -kv[1][2]):
        ax.plot(fpr, tpr, color=MODEL_COLORS[clave], linewidth=1.8,
                label=f"{MODEL_LABELS[clave]} (AUC = {a:.3f})")
    ax.plot([0, 1], [0, 1], color="#c3c2b7", linestyle="--", linewidth=1,
            label="Azar (AUC = 0.500)")
    ax.set_xlabel("Tasa de falsos positivos")
    ax.set_ylabel("Tasa de verdaderos positivos")
    ax.set_title(titulo, fontsize=11)
    ax.legend(loc="lower right", fontsize=8)
    fig.tight_layout()
    fig.savefig(path_png, dpi=300)
    plt.close(fig)


def graficar_barras_f1(path_png: str, filas_cv, filas_grp, filas_ext):
    claves = list(MODEL_LABELS)
    cv = {f["clave"]: f["f1_media"] for f in filas_cv}
    grp = {f["clave"]: f["f1_media"] for f in filas_grp}
    ext = {f["clave"]: f["f1"] for f in filas_ext if f["umbral"] == "elegido"}
    x = np.arange(len(claves))
    w = 0.27
    fig, ax = plt.subplots(figsize=(8, 4.5))
    for i, (nombre, d, color) in enumerate([
            ("CV 5x5", cv, "#2a78d6"),
            ("Agrupada por hostname", grp, "#eb6834"),
            ("Validacion externa (umbral elegido)", ext, "#2f9e44")]):
        ax.bar(x + (i - 1) * w, [d[c] for c in claves], w, label=nombre, color=color)
    ax.set_xticks(x)
    ax.set_xticklabels([MODEL_LABELS[c] for c in claves], rotation=20, ha="right", fontsize=9)
    ax.set_ylabel("F1")
    lo = min(min(cv.values()), min(grp.values()), min(ext.values()))
    ax.set_ylim(max(0.0, lo - 0.05), 1.0)
    ax.set_title("F1 por modelo y esquema de validacion", fontsize=11)
    ax.legend(fontsize=8, loc="lower right")
    fig.tight_layout()
    fig.savefig(path_png, dpi=300)
    plt.close(fig)


def elegir_umbral(modelo, X, y, groups) -> dict:
    """Misma metodologia que phishing_ai/elegir_umbral_v2.py, para cualquier modelo."""
    splitter = GroupShuffleSplit(n_splits=1, test_size=0.15, random_state=RANDOM_STATE)
    idx_tr, idx_val = next(splitter.split(X, y, groups))
    modelo.fit(X[idx_tr], y[idx_tr])
    proba = modelo.predict_proba(X[idx_val])[:, 1]
    mejor_t, mejor_f1 = 0.5, -1.0
    for t in np.arange(0.01, 1.00, 0.01):
        f1 = f1_score(y[idx_val], (proba >= t).astype(int), zero_division=0)
        if f1 > mejor_f1:
            mejor_f1, mejor_t = f1, round(float(t), 2)
    f1_05 = f1_score(y[idx_val], (proba >= 0.5).astype(int), zero_division=0)
    return {"umbral_optimo": mejor_t, "f1_umbral_optimo": round(float(mejor_f1), 4),
            "f1_umbral_0.5": round(float(f1_05), 4),
            "n_train": int(len(idx_tr)), "n_val": int(len(idx_val))}


def mcnemar_exacto(y, pred_a, pred_b) -> tuple:
    """McNemar exacto (binomial) sobre los aciertos discordantes."""
    ok_a, ok_b = pred_a == y, pred_b == y
    b = int(np.sum(ok_a & ~ok_b))
    c = int(np.sum(~ok_a & ok_b))
    if b + c == 0:
        return b, c, 1.0
    return b, c, float(binomtest(b, b + c, 0.5).pvalue)


def latencia_en_proceso(modelo, urls, n=500) -> dict:
    """extract_features + predict_proba de una URL por llamada (como /analyze)."""
    muestra = (urls * (n // len(urls) + 1))[:n]
    for u in muestra[:20]:  # calentamiento
        modelo.predict_proba(np.array(extract_features(u)).reshape(1, -1))
    tiempos = []
    for u in muestra:
        t0 = time.perf_counter()
        modelo.predict_proba(np.array(extract_features(u)).reshape(1, -1))
        tiempos.append((time.perf_counter() - t0) * 1000)
    orden = sorted(tiempos)
    return {"n": n,
            "promedio_ms": round(statistics.mean(tiempos), 3),
            "mediana_ms": round(statistics.median(tiempos), 3),
            "p95_ms": round(orden[min(n - 1, int(round(0.95 * (n - 1))))], 3),
            "maximo_ms": round(max(tiempos), 3)}


def cargar_externa():
    urls, labels = [], []
    with open(EXTERNAL_CSV_V2, newline="", encoding="utf-8") as f:
        for row in csv.DictReader(f):
            url = (row.get("url") or "").strip()
            raw = (row.get("label") or "").strip().lower()
            if url and raw in v1.LABEL_MAP:
                urls.append(url)
                labels.append(v1.LABEL_MAP[raw])
    return urls, np.array(labels)


def main():
    for p in (DATASET_V2, EXTERNAL_CSV_V2):
        if not os.path.exists(p):
            print(f"ERROR: no existe {p}", file=sys.stderr)
            sys.exit(1)
    os.makedirs(RESULTS_DIR, exist_ok=True)
    os.makedirs(MODELS_DIR, exist_ok=True)

    urls, labels = cargar_dataset_v2(DATASET_V2)
    y = np.array(labels)
    groups = np.array([v1.hostname_sin_www(u) for u in urls])
    print(f"Dataset v2: {len(urls)} URLs ({int(y.sum())} phishing / "
          f"{int(len(y) - y.sum())} legitimo), {len(set(groups))} hostnames.")
    X = np.array([extract_features(u) for u in urls])

    ext_urls, y_ext = cargar_externa()
    X_ext = np.array([extract_features(u) for u in ext_urls])
    print(f"Validacion externa: {len(ext_urls)} URLs ({int(y_ext.sum())} phishing / "
          f"{int(len(y_ext) - y_ext.sum())} legitimo)")

    cv_rep = RepeatedStratifiedKFold(n_splits=N_SPLITS, n_repeats=N_REPEATS,
                                      random_state=RANDOM_STATE)
    cv_single = StratifiedKFold(n_splits=N_SPLITS, shuffle=True, random_state=RANDOM_STATE)
    cv_grp = StratifiedGroupKFold(n_splits=N_SPLITS, shuffle=True, random_state=RANDOM_STATE)

    filas_cv, filas_grp, filas_ext, filas_lat = [], [], [], []
    umbrales, curvas_roc, preds_ext = {}, {}, {}

    for clave in MODEL_LABELS:
        etiqueta = MODEL_LABELS[clave]
        print(f"\n=== {etiqueta} ===")
        t_modelo = time.time()

        print(f"  [1/5] Validacion cruzada repetida {N_SPLITS}x{N_REPEATS}...")
        res = cross_validate(construir_modelos()[clave], X, y, cv=cv_rep,
                             scoring=SCORING, n_jobs=N_JOBS)
        fila = {"clave": clave, "modelo": etiqueta, **resumen_cv(res)}
        filas_cv.append(fila)
        imprimir(fila)

        # Prediccion agregada (cada URL validada una vez) para matriz y ROC
        proba_oof = cross_val_predict(construir_modelos()[clave], X, y, cv=cv_single,
                                      n_jobs=N_JOBS, method="predict_proba")[:, 1]
        cm = confusion_matrix(y, (proba_oof > 0.5).astype(int))
        guardar_matriz(os.path.join(RESULTS_DIR, f"comparacion_matriz_cv_{clave}.csv"), cm)
        fpr, tpr, _ = roc_curve(y, proba_oof)
        curvas_roc[clave] = (fpr, tpr, auc(fpr, tpr))

        print(f"  [2/5] StratifiedGroupKFold({N_SPLITS}) por hostname...")
        res = cross_validate(construir_modelos()[clave], X, y, cv=cv_grp, groups=groups,
                             scoring=SCORING, n_jobs=N_JOBS)
        fila = {"clave": clave, "modelo": etiqueta, **resumen_cv(res)}
        filas_grp.append(fila)
        imprimir(fila)

        print("  [3/5] Umbral de decision (split 85/15 agrupado por hostname)...")
        umbrales[clave] = elegir_umbral(construir_modelos()[clave], X, y, groups)
        print(f"    umbral={umbrales[clave]['umbral_optimo']}  "
              f"F1={umbrales[clave]['f1_umbral_optimo']} (a 0.5: {umbrales[clave]['f1_umbral_0.5']})")

        print("  [4/5] Modelo final (80% estratificado, como model_trainer_v2.py)...")
        X_tr, X_te, y_tr, y_te = train_test_split(X, y, test_size=0.2,
                                                  random_state=RANDOM_STATE, stratify=y)
        modelo = construir_modelos()[clave]
        modelo.fit(X_tr, y_tr)
        pkl = os.path.join(MODELS_DIR, f"{clave}.pkl")
        joblib.dump(modelo, pkl)
        print(f"    F1 hold-out 20%: {f1_score(y_te, modelo.predict(X_te)):.4f}  -> {os.path.relpath(pkl, BASE_DIR)}")

        print("  [5/5] Validacion externa y latencia en proceso...")
        proba_ext = modelo.predict_proba(X_ext)[:, 1]
        for nombre_t, t in (("0.5", 0.5), ("elegido", umbrales[clave]["umbral_optimo"])):
            pred = (proba_ext >= t).astype(int)
            preds_ext[(clave, nombre_t)] = pred
            cm = confusion_matrix(y_ext, pred)
            filas_ext.append({
                "clave": clave, "modelo": etiqueta, "umbral": nombre_t, "valor_umbral": t,
                "n_urls": len(ext_urls),
                "accuracy": round(accuracy_score(y_ext, pred), 4),
                "precision": round(precision_score(y_ext, pred, zero_division=0), 4),
                "recall": round(recall_score(y_ext, pred, zero_division=0), 4),
                "f1": round(f1_score(y_ext, pred, zero_division=0), 4),
                "roc_auc": round(roc_auc_score(y_ext, proba_ext), 4),
                "tn": int(cm[0, 0]), "fp": int(cm[0, 1]), "fn": int(cm[1, 0]), "tp": int(cm[1, 1]),
            })
            f = filas_ext[-1]
            print(f"    externa umbral {nombre_t}={t}: F1={f['f1']} precision={f['precision']} "
                  f"recall={f['recall']} AUC={f['roc_auc']}")
        lat = latencia_en_proceso(modelo, ext_urls)
        filas_lat.append({"clave": clave, "modelo": etiqueta, **lat})
        print(f"    latencia en proceso: mediana={lat['mediana_ms']} ms  p95={lat['p95_ms']} ms")
        print(f"  ({time.time() - t_modelo:.0f} s)")

    # McNemar de cada modelo contra Random Forest (produccion), umbral elegido
    for f in filas_ext:
        if f["clave"] == "random_forest":
            f["mcnemar_vs_rf_b"] = f["mcnemar_vs_rf_c"] = ""
            f["mcnemar_vs_rf_p"] = ""
            continue
        b, c, p = mcnemar_exacto(y_ext, preds_ext[(f["clave"], f["umbral"])],
                                 preds_ext[("random_forest", f["umbral"])])
        # b = URLs que este modelo acierta y RF falla; c = al reves
        f["mcnemar_vs_rf_b"], f["mcnemar_vs_rf_c"], f["mcnemar_vs_rf_p"] = b, c, round(p, 4)

    v1.guardar_tabla_csv(os.path.join(RESULTS_DIR, "comparacion_metricas_cv_5x5.csv"), filas_cv)
    v1.guardar_tabla_csv(os.path.join(RESULTS_DIR, "comparacion_metricas_agrupado_hostname.csv"),
                          filas_grp)
    v1.guardar_tabla_csv(os.path.join(RESULTS_DIR, "comparacion_metricas_validacion_externa.csv"),
                          filas_ext)
    v1.guardar_tabla_csv(os.path.join(RESULTS_DIR, "comparacion_latencia_en_proceso.csv"),
                          filas_lat)
    with open(UMBRALES_JSON, "w", encoding="utf-8") as fh:
        json.dump(umbrales, fh, ensure_ascii=False, indent=2)
    graficar_roc(os.path.join(RESULTS_DIR, "comparacion_roc_cv.png"), curvas_roc,
                 "Curva ROC (5-fold) - comparacion de modelos, dataset v2")
    graficar_barras_f1(os.path.join(RESULTS_DIR, "comparacion_f1.png"),
                       filas_cv, filas_grp, filas_ext)

    print("\n=== Resumen (F1) ===")
    print(f"  {'modelo':24s} {'CV 5x5':>8s} {'hostname':>9s} {'externa@t':>10s} {'umbral':>7s}")
    for clave in MODEL_LABELS:
        cv = next(f for f in filas_cv if f["clave"] == clave)["f1_media"]
        gr = next(f for f in filas_grp if f["clave"] == clave)["f1_media"]
        ex = next(f for f in filas_ext if f["clave"] == clave and f["umbral"] == "elegido")
        print(f"  {MODEL_LABELS[clave]:24s} {cv:8.4f} {gr:9.4f} {ex['f1']:10.4f} {ex['valor_umbral']:7.2f}")
    print(f"\n>> Resultados en {RESULTS_DIR} (prefijo comparacion_)")
    print(f">> Modelos en {MODELS_DIR}")
    print(">> Siguiente paso: python seleccionar_modelo.py")


if __name__ == "__main__":
    main()
