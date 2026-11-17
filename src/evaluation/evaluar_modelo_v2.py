"""
Evaluacion de la v2 (dataset_entrenamiento_v2.csv, phishing_model_v2.pkl)
con la misma metodologia que evaluar_modelo.py: CV repetida, CV agrupada
por hostname, importancia de features y validacion externa a umbral 0.5 y
al umbral elegido en elegir_umbral_v2.py.

Uso:
    cd evaluation
    python evaluar_modelo_v2.py
"""

import csv
import json
import os
import sys

import joblib
import numpy as np

import evaluar_modelo as v1  # reutiliza construir_modelos, graficas, etc.

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
PHISHING_AI_DIR = v1.PHISHING_AI_DIR
DATA_DIR = v1.DATA_DIR
RESULTS_DIR = v1.RESULTS_DIR

DATASET_V2 = os.path.join(DATA_DIR, "dataset_entrenamiento_v2.csv")
MODEL_PKL_V2 = os.path.join(PHISHING_AI_DIR, "phishing_model_v2.pkl")
EXTERNAL_CSV_V2 = os.path.join(BASE_DIR, "validacion_externa_v2.csv")
UMBRAL_JSON = os.path.join(PHISHING_AI_DIR, "reports", "umbral_v2.json")

LABEL_MAP = {"phishing": 1, "legitimo": 0}

extract_features = v1.extract_features
FEATURE_NAMES = v1.FEATURE_NAMES


def cargar_dataset_v2(path: str):
    urls, labels = [], []
    with open(path, newline="", encoding="utf-8") as f:
        for row in csv.DictReader(f):
            url = (row.get("url") or "").strip()
            label = (row.get("label") or "").strip().lower()
            if not url or label not in LABEL_MAP:
                continue
            urls.append(url)
            labels.append(LABEL_MAP[label])
    return urls, labels


def leer_umbral_elegido() -> float:
    if not os.path.exists(UMBRAL_JSON):
        print(f"AVISO: no existe {UMBRAL_JSON}; corre elegir_umbral_v2.py primero. "
              "Se usa 0.5 como umbral 'elegido' por defecto.")
        return 0.5
    with open(UMBRAL_JSON, encoding="utf-8") as f:
        return json.load(f)["umbral_optimo"]


def evaluar_escenario_v2():
    print(f"\n=== Escenario 'v2' ({os.path.relpath(DATASET_V2, BASE_DIR)}) ===")
    urls, labels = cargar_dataset_v2(DATASET_V2)
    y = np.array(labels)
    print(f"  {len(urls)} URLs ({int(y.sum())} phishing / {int(len(y) - y.sum())} legitimo)")
    print("  Extrayendo caracteristicas...")
    X = np.array([extract_features(u) for u in urls])

    from sklearn.metrics import auc, confusion_matrix, roc_curve
    from sklearn.model_selection import (RepeatedStratifiedKFold, StratifiedKFold,
                                          cross_val_predict, cross_validate)

    cv_repeated = RepeatedStratifiedKFold(n_splits=v1.N_SPLITS, n_repeats=v1.N_REPEATS,
                                           random_state=v1.RANDOM_STATE)
    cv_single = StratifiedKFold(n_splits=v1.N_SPLITS, shuffle=True, random_state=v1.RANDOM_STATE)

    filas_metricas = []
    curvas_roc = {}
    for clave, modelo in v1.construir_modelos().items():
        print(f"  {v1.MODEL_LABELS[clave]}: validacion cruzada repetida "
              f"({v1.N_SPLITS} folds x {v1.N_REPEATS} repeticiones)...")
        res = cross_validate(modelo, X, y, cv=cv_repeated, scoring=v1.SCORING, n_jobs=1)
        fila = {"modelo": v1.MODEL_LABELS[clave]}
        for m in v1.SCORING:
            vals = res[f"test_{m}"]
            fila[f"{m}_media"] = round(float(vals.mean()), 4)
            fila[f"{m}_std"] = round(float(vals.std()), 4)
        filas_metricas.append(fila)
        for m in v1.SCORING:
            print(f"    {m:10s}: {fila[f'{m}_media']*100:6.2f}% (+-{fila[f'{m}_std']*100:.2f})")

        y_pred = cross_val_predict(modelo, X, y, cv=cv_single, n_jobs=1)
        y_proba = cross_val_predict(modelo, X, y, cv=cv_single, n_jobs=1,
                                     method="predict_proba")[:, 1]
        cm = confusion_matrix(y, y_pred)
        v1.guardar_tabla_csv(
            os.path.join(RESULTS_DIR, f"matriz_confusion_v2_{clave}.csv"),
            [{"": "Real Legitimo", "Pred Legitimo": cm[0, 0], "Pred Phishing": cm[0, 1]},
             {"": "Real Phishing", "Pred Legitimo": cm[1, 0], "Pred Phishing": cm[1, 1]}])
        v1.graficar_matriz_confusion(
            os.path.join(RESULTS_DIR, f"matriz_confusion_v2_{clave}.png"), cm,
            f"Matriz de confusion - {v1.MODEL_LABELS[clave]} (v2)")

        fpr, tpr, _ = roc_curve(y, y_proba)
        curvas_roc[clave] = (fpr, tpr, auc(fpr, tpr))

    v1.guardar_tabla_csv(os.path.join(RESULTS_DIR, "metricas_v2.csv"), filas_metricas)
    v1.graficar_roc(os.path.join(RESULTS_DIR, "roc_v2.png"), curvas_roc,
                     "Curva ROC - escenario v2")

    print("  Entrenando Random Forest con el 100% del escenario "
          "para importancia de caracteristicas...")
    rf_full = v1.construir_modelos()["random_forest"]
    rf_full.fit(X, y)
    importancias = sorted(zip(FEATURE_NAMES, rf_full.feature_importances_),
                           key=lambda x: x[1], reverse=True)
    v1.guardar_tabla_csv(
        os.path.join(RESULTS_DIR, "importancia_features_v2.csv"),
        [{"feature": n, "importancia": round(float(v), 4)} for n, v in importancias])
    v1.graficar_importancia(
        os.path.join(RESULTS_DIR, "importancia_features_v2.png"), importancias,
        "Importancia de caracteristicas (RF) - escenario v2")

    print("\n=== StratifiedGroupKFold por hostname (v2) ===")
    groups = [v1.hostname_sin_www(u) for u in urls]
    filas_agrupado = v1.evaluar_agrupado_hostname(X, y, groups, "v2_completo")
    v1.guardar_tabla_csv(os.path.join(RESULTS_DIR, "metricas_v2_agrupado_hostname.csv"),
                          filas_agrupado)

    # %% has_https / has_ip_address por clase, en v2
    idx_https = FEATURE_NAMES.index("has_https")
    idx_ip = FEATURE_NAMES.index("has_ip_address")
    filas_pct = []
    for clase, val in (("phishing", 1), ("legitimo", 0)):
        mask = y == val
        filas_pct.append({
            "clase": clase,
            "n": int(mask.sum()),
            "pct_has_https": round(float(X[mask, idx_https].mean() * 100), 2),
            "pct_has_ip_address": round(float(X[mask, idx_ip].mean() * 100), 2),
        })
    v1.guardar_tabla_csv(os.path.join(RESULTS_DIR, "composicion_features_v2.csv"), filas_pct)
    print("\n=== % has_https / has_ip_address por clase (v2) ===")
    for f in filas_pct:
        print(f"  {f['clase']:10s} n={f['n']:6d}  https={f['pct_has_https']:.1f}%  "
              f"ip={f['pct_has_ip_address']:.1f}%")

    return filas_metricas, filas_agrupado


def evaluar_validacion_externa_v2(umbral_elegido: float):
    if not os.path.exists(EXTERNAL_CSV_V2):
        print(f"\n(No se encontro {os.path.relpath(EXTERNAL_CSV_V2, BASE_DIR)}; "
              "corre construir_validacion_externa_v2.py primero. Se omite.)")
        return
    if not os.path.exists(MODEL_PKL_V2):
        print(f"\nAVISO: no existe {MODEL_PKL_V2}; corre model_trainer_v2.py primero.")
        return

    print(f"\n=== Validacion externa v2 ({os.path.relpath(EXTERNAL_CSV_V2, BASE_DIR)}) ===")
    urls, labels = [], []
    with open(EXTERNAL_CSV_V2, newline="", encoding="utf-8") as f:
        for row in csv.DictReader(f):
            url = (row.get("url") or "").strip()
            raw = (row.get("label") or "").strip().lower()
            if not url or raw not in LABEL_MAP:
                continue
            urls.append(url)
            labels.append(LABEL_MAP[raw])

    y = np.array(labels)
    print(f"  {len(urls)} URLs ({int(y.sum())} phishing / {int(len(y) - y.sum())} legitimo)")

    modelo = joblib.load(MODEL_PKL_V2)
    X = np.array([extract_features(u) for u in urls])
    y_proba = modelo.predict_proba(X)[:, 1]

    from sklearn.metrics import (accuracy_score, confusion_matrix, f1_score,
                                  precision_score, recall_score, roc_auc_score)

    filas = []
    for etiqueta_umbral, t in (("0.5", 0.5), ("elegido", umbral_elegido)):
        y_pred = (y_proba >= t).astype(int)
        cm = confusion_matrix(y, y_pred)
        fila = {
            "umbral": etiqueta_umbral,
            "valor_umbral": t,
            "n_urls": len(urls),
            "accuracy": round(accuracy_score(y, y_pred), 4),
            "precision": round(precision_score(y, y_pred, zero_division=0), 4),
            "recall": round(recall_score(y, y_pred, zero_division=0), 4),
            "f1": round(f1_score(y, y_pred, zero_division=0), 4),
            "roc_auc": round(roc_auc_score(y, y_proba), 4),
        }
        filas.append(fila)
        print(f"  [umbral {etiqueta_umbral} = {t}] accuracy={fila['accuracy']} "
              f"precision={fila['precision']} recall={fila['recall']} f1={fila['f1']} "
              f"roc_auc={fila['roc_auc']}")
        v1.guardar_tabla_csv(
            os.path.join(RESULTS_DIR, f"matriz_confusion_externa_v2_umbral_{etiqueta_umbral}.csv"),
            [{"": "Real Legitimo", "Pred Legitimo": cm[0, 0], "Pred Phishing": cm[0, 1]},
             {"": "Real Phishing", "Pred Legitimo": cm[1, 0], "Pred Phishing": cm[1, 1]}])

    v1.guardar_tabla_csv(os.path.join(RESULTS_DIR, "metricas_validacion_externa_v2.csv"), filas)


def exportar_importancia_produccion_v2():
    if not os.path.exists(MODEL_PKL_V2):
        return
    modelo = joblib.load(MODEL_PKL_V2)
    importancias = sorted(zip(FEATURE_NAMES, modelo.feature_importances_),
                           key=lambda x: x[1], reverse=True)
    v1.guardar_tabla_csv(
        os.path.join(RESULTS_DIR, "importancia_features_produccion_v2.csv"),
        [{"feature": n, "importancia": round(float(val), 4)} for n, val in importancias])


def main():
    os.makedirs(RESULTS_DIR, exist_ok=True)
    if not os.path.exists(DATASET_V2):
        print(f"ERROR: no existe {DATASET_V2}; corre build_training_dataset_v2.py primero.",
              file=sys.stderr)
        sys.exit(1)

    evaluar_escenario_v2()
    umbral = leer_umbral_elegido()
    print(f"\nUmbral elegido (de {UMBRAL_JSON}): {umbral}")
    evaluar_validacion_externa_v2(umbral)
    exportar_importancia_produccion_v2()

    print(f"\n>> Resultados v2 guardados en {RESULTS_DIR}")


if __name__ == "__main__":
    main()
