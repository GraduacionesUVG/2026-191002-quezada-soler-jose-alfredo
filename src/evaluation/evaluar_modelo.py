"""
Validacion cruzada de Random Forest y Logistic Regression en dos escenarios:

  - "gt":     phishing_ai/data/dataset_comparativo_gt.csv (300 URLs)
  - "global": phishing_ai/data/dataset_entrenamiento.csv (34,031 URLs)

El escenario global usa dataset_entrenamiento.csv y no
dataset_phishing_global.csv porque este ultimo solo tiene la clase phishing.

  - RepeatedStratifiedKFold(5, 5 repeticiones): metricas resumen.
  - StratifiedKFold(5): predicciones agregadas para la matriz de confusion
    y la curva ROC, donde cada URL debe validarse una sola vez.

Uso:
    cd evaluation
    pip install -r requirements.txt
    python evaluar_modelo.py
"""

import csv
import os
import sys
import urllib.parse

import joblib
import numpy as np

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (accuracy_score, auc, confusion_matrix,
                              f1_score, precision_score, recall_score,
                              roc_auc_score, roc_curve)
from sklearn.model_selection import (RepeatedStratifiedKFold, StratifiedGroupKFold,
                                      StratifiedKFold, cross_val_predict, cross_validate)
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
PHISHING_AI_DIR = os.path.abspath(os.path.join(BASE_DIR, "..", "phishing_ai"))
sys.path.insert(0, PHISHING_AI_DIR)

from feature_extractor import extract_features, FEATURE_NAMES  # noqa: E402

DATA_DIR = os.path.join(PHISHING_AI_DIR, "data")
RESULTS_DIR = os.path.join(BASE_DIR, "resultados")
MODEL_PKL = os.path.join(PHISHING_AI_DIR, "phishing_model.pkl")
EXTERNAL_CSV = os.path.join(BASE_DIR, "validacion_externa.csv")

LABEL_MAP = {"phishing": 1, "legitimo": 0}

SCENARIOS = {
    "gt": os.path.join(DATA_DIR, "dataset_comparativo_gt.csv"),
    "global": os.path.join(DATA_DIR, "dataset_entrenamiento.csv"),
}

N_SPLITS = 5
N_REPEATS = 5
RANDOM_STATE = 42
SCORING = ["accuracy", "precision", "recall", "f1", "roc_auc"]

MODEL_LABELS = {"random_forest": "Random Forest", "logistic_regression": "Logistic Regression"}
MODEL_COLORS = {"random_forest": "#2a78d6", "logistic_regression": "#eb6834"}


def construir_modelos():
    """Misma configuracion que produccion (ver phishing_ai/model_trainer.py)."""
    rf = RandomForestClassifier(n_estimators=100, random_state=RANDOM_STATE, n_jobs=-1,
                                 class_weight="balanced", min_samples_leaf=5)
    lr = Pipeline([("sc", StandardScaler()),
                   ("clf", LogisticRegression(max_iter=1000, random_state=RANDOM_STATE,
                                               class_weight="balanced"))])
    return {"random_forest": rf, "logistic_regression": lr}


def hostname_sin_www(url: str) -> str:
    """Hostname en minusculas, sin el prefijo 'www.' (para agrupar por host
    real: 'www.ejemplo.com' y 'ejemplo.com' cuentan como el mismo grupo)."""
    host = (urllib.parse.urlparse(url).hostname or url).lower()
    return host[4:] if host.startswith("www.") else host


def cargar_dataset(path: str):
    urls, labels = [], []
    with open(path, newline="", encoding="utf-8") as f:
        for row in csv.DictReader(f):
            url = (row.get("url") or "").strip()
            etiqueta = (row.get("etiqueta") or "").strip().lower()
            if not url or etiqueta not in LABEL_MAP:
                continue
            urls.append(url)
            labels.append(LABEL_MAP[etiqueta])
    return urls, labels


def guardar_tabla_csv(path: str, filas: list):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(filas[0].keys()))
        w.writeheader()
        w.writerows(filas)


def graficar_matriz_confusion(path_png: str, cm, titulo: str):
    fig, ax = plt.subplots(figsize=(4.2, 4))
    ax.imshow(cm, cmap="Blues")
    labels = ["Legitimo", "Phishing"]
    ax.set_xticks([0, 1]); ax.set_xticklabels(labels)
    ax.set_yticks([0, 1]); ax.set_yticklabels(labels)
    ax.set_xlabel("Prediccion"); ax.set_ylabel("Real")
    ax.set_title(titulo, fontsize=11)
    vmax = cm.max()
    for i in range(2):
        for j in range(2):
            color = "white" if cm[i, j] > vmax / 2 else "black"
            ax.text(j, i, str(cm[i, j]), ha="center", va="center", color=color, fontsize=13)
    fig.tight_layout()
    fig.savefig(path_png, dpi=300)
    plt.close(fig)


def graficar_roc(path_png: str, curvas: dict, titulo: str):
    fig, ax = plt.subplots(figsize=(5, 4.5))
    for modelo, (fpr, tpr, roc_auc) in curvas.items():
        ax.plot(fpr, tpr, color=MODEL_COLORS[modelo], linewidth=2,
                label=f"{MODEL_LABELS[modelo]} (AUC = {roc_auc:.3f})")
    ax.plot([0, 1], [0, 1], color="#c3c2b7", linestyle="--", linewidth=1, label="Azar (AUC = 0.500)")
    ax.set_xlabel("Tasa de falsos positivos")
    ax.set_ylabel("Tasa de verdaderos positivos")
    ax.set_title(titulo, fontsize=11)
    ax.legend(loc="lower right", fontsize=9)
    fig.tight_layout()
    fig.savefig(path_png, dpi=300)
    plt.close(fig)


def graficar_importancia(path_png: str, importancias: list, titulo: str):
    nombres = [n for n, _ in importancias][::-1]
    valores = [v for _, v in importancias][::-1]
    fig, ax = plt.subplots(figsize=(6, 4.5))
    ax.barh(nombres, valores, color="#2a78d6")
    ax.set_xlabel("Importancia (Random Forest)")
    ax.set_title(titulo, fontsize=11)
    fig.tight_layout()
    fig.savefig(path_png, dpi=300)
    plt.close(fig)


def evaluar_escenario(nombre: str, path: str):
    print(f"\n=== Escenario '{nombre}' ({os.path.relpath(path, BASE_DIR)}) ===")
    urls, labels = cargar_dataset(path)
    y = np.array(labels)
    print(f"  {len(urls)} URLs ({int(y.sum())} phishing / {int(len(y) - y.sum())} legitimo)")
    print("  Extrayendo caracteristicas...")
    X = np.array([extract_features(u) for u in urls])

    cv_repeated = RepeatedStratifiedKFold(n_splits=N_SPLITS, n_repeats=N_REPEATS,
                                           random_state=RANDOM_STATE)
    cv_single = StratifiedKFold(n_splits=N_SPLITS, shuffle=True, random_state=RANDOM_STATE)

    filas_metricas = []
    curvas_roc = {}

    for clave, modelo in construir_modelos().items():
        print(f"  {MODEL_LABELS[clave]}: validacion cruzada repetida "
              f"({N_SPLITS} folds x {N_REPEATS} repeticiones)...")
        res = cross_validate(modelo, X, y, cv=cv_repeated, scoring=SCORING, n_jobs=1)
        fila = {"modelo": MODEL_LABELS[clave]}
        for m in SCORING:
            vals = res[f"test_{m}"]
            fila[f"{m}_media"] = round(float(vals.mean()), 4)
            fila[f"{m}_std"] = round(float(vals.std()), 4)
        filas_metricas.append(fila)
        for m in SCORING:
            print(f"    {m:10s}: {fila[f'{m}_media']*100:6.2f}% "
                  f"(+-{fila[f'{m}_std']*100:.2f})")

        print(f"  {MODEL_LABELS[clave]}: prediccion agregada "
              f"({N_SPLITS}-fold simple) para matriz de confusion y curva ROC...")
        y_pred = cross_val_predict(modelo, X, y, cv=cv_single, n_jobs=1)
        y_proba = cross_val_predict(modelo, X, y, cv=cv_single, n_jobs=1,
                                     method="predict_proba")[:, 1]

        cm = confusion_matrix(y, y_pred)
        guardar_tabla_csv(
            os.path.join(RESULTS_DIR, f"matriz_confusion_{nombre}_{clave}.csv"),
            [{"": "Real Legitimo", "Pred Legitimo": cm[0, 0], "Pred Phishing": cm[0, 1]},
             {"": "Real Phishing", "Pred Legitimo": cm[1, 0], "Pred Phishing": cm[1, 1]}])
        graficar_matriz_confusion(
            os.path.join(RESULTS_DIR, f"matriz_confusion_{nombre}_{clave}.png"), cm,
            f"Matriz de confusion - {MODEL_LABELS[clave]} ({nombre})")

        fpr, tpr, _ = roc_curve(y, y_proba)
        curvas_roc[clave] = (fpr, tpr, auc(fpr, tpr))

    guardar_tabla_csv(os.path.join(RESULTS_DIR, f"metricas_{nombre}.csv"), filas_metricas)
    graficar_roc(os.path.join(RESULTS_DIR, f"roc_{nombre}.png"), curvas_roc,
                 f"Curva ROC - escenario {nombre}")

    print("  Entrenando Random Forest con el 100% del escenario "
          "para importancia de caracteristicas...")
    rf_full = construir_modelos()["random_forest"]
    rf_full.fit(X, y)
    importancias = sorted(zip(FEATURE_NAMES, rf_full.feature_importances_),
                           key=lambda x: x[1], reverse=True)
    guardar_tabla_csv(
        os.path.join(RESULTS_DIR, f"importancia_features_{nombre}.csv"),
        [{"feature": n, "importancia": round(float(v), 4)} for n, v in importancias])
    graficar_importancia(
        os.path.join(RESULTS_DIR, f"importancia_features_{nombre}.png"), importancias,
        f"Importancia de caracteristicas (RF) - escenario {nombre}")

    return filas_metricas


def evaluar_agrupado_hostname(X: np.ndarray, y: np.ndarray, groups: list, etiqueta: str):
    """StratifiedGroupKFold(5) agrupando por hostname (sin 'www.').

    Mas estricto que agrupar por la columna 'dominio' del CSV: dos filas con
    'www.sitio.com' y 'sitio.com' quedan en el mismo grupo, y ninguna URL de
    un mismo host puede aparecer a la vez en train y en validacion. Mide el
    desempeno del modelo sobre hosts que nunca vio, no sobre un split
    aleatorio (que puede dejar URLs "hermanas" del mismo host a ambos lados).
    """
    cv = StratifiedGroupKFold(n_splits=N_SPLITS, shuffle=True, random_state=RANDOM_STATE)
    filas = []
    for clave, modelo in construir_modelos().items():
        print(f"  {MODEL_LABELS[clave]}: StratifiedGroupKFold({N_SPLITS}) por hostname "
              f"[{etiqueta}]...")
        res = cross_validate(modelo, X, y, cv=cv, groups=groups, scoring=SCORING, n_jobs=1)
        fila = {"experimento": etiqueta, "modelo": MODEL_LABELS[clave]}
        for m in SCORING:
            vals = res[f"test_{m}"]
            fila[f"{m}_media"] = round(float(vals.mean()), 4)
            fila[f"{m}_std"] = round(float(vals.std()), 4)
        filas.append(fila)
        for m in SCORING:
            print(f"    {m:10s}: {fila[f'{m}_media']*100:6.2f}% (+-{fila[f'{m}_std']*100:.2f})")
    return filas


def evaluar_agrupado_y_ablacion_global():
    """Solo para el escenario 'global': valida agrupando
    por hostname real y mide cuanto F1 se pierde sin has_https/has_ip_address
    (las dos features que dominan la importancia del RF de produccion)."""
    path = SCENARIOS["global"]
    if not os.path.exists(path):
        print(f"\nAVISO: no se encontro {path}; se omite la validacion agrupada por hostname.")
        return

    print("\n=== Validacion agrupada por hostname (escenario 'global') ===")
    urls, labels = cargar_dataset(path)
    y = np.array(labels)
    X = np.array([extract_features(u) for u in urls])
    groups = [hostname_sin_www(u) for u in urls]
    n_hosts = len(set(groups))
    print(f"  {len(urls)} URLs sobre {n_hosts} hostnames distintos "
          f"(promedio {len(urls) / n_hosts:.1f} URLs/host)")

    filas_completo = evaluar_agrupado_hostname(X, y, groups, "completo_15_features")
    guardar_tabla_csv(os.path.join(RESULTS_DIR, "metricas_global_agrupado_hostname.csv"),
                       filas_completo)

    print("\n=== Ablacion: mismo esquema, sin has_https ni has_ip_address ===")
    idx_quitar = {FEATURE_NAMES.index("has_https"), FEATURE_NAMES.index("has_ip_address")}
    idx_mantener = [i for i in range(len(FEATURE_NAMES)) if i not in idx_quitar]
    X_ablacion = X[:, idx_mantener]
    filas_ablacion = evaluar_agrupado_hostname(X_ablacion, y, groups, "sin_https_ni_ip")
    guardar_tabla_csv(os.path.join(RESULTS_DIR, "metricas_global_agrupado_ablacion.csv"),
                       filas_ablacion)

    f1_completo = {f["modelo"]: f["f1_media"] for f in filas_completo}
    f1_ablacion = {f["modelo"]: f["f1_media"] for f in filas_ablacion}
    filas_diff = []
    print("\n=== Diferencia de F1 (agrupado por hostname): completo vs. sin has_https/has_ip_address ===")
    for modelo in f1_completo:
        diff = f1_completo[modelo] - f1_ablacion[modelo]
        filas_diff.append({
            "modelo": modelo,
            "f1_completo": f1_completo[modelo],
            "f1_sin_https_ip": f1_ablacion[modelo],
            "diferencia_f1": round(diff, 4),
            "diferencia_f1_pts": round(diff * 100, 2),
        })
        print(f"  {modelo:20s} F1 {f1_completo[modelo]*100:.2f}% -> "
              f"{f1_ablacion[modelo]*100:.2f}%  (-{diff*100:.2f} pts)")
    guardar_tabla_csv(os.path.join(RESULTS_DIR, "ablacion_diferencia_f1.csv"), filas_diff)


def evaluar_validacion_externa():
    if not os.path.exists(EXTERNAL_CSV):
        print(f"\n(No se encontro {os.path.relpath(EXTERNAL_CSV, BASE_DIR)}; "
              "se omite la validacion externa. Crea ese archivo con columnas "
              "'url,label' para activarla.)")
        return
    if not os.path.exists(MODEL_PKL):
        print(f"\nADVERTENCIA: no existe {MODEL_PKL}; corre model_trainer.py primero. "
              "Se omite la validacion externa.")
        return

    print(f"\n=== Validacion externa ({os.path.relpath(EXTERNAL_CSV, BASE_DIR)}) ===")
    urls, labels = [], []
    with open(EXTERNAL_CSV, newline="", encoding="utf-8") as f:
        for row in csv.DictReader(f):
            url = (row.get("url") or "").strip()
            raw_label = (row.get("label") or "").strip().lower()
            if not url or raw_label == "":
                continue
            if raw_label in LABEL_MAP:
                label = LABEL_MAP[raw_label]
            elif raw_label in ("0", "1"):
                label = int(raw_label)
            else:
                continue
            urls.append(url)
            labels.append(label)

    if not urls:
        print("  El archivo no tiene filas validas (columnas esperadas: url,label). Se omite.")
        return

    y = np.array(labels)
    print(f"  {len(urls)} URLs ({int(y.sum())} phishing / {int(len(y) - y.sum())} legitimo)")

    modelo = joblib.load(MODEL_PKL)
    X = np.array([extract_features(u) for u in urls])
    y_pred = modelo.predict(X)
    y_proba = modelo.predict_proba(X)[:, 1]

    fila = {
        "n_urls": len(urls),
        "accuracy": round(accuracy_score(y, y_pred), 4),
        "precision": round(precision_score(y, y_pred, zero_division=0), 4),
        "recall": round(recall_score(y, y_pred, zero_division=0), 4),
        "f1": round(f1_score(y, y_pred, zero_division=0), 4),
    }
    if len(set(y.tolist())) == 2:
        fila["roc_auc"] = round(roc_auc_score(y, y_proba), 4)
    else:
        fila["roc_auc"] = "N/D (una sola clase presente)"

    for k, v in fila.items():
        print(f"    {k}: {v}")

    guardar_tabla_csv(os.path.join(RESULTS_DIR, "metricas_validacion_externa.csv"), [fila])


def exportar_importancia_produccion():
    """Importancia de features del RF de produccion (phishing_model.pkl tal
    cual esta desplegado), no de una copia reentrenada."""
    if not os.path.exists(MODEL_PKL):
        print(f"\nAVISO: no existe {MODEL_PKL}; se omite la importancia de produccion.")
        return
    print(f"\n=== Importancia de caracteristicas del modelo de produccion ({MODEL_PKL}) ===")
    modelo = joblib.load(MODEL_PKL)
    importancias = sorted(zip(FEATURE_NAMES, modelo.feature_importances_),
                           key=lambda x: x[1], reverse=True)
    for nombre, valor in importancias:
        print(f"  {nombre:24s} {valor:.4f}")
    guardar_tabla_csv(
        os.path.join(RESULTS_DIR, "importancia_features_produccion.csv"),
        [{"feature": n, "importancia": round(float(v), 4)} for n, v in importancias])


def main():
    os.makedirs(RESULTS_DIR, exist_ok=True)
    for nombre, path in SCENARIOS.items():
        if not os.path.exists(path):
            print(f"AVISO: no se encontro {path}; se omite el escenario '{nombre}'.")
            continue
        evaluar_escenario(nombre, path)

    evaluar_agrupado_y_ablacion_global()

    evaluar_validacion_externa()

    exportar_importancia_produccion()

    print(f"\n>> Resultados guardados en {RESULTS_DIR}")


if __name__ == "__main__":
    main()
