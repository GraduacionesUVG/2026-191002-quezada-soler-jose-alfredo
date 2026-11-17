"""
Entrena un Random Forest y Logistic Regression sobre el dataset de
entrenamiento (por defecto data/dataset_entrenamiento.csv, generado por
build_training_dataset.py) y guarda los modelos en phishing_model.pkl /
phishing_model_lr.pkl.

Ejecutar una sola vez antes de arrancar el microservicio:
    python model_trainer.py

Para usar otro CSV (por ejemplo el dataset legado solo-Guatemala,
data/dataset_comparativo_gt.csv), definir la variable de entorno:
    DATASET_PATH=data/dataset_comparativo_gt.csv python model_trainer.py
"""

import csv
import os
import urllib.parse
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import (train_test_split, learning_curve,
                                      StratifiedKFold, GroupKFold)
from sklearn.metrics import (accuracy_score, precision_score,
                              recall_score, f1_score, classification_report,
                              confusion_matrix)
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
import joblib

from feature_extractor import extract_features, FEATURE_NAMES
from report_generator import save_report, now_iso

np.random.seed(42)

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(BASE_DIR, "data")
REPORTS_DIR = os.path.join(BASE_DIR, "reports")

# Dataset por defecto: el grande y balanceado (phishing global + legitimo
# GT + Tranco global). Si no existe (no se corrio build_training_dataset.py
# todavia), cae de vuelta al dataset legado solo-Guatemala.
_DEFAULT_DATASET = os.path.join(DATA_DIR, "dataset_entrenamiento.csv")
_LEGACY_DATASET = os.path.join(DATA_DIR, "dataset_comparativo_gt.csv")
DATASET_PATH = os.getenv("DATASET_PATH") or (
    _DEFAULT_DATASET if os.path.exists(_DEFAULT_DATASET) else _LEGACY_DATASET
)

LABEL_MAP = {"phishing": 1, "legitimo": 0}

# Curva de aprendizaje: fracciones del conjunto de entrenamiento y folds de CV.
LC_TRAIN_SIZES = np.linspace(0.1, 1.0, 6)
LC_FOLDS = 5


def _load_rows(path: str = DATASET_PATH):
    """Filas validas del CSV como (url, etiqueta 0/1, dominio)."""
    rows = []
    with open(path, newline="", encoding="utf-8") as f:
        for row in csv.DictReader(f):
            url = (row.get("url") or "").strip()
            etiqueta = (row.get("etiqueta") or "").strip().lower()
            if not url or etiqueta not in LABEL_MAP:
                continue
            dominio = (row.get("dominio") or "").strip().lower() \
                or (urllib.parse.urlparse(url).hostname or url).lower()
            rows.append((url, LABEL_MAP[etiqueta], dominio))
    return rows


def load_dataset(path: str = DATASET_PATH):
    """Carga todas las filas validas del CSV (url no vacia + etiqueta reconocida).

    No aplica ningun muestreo ni limite: si una fila no aparece aqui es porque
    le falta 'url' o su 'etiqueta' no es 'phishing'/'legitimo' (revisar
    dataset_stats() para ver cuantas filas se descartaron y por que).
    """
    rows = _load_rows(path)
    return [r[0] for r in rows], [r[1] for r in rows]


def load_domains(path: str = DATASET_PATH):
    """Dominio de cada fila devuelta por load_dataset() (mismo orden)."""
    return [r[2] for r in _load_rows(path)]


def compute_learning_curves(X, y, groups, estimators: dict) -> dict:
    """Curvas de aprendizaje (F1 de entrenamiento vs. validacion) por modelo.

    Sirven para comprobar sobreajuste: si la validacion converge hacia el
    entrenamiento al crecer los datos el modelo generaliza; una brecha grande
    y persistente indica sobreajuste. Se calculan con dos esquemas de CV:

    - "stratified": K-fold estratificado aleatorio (el mismo criterio que el
      split train/test principal).
    - "grouped": GroupKFold por dominio. Como ~80% de las URLs comparten
      dominio con otra, un split aleatorio deja URLs hermanas a ambos lados
      y puede sobreestimar el desempeno; este esquema mide el desempeno
      sobre dominios que el modelo nunca vio.
    """
    X, y, groups = np.asarray(X), np.asarray(y), np.asarray(groups)
    schemes = {
        "stratified": (StratifiedKFold(LC_FOLDS, shuffle=True, random_state=42), None),
        "grouped": (GroupKFold(LC_FOLDS), groups),
    }
    result = {"scoring": "f1", "folds": LC_FOLDS, "cv": {}}
    for scheme, (cv, grp) in schemes.items():
        result["cv"][scheme] = {}
        for key, est in estimators.items():
            sizes, train, valid = learning_curve(
                est, X, y, train_sizes=LC_TRAIN_SIZES, cv=cv, groups=grp,
                scoring="f1", shuffle=True, random_state=42, n_jobs=1)
            result["cv"][scheme][key] = {
                "train_sizes": [int(s) for s in sizes],
                "train_mean": [float(v) for v in train.mean(axis=1)],
                "valid_mean": [float(v) for v in valid.mean(axis=1)],
                "valid_std": [float(v) for v in valid.std(axis=1)],
            }
    return result


def dataset_stats(path: str = DATASET_PATH) -> dict:
    """Estadisticas completas del CSV crudo (incluye filas descartadas y fuentes)."""
    total_rows = 0
    discarded = 0
    sources = {}
    phishing = 0
    legitimo = 0
    with open(path, newline="", encoding="utf-8") as f:
        for row in csv.DictReader(f):
            total_rows += 1
            url = (row.get("url") or "").strip()
            etiqueta = (row.get("etiqueta") or "").strip().lower()
            if not url or etiqueta not in LABEL_MAP:
                discarded += 1
                continue
            fuente = (row.get("fuente") or "desconocida").strip()
            sources[fuente] = sources.get(fuente, 0) + 1
            if etiqueta == "phishing":
                phishing += 1
            else:
                legitimo += 1
    return {
        "path": os.path.relpath(path, BASE_DIR).replace("\\", "/"),
        "total_rows_csv": total_rows,
        "discarded_rows": discarded,
        "total": phishing + legitimo,
        "phishing": phishing,
        "legitimo": legitimo,
        "sources": dict(sorted(sources.items(), key=lambda x: -x[1])),
    }


def train_and_save():
    stats = dataset_stats()
    print(f"Cargando dataset real desde {DATASET_PATH} ...")
    print(f"  Filas en el CSV: {stats['total_rows_csv']} "
          f"(descartadas por url/etiqueta invalida: {stats['discarded_rows']})")
    print(f"  Filas usadas para entrenar+probar: {stats['total']} "
          f"({stats['phishing']} phishing / {stats['legitimo']} legitimo, "
          f"{stats['phishing'] / stats['total'] * 100:.1f}% phishing)")
    print(f"  Fuentes: {stats['sources']}")

    urls, labels = load_dataset()
    assert len(urls) == stats["total"], \
        "load_dataset() y dataset_stats() no coinciden: revisar el CSV"

    print("Extrayendo caracteristicas...")
    X = [extract_features(u) for u in urls]
    y = labels

    X_tr, X_te, y_tr, y_te = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y)

    print("\nEntrenando Random Forest...")
    # min_samples_leaf=5 reduce la brecha train/validacion bajo GroupKFold
    # por dominio (de 2.66 a 1.28 puntos).
    rf = RandomForestClassifier(n_estimators=100, random_state=42, n_jobs=-1,
                                 class_weight="balanced", min_samples_leaf=5)
    rf.fit(X_tr, y_tr)
    y_pred = rf.predict(X_te)

    rf_metrics = {
        "label": "Random Forest",
        "accuracy": accuracy_score(y_te, y_pred),
        "precision": precision_score(y_te, y_pred),
        "recall": recall_score(y_te, y_pred),
        "f1": f1_score(y_te, y_pred),
        "confusion_matrix": confusion_matrix(y_te, y_pred).tolist(),
        "feature_importances": [
            {"name": name, "importance": float(imp)}
            for name, imp in sorted(zip(FEATURE_NAMES, rf.feature_importances_),
                                     key=lambda x: x[1], reverse=True)
        ],
    }

    print("\n=== Random Forest ===")
    print(f"  Accuracy : {rf_metrics['accuracy']:.4f}")
    print(f"  Precision: {rf_metrics['precision']:.4f}")
    print(f"  Recall   : {rf_metrics['recall']:.4f}")
    print(f"  F1       : {rf_metrics['f1']:.4f}")
    print(classification_report(y_te, y_pred, target_names=['Legitimo', 'Phishing']))

    joblib.dump(rf, "phishing_model.pkl")
    print(">> phishing_model.pkl guardado")

    print("\nEntrenando Logistic Regression (comparacion)...")
    lr = Pipeline([('sc', StandardScaler()),
                   ('clf', LogisticRegression(max_iter=1000, random_state=42,
                                               class_weight="balanced"))])
    lr.fit(X_tr, y_tr)
    y_pred_lr = lr.predict(X_te)

    lr_metrics = {
        "label": "Logistic Regression",
        "accuracy": accuracy_score(y_te, y_pred_lr),
        "precision": precision_score(y_te, y_pred_lr),
        "recall": recall_score(y_te, y_pred_lr),
        "f1": f1_score(y_te, y_pred_lr),
        "confusion_matrix": confusion_matrix(y_te, y_pred_lr).tolist(),
    }

    print("\n=== Logistic Regression ===")
    print(f"  Accuracy : {lr_metrics['accuracy']:.4f}")
    print(f"  Precision: {lr_metrics['precision']:.4f}")
    print(f"  Recall   : {lr_metrics['recall']:.4f}")
    print(f"  F1       : {lr_metrics['f1']:.4f}")

    joblib.dump(lr, "phishing_model_lr.pkl")
    print(">> phishing_model_lr.pkl guardado")

    print("\n=== Top 5 Features (RF) ===")
    for f in rf_metrics["feature_importances"][:5]:
        print(f"  {f['name']}: {f['importance']:.4f}")

    print("\nCalculando curvas de aprendizaje (deteccion de sobreajuste)...")
    lc = compute_learning_curves(X, y, load_domains(),
                                  {"random_forest": rf, "logistic_regression": lr})
    for scheme, models in lc["cv"].items():
        for key, c in models.items():
            tr, va = c["train_mean"][-1], c["valid_mean"][-1]
            print(f"  [{scheme:10s}] {key:20s} F1 train={tr:.4f}  "
                  f"valid={va:.4f} (+-{c['valid_std'][-1]:.4f})  brecha={tr - va:.4f}")

    report = {
        "generated_at": now_iso(),
        "dataset": stats,
        "split": {
            "train": len(X_tr),
            "test": len(X_te),
            "test_size": 0.2,
        },
        "models": {
            "random_forest": rf_metrics,
            "logistic_regression": lr_metrics,
        },
        "learning_curve": lc,
    }
    json_path, html_path = save_report(report, REPORTS_DIR)
    print(f"\n>> Reporte guardado en:\n   {json_path}\n   {html_path}")


if __name__ == "__main__":
    train_and_save()
