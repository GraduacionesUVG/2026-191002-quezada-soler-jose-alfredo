"""
Entrena Random Forest y Logistic Regression sobre
data/dataset_entrenamiento_v2.csv con la misma configuracion que
model_trainer.py y guarda phishing_model_v2.pkl / phishing_model_lr_v2.pkl.

Uso:
    python model_trainer_v2.py
"""

import csv
import os
import urllib.parse

import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (accuracy_score, classification_report,
                              confusion_matrix, f1_score, precision_score,
                              recall_score)
from sklearn.model_selection import (GroupKFold, StratifiedKFold,
                                      learning_curve, train_test_split)
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
import joblib

from feature_extractor import extract_features, FEATURE_NAMES
from report_generator import save_report, now_iso

np.random.seed(42)

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(BASE_DIR, "data")
REPORTS_DIR = os.path.join(BASE_DIR, "reports")
DATASET_PATH = os.path.join(DATA_DIR, "dataset_entrenamiento_v2.csv")

LABEL_MAP = {"phishing": 1, "legitimo": 0}
LC_TRAIN_SIZES = np.linspace(0.1, 1.0, 6)
LC_FOLDS = 5


def _load_rows(path: str = DATASET_PATH):
    rows = []
    with open(path, newline="", encoding="utf-8") as f:
        for row in csv.DictReader(f):
            url = (row.get("url") or "").strip()
            label = (row.get("label") or "").strip().lower()
            if not url or label not in LABEL_MAP:
                continue
            dominio = (urllib.parse.urlparse(url).hostname or url).lower()
            rows.append((url, LABEL_MAP[label], dominio))
    return rows


def load_dataset(path: str = DATASET_PATH):
    rows = _load_rows(path)
    return [r[0] for r in rows], [r[1] for r in rows]


def load_domains(path: str = DATASET_PATH):
    return [r[2] for r in _load_rows(path)]


def dataset_stats(path: str = DATASET_PATH) -> dict:
    total_rows = 0
    discarded = 0
    sources = {}
    phishing = 0
    legitimo = 0
    with open(path, newline="", encoding="utf-8") as f:
        for row in csv.DictReader(f):
            total_rows += 1
            url = (row.get("url") or "").strip()
            label = (row.get("label") or "").strip().lower()
            if not url or label not in LABEL_MAP:
                discarded += 1
                continue
            fuente = (row.get("fuente") or "desconocida").strip()
            sources[fuente] = sources.get(fuente, 0) + 1
            if label == "phishing":
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


def compute_learning_curves(X, y, groups, estimators: dict) -> dict:
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


def train_and_save():
    stats = dataset_stats()
    print(f"Cargando dataset v2 desde {DATASET_PATH} ...")
    print(f"  Filas en el CSV: {stats['total_rows_csv']} "
          f"(descartadas por url/label invalida: {stats['discarded_rows']})")
    print(f"  Filas usadas para entrenar+probar: {stats['total']} "
          f"({stats['phishing']} phishing / {stats['legitimo']} legitimo, "
          f"{stats['phishing'] / stats['total'] * 100:.1f}% phishing)")
    print(f"  Fuentes: {stats['sources']}")

    urls, labels = load_dataset()
    assert len(urls) == stats["total"]

    print("Extrayendo caracteristicas...")
    X = [extract_features(u) for u in urls]
    y = labels

    X_tr, X_te, y_tr, y_te = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y)

    print("\nEntrenando Random Forest...")
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
    print("\n=== Random Forest (v2) ===")
    print(f"  Accuracy : {rf_metrics['accuracy']:.4f}")
    print(f"  Precision: {rf_metrics['precision']:.4f}")
    print(f"  Recall   : {rf_metrics['recall']:.4f}")
    print(f"  F1       : {rf_metrics['f1']:.4f}")
    print(classification_report(y_te, y_pred, target_names=['Legitimo', 'Phishing']))

    joblib.dump(rf, os.path.join(BASE_DIR, "phishing_model_v2.pkl"))
    print(">> phishing_model_v2.pkl guardado")

    print("\nEntrenando Logistic Regression (v2, comparacion)...")
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
    print("\n=== Logistic Regression (v2) ===")
    print(f"  Accuracy : {lr_metrics['accuracy']:.4f}")
    print(f"  Precision: {lr_metrics['precision']:.4f}")
    print(f"  Recall   : {lr_metrics['recall']:.4f}")
    print(f"  F1       : {lr_metrics['f1']:.4f}")

    joblib.dump(lr, os.path.join(BASE_DIR, "phishing_model_lr_v2.pkl"))
    print(">> phishing_model_lr_v2.pkl guardado")

    print("\n=== Top 5 Features (RF, v2) ===")
    for f in rf_metrics["feature_importances"][:5]:
        print(f"  {f['name']}: {f['importance']:.4f}")

    print("\nCalculando curvas de aprendizaje (v2)...")
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
        "split": {"train": len(X_tr), "test": len(X_te), "test_size": 0.2},
        "models": {"random_forest": rf_metrics, "logistic_regression": lr_metrics},
        "learning_curve": lc,
    }
    json_path, html_path = save_report(report, REPORTS_DIR, prefix="training_report_v2")
    print(f"\n>> Reporte guardado en:\n   {json_path}\n   {html_path}")


if __name__ == "__main__":
    train_and_save()
