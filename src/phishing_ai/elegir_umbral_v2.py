"""
Elige el umbral de decision que maximiza F1 para el Random Forest v2.

  1. GroupShuffleSplit(test_size=0.15, random_state=42) agrupado por
     hostname sin "www.".
  2. Entrena un Random Forest con el 85% (misma configuracion que produccion).
  3. Sobre el 15% restante prueba umbrales de 0.01 a 0.99 y calcula F1.
  4. Guarda el mejor umbral y la curva F1-vs-umbral en reports/umbral_v2.json.

La validacion externa no se usa aqui: queda reservada para medir desempeno.
El modelo de este script solo sirve para elegir el umbral; el que se
despliega es el de model_trainer_v2.py.

Uso:
    python elegir_umbral_v2.py
"""

import csv
import json
import os
import urllib.parse

import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import f1_score, precision_score, recall_score
from sklearn.model_selection import GroupShuffleSplit

from feature_extractor import extract_features

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATASET_PATH = os.path.join(BASE_DIR, "data", "dataset_entrenamiento_v2.csv")
REPORTS_DIR = os.path.join(BASE_DIR, "reports")
OUT_PATH = os.path.join(REPORTS_DIR, "umbral_v2.json")

LABEL_MAP = {"phishing": 1, "legitimo": 0}
RANDOM_STATE = 42


def hostname_sin_www(url: str) -> str:
    host = (urllib.parse.urlparse(url).hostname or url).lower()
    return host[4:] if host.startswith("www.") else host


def cargar():
    urls, labels, groups = [], [], []
    with open(DATASET_PATH, newline="", encoding="utf-8") as f:
        for row in csv.DictReader(f):
            url = (row.get("url") or "").strip()
            label = (row.get("label") or "").strip().lower()
            if not url or label not in LABEL_MAP:
                continue
            urls.append(url)
            labels.append(LABEL_MAP[label])
            groups.append(hostname_sin_www(url))
    return urls, np.array(labels), np.array(groups)


def main():
    urls, y, groups = cargar()
    print(f"Dataset v2: {len(urls)} URLs, {len(set(groups))} hostnames distintos.")
    X = np.array([extract_features(u) for u in urls])

    splitter = GroupShuffleSplit(n_splits=1, test_size=0.15, random_state=RANDOM_STATE)
    idx_train, idx_val = next(splitter.split(X, y, groups))
    print(f"Split agrupado por hostname: {len(idx_train)} train (85%) / "
          f"{len(idx_val)} validacion-umbral (15%)")
    hosts_train = set(groups[idx_train])
    hosts_val = set(groups[idx_val])
    print(f"  Hostnames compartidos entre ambos lados (debe ser 0): "
          f"{len(hosts_train & hosts_val)}")

    modelo = RandomForestClassifier(n_estimators=100, random_state=RANDOM_STATE, n_jobs=-1,
                                     class_weight="balanced", min_samples_leaf=5)
    modelo.fit(X[idx_train], y[idx_train])

    proba_val = modelo.predict_proba(X[idx_val])[:, 1]
    y_val = y[idx_val]

    curva = []
    mejor_umbral, mejor_f1 = 0.5, -1.0
    for t in np.arange(0.01, 1.00, 0.01):
        pred = (proba_val >= t).astype(int)
        f1 = f1_score(y_val, pred, zero_division=0)
        curva.append({
            "umbral": round(float(t), 2),
            "f1": round(float(f1), 4),
            "precision": round(float(precision_score(y_val, pred, zero_division=0)), 4),
            "recall": round(float(recall_score(y_val, pred, zero_division=0)), 4),
        })
        if f1 > mejor_f1:
            mejor_f1, mejor_umbral = f1, round(float(t), 2)

    f1_umbral_05 = next(c["f1"] for c in curva if c["umbral"] == 0.5)
    print(f"\nUmbral 0.50 (default)  -> F1 = {f1_umbral_05:.4f}")
    print(f"Umbral {mejor_umbral:.2f} (optimo)  -> F1 = {mejor_f1:.4f}")

    resultado = {
        "umbral_optimo": mejor_umbral,
        "f1_umbral_optimo": round(mejor_f1, 4),
        "f1_umbral_0.5": f1_umbral_05,
        "n_train": int(len(idx_train)),
        "n_val": int(len(idx_val)),
        "curva_f1_vs_umbral": curva,
    }
    os.makedirs(REPORTS_DIR, exist_ok=True)
    with open(OUT_PATH, "w", encoding="utf-8") as f:
        json.dump(resultado, f, ensure_ascii=False, indent=2)
    print(f"\n>> Guardado en {OUT_PATH}")


if __name__ == "__main__":
    main()
