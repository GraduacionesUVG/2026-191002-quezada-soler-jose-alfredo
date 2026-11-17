"""
Mide el solapamiento por hostname entre validacion_externa_v2.csv y el
dataset de entrenamiento v2, y la tasa de falsos positivos por subgrupo.

El lado legitimo de la validacion externa reutiliza URLs .gt que comparten
hostname (no URL exacta) con el entrenamiento.

Uso:
    cd evaluation
    python analisis_solapamiento_externa.py
"""

import csv
import json
import os

import joblib
import numpy as np

import evaluar_modelo as v1
from evaluar_modelo_v2 import cargar_dataset_v2

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
RESULTS_DIR = v1.RESULTS_DIR
MODELS_DIR = os.path.join(v1.PHISHING_AI_DIR, "modelos_comparacion")
DATASET_V2 = os.path.join(v1.DATA_DIR, "dataset_entrenamiento_v2.csv")
EXTERNAL = os.path.join(BASE_DIR, "validacion_externa_v2.csv")
UMBRALES = os.path.join(RESULTS_DIR, "comparacion_umbrales.json")
CLAVES = ["random_forest", "logistic_regression", "hist_gradient_boosting",
          "svm_rbf", "knn", "mlp"]


def main():
    train_urls, _ = cargar_dataset_v2(DATASET_V2)
    train_urls_set = set(train_urls)
    train_hosts = {v1.hostname_sin_www(u) for u in train_urls}

    urls, y = [], []
    with open(EXTERNAL, newline="", encoding="utf-8") as f:
        for row in csv.DictReader(f):
            urls.append(row["url"].strip())
            y.append(v1.LABEL_MAP[row["label"].strip().lower()])
    y = np.array(y)
    compartido = np.array([v1.hostname_sin_www(u) in train_hosts for u in urls])
    identica = sum(u in train_urls_set for u in urls)

    resumen = {
        "n_externa": len(urls),
        "url_identica_en_entrenamiento": identica,
        "phishing_total": int((y == 1).sum()),
        "phishing_hostname_compartido": int(((y == 1) & compartido).sum()),
        "legitimo_total": int((y == 0).sum()),
        "legitimo_hostname_compartido": int(((y == 0) & compartido).sum()),
        "hostnames_legitimos_distintos": len({v1.hostname_sin_www(u)
                                              for u, l in zip(urls, y) if l == 0}),
    }
    print(json.dumps(resumen, indent=2))

    with open(UMBRALES, encoding="utf-8") as f:
        umbrales = json.load(f)
    X = np.array([v1.extract_features(u) for u in urls])
    leg_comp = (y == 0) & compartido
    leg_nuevo = (y == 0) & ~compartido
    filas = []
    for clave in CLAVES:
        t = umbrales[clave]["umbral_optimo"]
        pred = (joblib.load(os.path.join(MODELS_DIR, f"{clave}.pkl"))
                .predict_proba(X)[:, 1] >= t).astype(int)
        filas.append({
            "clave": clave, "umbral": t,
            "fp_legitimo_host_compartido": int(pred[leg_comp].sum()),
            "n_legitimo_host_compartido": int(leg_comp.sum()),
            "fpr_host_compartido": round(float(pred[leg_comp].mean()), 4),
            "fp_legitimo_host_nuevo": int(pred[leg_nuevo].sum()),
            "n_legitimo_host_nuevo": int(leg_nuevo.sum()),
            "fpr_host_nuevo": round(float(pred[leg_nuevo].mean()), 4),
            "recall_phishing": round(float(pred[y == 1].mean()), 4),
        })
        print(filas[-1])

    v1.guardar_tabla_csv(os.path.join(RESULTS_DIR, "solapamiento_externa_por_modelo.csv"), filas)
    with open(os.path.join(RESULTS_DIR, "solapamiento_externa_resumen.json"), "w",
              encoding="utf-8") as f:
        json.dump(resumen, f, ensure_ascii=False, indent=2)


if __name__ == "__main__":
    main()
