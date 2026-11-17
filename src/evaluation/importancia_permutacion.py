"""
Importancia de caracteristicas por permutacion
(sklearn.inspection.permutation_importance) del modelo de produccion
(HistGradientBoosting), que no tiene feature_importances_. Para comparar,
se calcula igual para el Random Forest de la comparacion.

Se mide sobre el 20% de prueba que no se uso para entrenar el modelo final
(mismo train_test_split que comparar_modelos.py: estratificado,
random_state=42). Metrica: caida de F1 al permutar cada caracteristica,
10 repeticiones.

Uso:
    cd evaluation
    python importancia_permutacion.py
"""

import os

import joblib
import numpy as np

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from sklearn.inspection import permutation_importance
from sklearn.model_selection import train_test_split

import evaluar_modelo as v1
from evaluar_modelo_v2 import cargar_dataset_v2

RESULTS_DIR = v1.RESULTS_DIR
MODELS_DIR = os.path.join(v1.PHISHING_AI_DIR, "modelos_comparacion")
DATASET_V2 = os.path.join(v1.DATA_DIR, "dataset_entrenamiento_v2.csv")
MODELOS = {"hist_gradient_boosting": "HistGradientBoosting", "random_forest": "Random Forest"}


def main():
    urls, labels = cargar_dataset_v2(DATASET_V2)
    X = np.array([v1.extract_features(u) for u in urls])
    y = np.array(labels)
    _, X_te, _, y_te = train_test_split(X, y, test_size=0.2,
                                        random_state=v1.RANDOM_STATE, stratify=y)
    print(f"Conjunto de prueba (20%): {len(y_te)} URLs")

    for clave, nombre in MODELOS.items():
        modelo = joblib.load(os.path.join(MODELS_DIR, f"{clave}.pkl"))
        r = permutation_importance(modelo, X_te, y_te, scoring="f1", n_repeats=10,
                                   random_state=v1.RANDOM_STATE, n_jobs=-1)
        orden = np.argsort(-r.importances_mean)
        filas = [{"feature": v1.FEATURE_NAMES[i],
                  "caida_f1_media": round(float(r.importances_mean[i]), 4),
                  "caida_f1_std": round(float(r.importances_std[i]), 4)} for i in orden]
        v1.guardar_tabla_csv(
            os.path.join(RESULTS_DIR, f"importancia_permutacion_{clave}.csv"), filas)

        fig, ax = plt.subplots(figsize=(6, 4.5))
        f = filas[::-1]
        ax.barh([x["feature"] for x in f], [x["caida_f1_media"] for x in f],
                xerr=[x["caida_f1_std"] for x in f], color="#2f9e44", ecolor="#555")
        ax.set_xlabel("Caida de F1 al permutar (media de 10 repeticiones)")
        ax.set_title(f"Importancia por permutacion - {nombre}", fontsize=11)
        fig.tight_layout()
        fig.savefig(os.path.join(RESULTS_DIR, f"importancia_permutacion_{clave}.png"), dpi=300)
        plt.close(fig)

        print(f"\n=== {nombre} ===")
        for x in filas:
            print(f"  {x['feature']:24s} {x['caida_f1_media']:.4f} (+-{x['caida_f1_std']:.4f})")


if __name__ == "__main__":
    main()
