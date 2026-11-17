"""
Figura para el documento: F1-score de los seis modelos en la validacion
externa (562 URLs), cada uno con su umbral de decision. Lee
resultados/comparacion_metricas_validacion_externa.csv (no recalcula nada).

Uso:
    cd evaluation
    python figura_f1_externa.py
"""

import csv
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
RESULTS_DIR = os.path.join(BASE_DIR, "resultados")
CSV_IN = os.path.join(RESULTS_DIR, "comparacion_metricas_validacion_externa.csv")
PNG_OUT = os.path.join(RESULTS_DIR, "figura_f1_validacion_externa.png")

NOMBRES = {
    "hist_gradient_boosting": "HistGradientBoosting",
    "svm_rbf": "SVM (RBF)",
    "random_forest": "Random Forest",
    "logistic_regression": "Regresión logística",
    "mlp": "MLP (64, 32)",
    "knn": "KNN (k = 15)",
}
META_HIPOTESIS = 0.85

DESTACADO = "#2a78d6"   # modelo seleccionado
NEUTRO = "#a9b1bb"      # resto
TINTA = "#1f2328"
TINTA_SEC = "#57606a"
GUIA = "#d0d7de"


def main():
    with open(CSV_IN, newline="", encoding="utf-8") as f:
        filas = [r for r in csv.DictReader(f) if r["umbral"] == "elegido"]
    filas.sort(key=lambda r: float(r["f1"]))  # el mayor queda arriba
    ganador = max(filas, key=lambda r: float(r["f1"]))["clave"]

    plt.rcParams["font.family"] = "Times New Roman"
    fig, ax = plt.subplots(figsize=(6.5, 3.4))

    y = range(len(filas))
    valores = [float(r["f1"]) for r in filas]
    colores = [DESTACADO if r["clave"] == ganador else NEUTRO for r in filas]
    ax.barh(y, valores, height=0.62, color=colores, zorder=2)

    for i, (r, v) in enumerate(zip(filas, valores)):
        # Las cifras empiezan siempre a la derecha de la linea de la meta.
        ax.text(max(v, META_HIPOTESIS) + 0.012, i, f"{v:.3f}", va="center", ha="left", fontsize=10.5,
                color=TINTA, fontweight="bold" if r["clave"] == ganador else "normal")

    ax.axvline(META_HIPOTESIS, color=TINTA_SEC, linestyle=(0, (4, 3)), linewidth=1, zorder=3)
    ax.text(META_HIPOTESIS - 0.012, len(filas) - 0.45, f"Criterio de aceptación ({META_HIPOTESIS:.2f})",
            ha="right", va="bottom", fontsize=9.5, color=TINTA_SEC)

    ax.set_yticks(list(y))
    ax.set_yticklabels(
        [f"{NOMBRES[r['clave']]}  (umbral {float(r['valor_umbral']):.2f})" for r in filas],
        fontsize=10.5, color=TINTA)
    for lbl, r in zip(ax.get_yticklabels(), filas):
        if r["clave"] == ganador:
            lbl.set_fontweight("bold")

    ax.set_xlim(0, 1.0)
    ax.set_ylim(-0.6, len(filas) + 0.05)
    ax.set_xticks([0, 0.2, 0.4, 0.6, 0.8, 1.0])
    ax.set_xlabel("F1-score en validación externa (n = 562 URL)", fontsize=10.5, color=TINTA)
    ax.tick_params(axis="x", labelsize=10, colors=TINTA_SEC, length=0)
    ax.tick_params(axis="y", length=0)
    ax.grid(axis="x", color=GUIA, linewidth=0.6, zorder=0)
    for lado in ("top", "right", "left"):
        ax.spines[lado].set_visible(False)
    ax.spines["bottom"].set_color(GUIA)

    fig.tight_layout()
    fig.savefig(PNG_OUT, dpi=300, facecolor="white")
    plt.close(fig)
    print(f">> {PNG_OUT}")


if __name__ == "__main__":
    main()
