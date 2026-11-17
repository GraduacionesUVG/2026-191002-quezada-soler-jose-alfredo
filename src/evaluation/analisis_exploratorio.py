"""Análisis exploratorio de los conjuntos de datos (v1, v2, guatemalteco)."""
import sys
import os; BASE = os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0, os.path.join(BASE, "..", "phishing_ai"))
import numpy as np, pandas as pd
import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
from feature_extractor import extract_features, FEATURE_NAMES

OUT = os.path.join(BASE, "resultados", "eda"); D = os.path.join(BASE, "..", "phishing_ai", "data"); os.makedirs(OUT, exist_ok=True)
plt.rcParams.update({"font.size": 9, "axes.spines.top": False, "axes.spines.right": False})
C_PH, C_LE = "#c92a2a", "#2a78d6"

def cargar(path, col):
    df = pd.read_csv(path)
    df = df[df[col].str.lower().isin(["phishing", "legitimo"])].copy()
    df["clase"] = df[col].str.lower()
    F = pd.DataFrame([extract_features(u) for u in df.url], columns=FEATURE_NAMES, index=df.index)
    return pd.concat([df[["url", "fuente", "clase"]], F], axis=1)

v1 = cargar(f"{D}/dataset_entrenamiento.csv", "etiqueta")
v2 = cargar(f"{D}/dataset_entrenamiento_v2.csv", "label")
gt = cargar(f"{D}/dataset_comparativo_gt.csv", "etiqueta")
ext = cargar(os.path.join(BASE, "validacion_externa_v2.csv"), "label")

# 1. Distribución de clases
dist = pd.DataFrame({n: d.clase.value_counts() for n, d in
                     [("v1", v1), ("v2", v2), ("Guatemalteco", gt), ("Validación externa", ext)]}).T
dist["total"] = dist.sum(1); dist["% phishing"] = (100 * dist.phishing / dist.total).round(1)
dist.to_csv(f"{OUT}/eda_distribucion_clases.csv"); print(dist)
fig, ax = plt.subplots(figsize=(6.5, 3.4)); x = np.arange(len(dist)); w = .38
for i, (c, col) in enumerate((("legitimo", C_LE), ("phishing", C_PH))):
    b = ax.bar(x + (i - .5) * w, dist[c], w, color=col, label="Legítimo" if c == "legitimo" else "Phishing")
    ax.bar_label(b, fmt="{:,.0f}", fontsize=7, padding=2)
ax.set_xticks(x, dist.index); ax.set_yscale("log"); ax.set_ylabel("Número de URL (escala log)")
ax.legend(frameon=False); fig.tight_layout(); fig.savefig(f"{OUT}/eda_fig_distribucion_clases.png", dpi=300); plt.close()

# 2. Estadística descriptiva por clase (v2)
desc = v2.groupby("clase")[FEATURE_NAMES].agg(["mean", "median", "std", "min", "max"]).T.unstack(1).round(2)
desc.to_csv(f"{OUT}/eda_descriptiva_v2.csv")
med = v2.groupby("clase")[FEATURE_NAMES].median().T; mean = v2.groupby("clase")[FEATURE_NAMES].mean().T
res = pd.DataFrame({"media_legitimo": mean.legitimo, "media_phishing": mean.phishing,
                    "mediana_legitimo": med.legitimo, "mediana_phishing": med.phishing}).round(2)
res.to_csv(f"{OUT}/eda_resumen_por_clase_v2.csv"); print(res)

# 3. Boxplots de variables numéricas por clase (v2)
num = ["length_url", "length_domain", "count_dots", "count_subdomains", "count_slashes", "count_digits_url"]
esc = ["count_hyphens", "suspicious_words_count", "count_question_marks", "count_equal_signs", "count_underscores", "count_percent"]
pe = (100 * v2.groupby("clase")[esc].agg(lambda s: (s > 0).mean())).round(1).T
pe.to_csv(f"{OUT}/eda_dispersas_pct_mayor_0_v2.csv"); print(pe)
fig, ax = plt.subplots(figsize=(7, 3.2)); xx = np.arange(len(esc))
for i, (c, col) in enumerate((("legitimo", C_LE), ("phishing", C_PH))):
    b = ax.bar(xx + (i - .5) * .38, pe[c], .38, color=col, label="Legítimo" if c == "legitimo" else "Phishing")
    ax.bar_label(b, fmt="%.1f", fontsize=7)
ax.set_xticks(xx, esc, rotation=20, ha="right", fontsize=8); ax.set_ylabel("% de URL con valor ≥ 1")
ax.legend(frameon=False); fig.tight_layout(); fig.savefig(f"{OUT}/eda_fig_dispersas_v2.png", dpi=300); plt.close()
fig, axs = plt.subplots(2, 3, figsize=(9, 5.5)); axs = axs.ravel()
for ax, f in zip(axs, num):
    data = [v2.loc[v2.clase == "legitimo", f], v2.loc[v2.clase == "phishing", f]]
    bp = ax.boxplot(data, tick_labels=["Legítimo", "Phishing"], showfliers=False, patch_artist=True, widths=.55)
    for p, c in zip(bp["boxes"], (C_LE, C_PH)): p.set_facecolor(c); p.set_alpha(.55)
    for mline in bp["medians"]: mline.set_color("black")
    ax.set_title(f, fontsize=8.5)
for ax in axs[len(num):]: ax.axis("off")
fig.tight_layout(); fig.savefig(f"{OUT}/eda_fig_boxplots_v2.png", dpi=300); plt.close()

# 4. Variables binarias (% por clase) v1 vs v2
binf = [f for f in FEATURE_NAMES if f.startswith("has_")]
tb = pd.concat({n: (100 * d.groupby("clase")[binf].mean()).round(1) for n, d in (("v1", v1), ("v2", v2))})
tb.to_csv(f"{OUT}/eda_binarias.csv"); print(tb)
fig, axs = plt.subplots(1, 3, figsize=(9, 3));
for ax, f in zip(axs, binf):
    vals = tb[f].unstack(); xx = np.arange(2)
    for i, (c, col) in enumerate((("legitimo", C_LE), ("phishing", C_PH))):
        b = ax.bar(xx + (i - .5) * .38, vals[c], .38, color=col, label="Legítimo" if c == "legitimo" else "Phishing")
        ax.bar_label(b, fmt="%.1f", fontsize=7)
    ax.set_xticks(xx, ["v1", "v2"]); ax.set_title(f); ax.set_ylim(0, 112); ax.set_ylabel("% de URL")
axs[0].legend(frameon=False, fontsize=7); fig.tight_layout(); fig.savefig(f"{OUT}/eda_fig_binarias.png", dpi=300); plt.close()

# 5. Correlación de Spearman (v2) + correlación con la clase
Z = v2[FEATURE_NAMES].copy(); Z["clase_phishing"] = (v2.clase == "phishing").astype(int)
corr = Z.corr(method="spearman"); corr.round(3).to_csv(f"{OUT}/eda_correlacion_spearman_v2.csv")
fig, ax = plt.subplots(figsize=(8, 7)); im = ax.imshow(corr, cmap="RdBu_r", vmin=-1, vmax=1)
ax.set_xticks(range(len(corr)), corr.columns, rotation=90, fontsize=7); ax.set_yticks(range(len(corr)), corr.columns, fontsize=7)
for i in range(len(corr)):
    for j in range(len(corr)):
        v = corr.iloc[i, j]; ax.text(j, i, f"{v:.2f}", ha="center", va="center", fontsize=5,
                                     color="white" if abs(v) > .6 else "black")
fig.colorbar(im, fraction=.046); fig.tight_layout(); fig.savefig(f"{OUT}/eda_fig_correlacion_v2.png", dpi=300); plt.close()
fuertes = [(a, b, round(corr.loc[a, b], 3)) for i, a in enumerate(FEATURE_NAMES) for b in FEATURE_NAMES[i+1:] if abs(corr.loc[a, b]) >= .7]
print("Pares |rho|>=0.7:", fuertes)
print("Corr con clase:\n", corr["clase_phishing"].drop("clase_phishing").sort_values().round(3))

# 6. Valores faltantes / duplicados
print("Duplicados v2:", v2.url.duplicated().sum(), " faltantes:", v2[FEATURE_NAMES].isna().sum().sum())

print("Fuentes v2:\n", v2.groupby(["clase", "fuente"]).size())
