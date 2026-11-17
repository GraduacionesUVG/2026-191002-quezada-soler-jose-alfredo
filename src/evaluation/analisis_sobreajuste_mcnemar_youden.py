"""Análisis pedidos por la asesora: sobreajuste, McNemar todos vs todos (Holm),
Cochran Q, índice de Youden, curvas de aprendizaje. Reutiliza el código del proyecto."""
import sys, os, json, itertools
import os; sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import numpy as np, pandas as pd
import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
import comparar_modelos as cm
from sklearn.model_selection import GroupShuffleSplit, StratifiedGroupKFold, train_test_split, cross_validate, learning_curve
from sklearn.metrics import f1_score, roc_curve, confusion_matrix, roc_auc_score
from scipy.stats import chi2

OUT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "resultados", "analisis_extra"); os.makedirs(OUT, exist_ok=True)
urls, labels = cm.cargar_dataset_v2(cm.DATASET_V2)
y = np.array(labels); groups = np.array([cm.v1.hostname_sin_www(u) for u in urls])
X = np.array([cm.extract_features(u) for u in urls])
eu, ye = cm.cargar_externa(); Xe = np.array([cm.extract_features(u) for u in eu])
umb = {"random_forest": .40, "logistic_regression": .38, "hist_gradient_boosting": .31,
       "svm_rbf": .29, "knn": .41, "mlp": .46}
claves = list(cm.MODEL_LABELS)

# 1) Reproducir modelos finales y predicciones externas
X_tr, X_te, y_tr, y_te = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
preds, probas, rep = {}, {}, []
for k in claves:
    m = cm.construir_modelos()[k]; m.fit(X_tr, y_tr)
    p = m.predict_proba(Xe)[:, 1]; probas[k] = p
    preds[k] = (p >= umb[k]).astype(int)
    tn, fp, fn, tp = confusion_matrix(ye, preds[k]).ravel()
    rep.append(dict(modelo=k, tn=tn, fp=fp, fn=fn, tp=tp, f1=round(f1_score(ye, preds[k]), 4),
                    auc=round(roc_auc_score(ye, p), 4),
                    f1_train80=round(f1_score(y_tr, (m.predict_proba(X_tr)[:, 1] >= umb[k]).astype(int)), 4),
                    f1_test20=round(f1_score(y_te, (m.predict_proba(X_te)[:, 1] >= umb[k]).astype(int)), 4)))
    print(rep[-1], flush=True)
pd.DataFrame(rep).to_csv(f"{OUT}/reproduccion_externa.csv", index=False)

# 2) McNemar todos contra todos + Holm
filas = []
for a, b in itertools.combinations(claves, 2):
    bb, cc, p = cm.mcnemar_exacto(ye, preds[a], preds[b])
    filas.append(dict(modelo_a=a, modelo_b=b, a_bien_b_mal=bb, a_mal_b_bien=cc, p=p))
df = pd.DataFrame(filas).sort_values("p").reset_index(drop=True)
m_ = len(df); adj = []; run = 0
for i, p in enumerate(df.p):
    run = max(run, min(1, (m_ - i) * p)); adj.append(run)
df["p_holm"] = adj; df["significativo_0.05"] = df.p_holm < 0.05
df.round(4).to_csv(f"{OUT}/mcnemar_pares_holm.csv", index=False); print(df.round(4), flush=True)
# Cochran Q
C = np.column_stack([(preds[k] == ye).astype(int) for k in claves]); kq = C.shape[1]
T = C.sum(0); R = C.sum(1); N = C.sum()
Q = (kq - 1) * (kq * (T**2).sum() - N**2) / (kq * N - (R**2).sum())
pq = chi2.sf(Q, kq - 1); json.dump({"Q": float(Q), "gl": kq - 1, "p": float(pq)}, open(f"{OUT}/cochran_q.json", "w"))
print("Cochran Q", Q, pq, flush=True)

# 3) Youden vs F1 en el mismo split 85/15 agrupado
yo = []
gss = GroupShuffleSplit(n_splits=1, test_size=0.15, random_state=42)
itr, iva = next(gss.split(X, y, groups))
for k in claves:
    m = cm.construir_modelos()[k]; m.fit(X[itr], y[itr]); p = m.predict_proba(X[iva])[:, 1]
    fpr, tpr, thr = roc_curve(y[iva], p); j = tpr - fpr; i = int(np.argmax(j)); tj = float(thr[i])
    pe = (probas[k] >= tj).astype(int); tn, fp, fn, tp = confusion_matrix(ye, pe).ravel()
    yo.append(dict(modelo=k, umbral_f1=umb[k], umbral_youden=round(tj, 3), J=round(float(j[i]), 4),
                   f1_ext_umbral_f1=round(f1_score(ye, preds[k]), 4), f1_ext_youden=round(f1_score(ye, pe), 4),
                   recall_ext_youden=round(tp / (tp + fn), 4), fp_ext_youden=int(fp)))
    print(yo[-1], flush=True)
pd.DataFrame(yo).to_csv(f"{OUT}/youden.csv", index=False)

# 4) Sobreajuste: F1 entrenamiento vs validación (StratifiedGroupKFold 5)
cv = StratifiedGroupKFold(n_splits=5, shuffle=True, random_state=42); so = []
for k in claves:
    r = cross_validate(cm.construir_modelos()[k], X, y, cv=cv, groups=groups, scoring="f1",
                       return_train_score=True, n_jobs=-1)
    so.append(dict(modelo=k, f1_train=round(r["train_score"].mean(), 4), f1_val=round(r["test_score"].mean(), 4),
                   brecha=round(r["train_score"].mean() - r["test_score"].mean(), 4)))
    print(so[-1], flush=True)
pd.DataFrame(so).to_csv(f"{OUT}/sobreajuste.csv", index=False)

# 5) Curvas de aprendizaje HGB y RF
fig, axs = plt.subplots(1, 2, figsize=(10, 4), sharey=True)
for ax, k in zip(axs, ["hist_gradient_boosting", "random_forest"]):
    sz, tr, te = learning_curve(cm.construir_modelos()[k], X, y, groups=groups, cv=cv, scoring="f1",
                                train_sizes=np.linspace(0.1, 1.0, 6), n_jobs=-1, shuffle=True, random_state=42)
    for s, lab, c in ((tr, "Entrenamiento", "#2a78d6"), (te, "Validación", "#eb6834")):
        ax.plot(sz, s.mean(1), "o-", color=c, label=lab)
        ax.fill_between(sz, s.mean(1) - s.std(1), s.mean(1) + s.std(1), color=c, alpha=.15)
    ax.set_title(cm.MODEL_LABELS[k]); ax.set_xlabel("Número de URL de entrenamiento"); ax.grid(alpha=.3)
axs[0].set_ylabel("F1-score"); axs[0].legend(); fig.tight_layout(); fig.savefig(f"{OUT}/curvas_aprendizaje.png", dpi=300)
print("FIN", flush=True)
