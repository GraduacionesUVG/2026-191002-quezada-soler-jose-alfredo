import sys, os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import numpy as np, pandas as pd
import evaluar_modelo as v1
from sklearn.model_selection import RepeatedStratifiedKFold, StratifiedKFold, cross_val_predict
from sklearn.metrics import roc_curve, f1_score, precision_score, recall_score
from sklearn.base import clone
urls, labels = v1.cargar_dataset(v1.SCENARIOS["gt"])
y = np.array(labels); X = np.array([v1.extract_features(u) for u in urls])
print(len(y), y.sum())
outer = RepeatedStratifiedKFold(n_splits=5, n_repeats=5, random_state=42)
res = []
for k, base in v1.construir_modelos().items():
    m = {"f1_05": [], "p_05": [], "r_05": [], "f1_y": [], "p_y": [], "r_y": [], "t": []}
    for tr, te in outer.split(X, y):
        inner = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
        oof = cross_val_predict(clone(base), X[tr], y[tr], cv=inner, method="predict_proba")[:, 1]
        fpr, tpr, thr = roc_curve(y[tr], oof); t = float(min(thr[np.argmax(tpr - fpr)], 1.0))
        mod = clone(base).fit(X[tr], y[tr]); p = mod.predict_proba(X[te])[:, 1]
        for suf, tt in (("05", .5), ("y", t)):
            pr = (p >= tt).astype(int)
            m["f1_" + suf].append(f1_score(y[te], pr, zero_division=0))
            m["p_" + suf].append(precision_score(y[te], pr, zero_division=0))
            m["r_" + suf].append(recall_score(y[te], pr, zero_division=0))
        m["t"].append(t)
    row = {"modelo": k, **{a + "_media": round(float(np.mean(b)), 4) for a, b in m.items()},
           **{a + "_std": round(float(np.std(b)), 4) for a, b in m.items()}}
    res.append(row); print(row)
out = os.path.join(os.path.dirname(os.path.abspath(__file__)), "resultados", "analisis_extra"); os.makedirs(out, exist_ok=True)
pd.DataFrame(res).to_csv(os.path.join(out, "youden_gt.csv"), index=False)
