"""
Genera un reporte de resultados de entrenamiento (JSON + HTML autocontenido)
a partir de las metricas calculadas en model_trainer.py.

El HTML no depende de ningun CDN ni de conexion a internet: se abre haciendo
doble clic sobre el archivo. Las graficas se construyen con SVG/CSS inline.
"""

import json
import os
from datetime import datetime, timezone

# Colores de las graficas
COLOR_RF = "#2a78d6"       # Random Forest
COLOR_LR = "#eb6834"       # Logistic Regression
COLOR_GOOD = "#0ca30c"     # legitimo
COLOR_CRITICAL = "#d03b3b"  # phishing

SEQUENTIAL_RAMP = [
    "#cde2fb", "#9ec5f4", "#6da7ec", "#3987e5", "#256abf", "#184f95", "#0d366b",
]

METRIC_LABELS = [
    ("accuracy", "Accuracy"),
    ("precision", "Precision"),
    ("recall", "Recall"),
    ("f1", "F1-score"),
]


def _pct(x: float) -> str:
    return f"{x * 100:.1f}%"


def _shade_for(value: float, max_value: float) -> str:
    if max_value <= 0:
        idx = 0
    else:
        idx = min(len(SEQUENTIAL_RAMP) - 1,
                   int(round((value / max_value) * (len(SEQUENTIAL_RAMP) - 1))))
    return SEQUENTIAL_RAMP[idx]


def _text_on(hex_color: str) -> str:
    """Devuelve negro o blanco segun el brillo del fondo para mantener contraste."""
    hex_color = hex_color.lstrip("#")
    r, g, b = (int(hex_color[i:i + 2], 16) for i in (0, 2, 4))
    luminance = (0.299 * r + 0.587 * g + 0.114 * b) / 255
    return "#0b0b0b" if luminance > 0.55 else "#ffffff"


def _kpi_card(value: str, label: str) -> str:
    return f"""
    <div class="kpi-card">
      <div class="kpi-value">{value}</div>
      <div class="kpi-label">{label}</div>
    </div>"""


def _metrics_comparison_html(models: dict) -> str:
    rows = []
    for key, label in METRIC_LABELS:
        rf_val = models["random_forest"][key]
        lr_val = models["logistic_regression"][key]
        rows.append(f"""
        <div class="metric-row">
          <div class="metric-name">{label}</div>
          <div class="bar-group">
            <div class="bar-track">
              <div class="bar-fill" style="width:{rf_val * 100:.1f}%; background:{COLOR_RF};"
                   title="Random Forest - {label}: {_pct(rf_val)}"></div>
            </div>
            <span class="bar-value">{_pct(rf_val)}</span>
          </div>
          <div class="bar-group">
            <div class="bar-track">
              <div class="bar-fill" style="width:{lr_val * 100:.1f}%; background:{COLOR_LR};"
                   title="Logistic Regression - {label}: {_pct(lr_val)}"></div>
            </div>
            <span class="bar-value">{_pct(lr_val)}</span>
          </div>
        </div>""")
    return "\n".join(rows)


def _donut_svg(phishing: int, legitimo: int) -> str:
    total = phishing + legitimo
    circumference = 2 * 3.14159265 * 52
    phishing_frac = phishing / total if total else 0
    phishing_len = circumference * phishing_frac
    return f"""
    <svg width="150" height="150" viewBox="0 0 150 150" role="img"
         aria-label="Distribucion de clases: {phishing} phishing, {legitimo} legitimas">
      <circle cx="75" cy="75" r="52" fill="none" stroke="var(--gridline)" stroke-width="18"/>
      <circle cx="75" cy="75" r="52" fill="none" stroke="{COLOR_GOOD}" stroke-width="18"
              stroke-dasharray="{circumference:.2f}" stroke-dashoffset="0"
              transform="rotate(-90 75 75)">
        <title>Legitimas: {legitimo} ({_pct(1 - phishing_frac)})</title>
      </circle>
      <circle cx="75" cy="75" r="52" fill="none" stroke="{COLOR_CRITICAL}" stroke-width="18"
              stroke-dasharray="{phishing_len:.2f} {circumference:.2f}"
              stroke-dashoffset="0" transform="rotate(-90 75 75)"
              style="transform-origin:75px 75px;">
        <title>Phishing: {phishing} ({_pct(phishing_frac)})</title>
      </circle>
      <text x="75" y="70" text-anchor="middle" class="donut-total">{total}</text>
      <text x="75" y="88" text-anchor="middle" class="donut-sub">URLs</text>
    </svg>"""


def _feature_importance_html(feature_importances: list, top_n: int = 10) -> str:
    items = feature_importances[:top_n]
    max_importance = max((f["importance"] for f in items), default=1) or 1
    rows = []
    for f in items:
        width = (f["importance"] / max_importance) * 100
        rows.append(f"""
        <div class="fi-row">
          <span class="fi-label">{f['name']}</span>
          <div class="fi-track">
            <div class="fi-fill" style="width:{width:.1f}%;"
                 title="{f['name']}: {f['importance']:.4f}"></div>
          </div>
          <span class="fi-value">{f['importance']:.3f}</span>
        </div>""")
    return "\n".join(rows)


def _confusion_matrix_html(cm: list, title: str) -> str:
    labels = ["Legitimo", "Phishing"]
    flat_max = max(max(row) for row in cm)
    cells = ""
    for i, row in enumerate(cm):
        for j, val in enumerate(row):
            bg = _shade_for(val, flat_max)
            fg = _text_on(bg)
            cells += (f'<td style="background:{bg}; color:{fg};">'
                      f'<div class="cm-val">{val}</div></td>')
    return f"""
    <div class="cm-block">
      <h4 class="cm-title">{title}</h4>
      <table class="cm">
        <tr>
          <th></th>
          <th colspan="2" class="cm-axis">Prediccion</th>
        </tr>
        <tr>
          <th class="cm-axis-v" rowspan="2">Real</th>
          <th>{labels[0]}</th><th>{labels[1]}</th>
        </tr>
        <tr>
          <th class="cm-row-label">{labels[0]}</th>
          <td style="background:{_shade_for(cm[0][0], flat_max)}; color:{_text_on(_shade_for(cm[0][0], flat_max))};">{cm[0][0]}</td>
          <td style="background:{_shade_for(cm[0][1], flat_max)}; color:{_text_on(_shade_for(cm[0][1], flat_max))};">{cm[0][1]}</td>
        </tr>
        <tr>
          <th class="cm-row-label">{labels[1]}</th>
          <td style="background:{_shade_for(cm[1][0], flat_max)}; color:{_text_on(_shade_for(cm[1][0], flat_max))};">{cm[1][0]}</td>
          <td style="background:{_shade_for(cm[1][1], flat_max)}; color:{_text_on(_shade_for(cm[1][1], flat_max))};">{cm[1][1]}</td>
        </tr>
      </table>
    </div>"""


def _learning_curve_svg(model_key: str, color: str, lc: dict) -> str:
    """Curva de aprendizaje de un modelo: F1 de entrenamiento vs. validacion."""
    strat = lc["cv"]["stratified"][model_key]
    grouped = lc["cv"]["grouped"][model_key]

    w, h = 460, 270
    ml, mr, mt, mb = 52, 16, 14, 44
    pw, ph = w - ml - mr, h - mt - mb

    all_vals = (strat["train_mean"] + strat["valid_mean"] + grouped["valid_mean"])
    y_min = max(0.0, int((min(all_vals) - 0.01) * 50) / 50)   # multiplo de 0.02
    y_max = 1.0
    x_max = max(strat["train_sizes"])

    def px(v):
        return ml + v / x_max * pw

    def py(v):
        return mt + (1 - (v - y_min) / (y_max - y_min)) * ph

    parts = []
    # cuadricula + eje Y (marcas cada 2 puntos porcentuales; y_min es multiplo de 0.02)
    n_ticks = int(round((y_max - y_min) / 0.02))
    for i in range(n_ticks + 1):
        val = y_min + 0.02 * i
        y = py(val)
        parts.append(f'<line x1="{ml}" y1="{y:.1f}" x2="{w - mr}" y2="{y:.1f}" '
                     f'stroke="var(--gridline)" stroke-width="1"/>')
        parts.append(f'<text x="{ml - 8}" y="{y + 4:.1f}" text-anchor="end" class="axis-t">'
                     f'{val * 100:.0f}%</text>')
    # eje X
    for size in strat["train_sizes"]:
        parts.append(f'<text x="{px(size):.1f}" y="{h - mb + 16}" text-anchor="middle" '
                     f'class="axis-t">{size:,}</text>'.replace(",", "."))
    parts.append(f'<text x="{ml + pw / 2:.1f}" y="{h - 6}" text-anchor="middle" class="axis-t">'
                 f'URLs usadas para entrenar</text>')

    def series(sizes, values, stroke, dash, label):
        pts = " ".join(f"{px(s):.1f},{py(v):.1f}" for s, v in zip(sizes, values))
        dash_attr = f' stroke-dasharray="{dash}"' if dash else ""
        out = [f'<polyline points="{pts}" fill="none" stroke="{stroke}" '
               f'stroke-width="2.2"{dash_attr} stroke-linejoin="round"/>']
        for s, v in zip(sizes, values):
            out.append(f'<circle cx="{px(s):.1f}" cy="{py(v):.1f}" r="3.4" fill="{stroke}">'
                       f'<title>{label} - {s} URLs: F1 {v * 100:.2f}%</title></circle>')
        return "".join(out)

    parts.append(series(strat["train_sizes"], strat["train_mean"],
                        "var(--text-muted)", None, "Entrenamiento"))
    parts.append(series(grouped["train_sizes"], grouped["valid_mean"],
                        color, "6 4", "Validacion (por dominio)"))
    parts.append(series(strat["train_sizes"], strat["valid_mean"],
                        color, None, "Validacion (aleatoria)"))

    return f"""
    <svg viewBox="0 0 {w} {h}" width="100%" role="img"
         aria-label="Curva de aprendizaje: F1 de entrenamiento y validacion segun el tamano del entrenamiento">
      {''.join(parts)}
    </svg>"""


def _learning_curve_html(lc: dict) -> str:
    names = [("random_forest", "Random Forest", COLOR_RF),
             ("logistic_regression", "Logistic Regression", COLOR_LR)]
    charts = "".join(f"""
      <div class="card lc-card">
        <h4 class="cm-title">{label}</h4>
        {_learning_curve_svg(key, color, lc)}
        <div class="lc-legend">
          <span><span class="lc-line" style="background:var(--text-muted);"></span>Entrenamiento</span>
          <span><span class="lc-line" style="background:{color};"></span>Validacion (K-fold aleatorio)</span>
          <span><span class="lc-line lc-dash" style="color:{color};"></span>Validacion (por dominio)</span>
        </div>
      </div>""" for key, label, color in names)

    rows = ""
    for key, label, _ in names:
        for scheme, scheme_label in (("stratified", "K-fold aleatorio"),
                                      ("grouped", "Agrupado por dominio")):
            c = lc["cv"][scheme][key]
            tr, va, sd = c["train_mean"][-1], c["valid_mean"][-1], c["valid_std"][-1]
            rows += (f"<tr><td>{label}</td><td>{scheme_label}</td><td>{_pct(tr)}</td>"
                     f"<td>{_pct(va)} <span class=\"muted\">&plusmn;{sd * 100:.1f}</span></td>"
                     f"<td>{(tr - va) * 100:.1f} pts</td></tr>")

    return f"""
  <div class="section">
    <h2>Curvas de aprendizaje (control de sobreajuste)</h2>
    <p class="hint">F1 medio con {lc['folds']}-fold CV al entrenar con una fraccion creciente de los datos.
      Si la curva de validacion se acerca a la de entrenamiento el modelo generaliza; una brecha grande y
      persistente indica sobreajuste. Ademas del K-fold aleatorio se valida <strong>agrupando por dominio</strong>:
      muchas URLs comparten dominio, y un split aleatorio deja URLs hermanas a ambos lados, lo que puede
      sobreestimar el desempeno sobre sitios nuevos.</p>
    <div class="lc-grid">{charts}</div>
    <div class="card" style="margin-top:20px;">
      <h4 class="cm-title">Resumen con el 100% de los datos de entrenamiento (F1)</h4>
      <table class="lc-table">
        <thead><tr><th>Modelo</th><th>Validacion</th><th>F1 entrenamiento</th>
          <th>F1 validacion</th><th>Brecha</th></tr></thead>
        <tbody>{rows}</tbody>
      </table>
    </div>
  </div>"""


def render_html(report: dict) -> str:
    ds = report["dataset"]
    split = report["split"]
    rf = report["models"]["random_forest"]
    lr = report["models"]["logistic_regression"]

    sources_html = "".join(
        f'<span class="tag">{name}: <strong>{count}</strong></span>'
        for name, count in ds["sources"].items()
    )

    return f"""<!doctype html>
<html lang="es">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>Reporte de entrenamiento - PhishingShield</title>
<style>
  :root {{
    color-scheme: light;
    --surface-0: #f9f9f7;
    --surface-1: #fcfcfb;
    --surface-2: #ffffff;
    --text-primary: #0b0b0b;
    --text-secondary: #52514e;
    --text-muted: #898781;
    --gridline: #e1e0d9;
    --border: rgba(11,11,11,0.10);
  }}
  @media (prefers-color-scheme: dark) {{
    :root:not([data-theme="light"]) {{
      color-scheme: dark;
      --surface-0: #0d0d0d;
      --surface-1: #1a1a19;
      --surface-2: #232322;
      --text-primary: #ffffff;
      --text-secondary: #c3c2b7;
      --text-muted: #898781;
      --gridline: #2c2c2a;
      --border: rgba(255,255,255,0.10);
    }}
  }}
  * {{ box-sizing: border-box; }}
  body {{
    margin: 0;
    background: var(--surface-0);
    color: var(--text-primary);
    font-family: system-ui, -apple-system, "Segoe UI", sans-serif;
    padding: 32px 16px 64px;
  }}
  .wrap {{ max-width: 980px; margin: 0 auto; }}
  header {{ margin-bottom: 28px; }}
  header h1 {{ margin: 0 0 4px; font-size: 1.6rem; }}
  header p {{ margin: 0; color: var(--text-secondary); font-size: .9rem; }}
  .tag {{
    display: inline-block; background: var(--surface-2); border: 1px solid var(--border);
    border-radius: 999px; padding: 4px 12px; font-size: .78rem; color: var(--text-secondary);
    margin: 4px 6px 0 0;
  }}
  .kpi-grid {{
    display: grid; grid-template-columns: repeat(auto-fit, minmax(150px, 1fr));
    gap: 12px; margin: 20px 0 32px;
  }}
  .kpi-card {{
    background: var(--surface-2); border: 1px solid var(--border); border-radius: 12px;
    padding: 16px 18px;
  }}
  .kpi-value {{ font-size: 1.7rem; font-weight: 700; line-height: 1.1; }}
  .kpi-label {{ color: var(--text-secondary); font-size: .75rem; text-transform: uppercase;
                letter-spacing: .05em; margin-top: 4px; }}
  .section {{ margin: 32px 0; }}
  .section h2 {{ font-size: 1.1rem; margin: 0 0 4px; }}
  .section .hint {{ color: var(--text-muted); font-size: .82rem; margin: 0 0 16px; }}
  .card {{
    background: var(--surface-2); border: 1px solid var(--border); border-radius: 14px;
    padding: 22px 24px;
  }}
  .grid-2 {{ display: grid; grid-template-columns: 1.4fr 1fr; gap: 20px; }}
  @media (max-width: 720px) {{ .grid-2 {{ grid-template-columns: 1fr; }} }}
  .legend {{ display: flex; gap: 18px; font-size: .82rem; color: var(--text-secondary);
             margin-bottom: 18px; flex-wrap: wrap; }}
  .legend-swatch {{ width: 10px; height: 10px; border-radius: 3px; display: inline-block;
                     margin-right: 6px; vertical-align: middle; }}
  .metric-row {{ margin-bottom: 16px; }}
  .metric-name {{ font-size: .85rem; color: var(--text-secondary); margin-bottom: 6px; }}
  .bar-group {{ display: flex; align-items: center; gap: 10px; margin-bottom: 4px; }}
  .bar-track {{ flex: 1; background: var(--gridline); border-radius: 5px; height: 10px;
                overflow: hidden; }}
  .bar-fill {{ height: 100%; border-radius: 5px; transition: width .4s ease; }}
  .bar-value {{ font-size: .78rem; color: var(--text-secondary); min-width: 44px; text-align: right; }}
  .donut-wrap {{ display: flex; flex-direction: column; align-items: center; gap: 14px; }}
  .donut-total {{ font-size: 22px; font-weight: 700; fill: var(--text-primary); }}
  .donut-sub {{ font-size: 10px; fill: var(--text-muted); text-transform: uppercase; }}
  .donut-legend {{ font-size: .82rem; color: var(--text-secondary); }}
  .donut-legend div {{ display: flex; align-items: center; gap: 8px; margin-bottom: 6px; }}
  .fi-row {{ display: flex; align-items: center; gap: 12px; margin-bottom: 10px; }}
  .fi-label {{ font-size: .82rem; color: var(--text-secondary); min-width: 160px; }}
  .fi-track {{ flex: 1; background: var(--gridline); border-radius: 4px; height: 9px; overflow: hidden; }}
  .fi-fill {{ height: 100%; border-radius: 4px; background: {COLOR_RF}; }}
  .fi-value {{ font-size: .78rem; color: var(--text-muted); min-width: 42px; text-align: right; }}
  .cm-grid {{ display: flex; gap: 24px; flex-wrap: wrap; }}
  .cm-block {{ flex: 1; min-width: 220px; }}
  .cm-title {{ font-size: .88rem; margin: 0 0 10px; color: var(--text-secondary); }}
  table.cm {{ border-collapse: separate; border-spacing: 3px; font-size: .82rem; }}
  table.cm th {{ font-weight: 500; color: var(--text-muted); font-size: .72rem;
                 text-transform: uppercase; letter-spacing: .04em; padding: 4px 8px; }}
  table.cm td {{ border-radius: 8px; padding: 14px 18px; text-align: center; font-weight: 700; }}
  .cm-axis {{ text-align: center; }}
  .lc-grid {{ display: grid; grid-template-columns: 1fr 1fr; gap: 20px; }}
  @media (max-width: 720px) {{ .lc-grid {{ grid-template-columns: 1fr; }} }}
  .lc-card {{ padding: 18px 18px 14px; }}
  .axis-t {{ font-size: 10px; fill: var(--text-muted); }}
  .lc-legend {{ display: flex; flex-wrap: wrap; gap: 6px 16px; font-size: .76rem;
                color: var(--text-secondary); margin-top: 8px; }}
  .lc-line {{ display: inline-block; width: 18px; height: 3px; border-radius: 2px;
              margin-right: 6px; vertical-align: middle; }}
  .lc-dash {{ background: repeating-linear-gradient(90deg, currentColor 0 6px, transparent 6px 10px); }}
  table.lc-table {{ width: 100%; border-collapse: collapse; font-size: .84rem; }}
  table.lc-table th {{ text-align: left; font-weight: 500; color: var(--text-muted); font-size: .72rem;
                       text-transform: uppercase; letter-spacing: .04em; padding: 6px 10px;
                       border-bottom: 1px solid var(--gridline); }}
  table.lc-table td {{ padding: 8px 10px; border-bottom: 1px solid var(--gridline);
                       color: var(--text-secondary); }}
  .muted {{ color: var(--text-muted); font-size: .78rem; }}
  footer {{ margin-top: 40px; color: var(--text-muted); font-size: .78rem; text-align: center; }}
</style>
</head>
<body>
<div class="wrap">
  <header>
    <h1>Reporte de entrenamiento - PhishingShield</h1>
    <p>Generado el {report['generated_at']} &middot; dataset: {ds['path']}</p>
    <div>{sources_html}</div>
  </header>

  <div class="kpi-grid">
    {_kpi_card(str(ds['total']), 'URLs totales')}
    {_kpi_card(str(ds['phishing']), 'Phishing')}
    {_kpi_card(str(ds['legitimo']), 'Legitimas')}
    {_kpi_card(_pct(ds['phishing'] / ds['total']), '% Phishing')}
    {_kpi_card(f"{split['train']} / {split['test']}", 'Train / Test')}
  </div>

  <div class="section">
    <h2>Comparacion de metricas</h2>
    <p class="hint">Random Forest vs Logistic Regression sobre el conjunto de prueba ({split['test']} URLs, {_pct(split['test_size'])} del total).</p>
    <div class="card">
      <div class="legend">
        <span><span class="legend-swatch" style="background:{COLOR_RF};"></span>Random Forest</span>
        <span><span class="legend-swatch" style="background:{COLOR_LR};"></span>Logistic Regression</span>
      </div>
      {_metrics_comparison_html(report['models'])}
    </div>
  </div>

  <div class="section">
    <h2>Dataset y caracteristicas</h2>
    <p class="hint">Distribucion de clases usada para el entrenamiento e importancia de variables (Random Forest).</p>
    <div class="grid-2">
      <div class="card">
        <h4 class="cm-title">Importancia de caracteristicas (Top 10, RF)</h4>
        {_feature_importance_html(rf['feature_importances'])}
      </div>
      <div class="card donut-wrap">
        {_donut_svg(ds['phishing'], ds['legitimo'])}
        <div class="donut-legend">
          <div><span class="legend-swatch" style="background:{COLOR_CRITICAL};"></span>Phishing &middot; {ds['phishing']} ({_pct(ds['phishing'] / ds['total'])})</div>
          <div><span class="legend-swatch" style="background:{COLOR_GOOD};"></span>Legitimas &middot; {ds['legitimo']} ({_pct(ds['legitimo'] / ds['total'])})</div>
        </div>
      </div>
    </div>
  </div>

  <div class="section">
    <h2>Matrices de confusion</h2>
    <p class="hint">Sobre el conjunto de prueba ({split['test']} URLs no vistas durante el entrenamiento).</p>
    <div class="card">
      <div class="cm-grid">
        {_confusion_matrix_html(rf['confusion_matrix'], 'Random Forest')}
        {_confusion_matrix_html(lr['confusion_matrix'], 'Logistic Regression')}
      </div>
    </div>
  </div>

  {_learning_curve_html(report['learning_curve']) if report.get('learning_curve') else ''}

  <footer>Generado automaticamente por model_trainer.py &middot; no requiere conexion a internet.</footer>
</div>
</body>
</html>"""


def save_report(report: dict, out_dir: str, prefix: str = "training_report") -> tuple:
    os.makedirs(out_dir, exist_ok=True)
    json_path = os.path.join(out_dir, f"{prefix}.json")
    html_path = os.path.join(out_dir, f"{prefix}.html")

    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(report, f, ensure_ascii=False, indent=2)

    with open(html_path, "w", encoding="utf-8") as f:
        f.write(render_html(report))

    return json_path, html_path


def now_iso() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M UTC")
