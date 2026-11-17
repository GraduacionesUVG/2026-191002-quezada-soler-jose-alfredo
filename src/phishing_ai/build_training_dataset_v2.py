"""
Construye data/dataset_entrenamiento_v2.csv. En la v1 el 98% de la clase
phishing venia de URLhaus (descargas de malware, no phishing de
credenciales) y el modelo tenia recall de 48.75% contra OpenPhish.

Phishing:
  - PhiUSIIL Phishing URL Dataset (UCI, label=0): fuente principal.
  - data/raw/dataset_phishing_gt.csv (46 filas).
  - dataset_phishing_global.csv, fuente "Manual/urlscan.io" (42 filas).
  - dataset_phishing_global.csv, fuente "URLhaus": maximo 15% de la clase.
  - OpenPhish queda fuera; se reserva para la validacion externa.

Legitimo:
  - Las 17,088 legitimas de dataset_entrenamiento.csv.
  - PhiUSIIL (label=1) solo si hiciera falta para llegar al 50/50.

Reglas: sin URLs duplicadas, maximo 5 URLs por hostname (sin "www.") en
cada clase y balance final ~50/50.

Uso:
    python build_training_dataset_v2.py
"""

import csv
import os
import random
import sys
import urllib.parse

import requests

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(BASE_DIR, "data")
RAW_DIR = os.path.join(DATA_DIR, "raw")
OUT_PATH = os.path.join(DATA_DIR, "dataset_entrenamiento_v2.csv")

V1_DATASET = os.path.join(DATA_DIR, "dataset_entrenamiento.csv")
PHISHING_GT_CSV = os.path.join(RAW_DIR, "dataset_phishing_gt.csv")
PHISHING_GLOBAL_CSV = os.path.join(RAW_DIR, "dataset_phishing_global.csv")

PHIUSIIL_URL = "https://archive.ics.uci.edu/static/public/967/phiusiil+phishing+url+dataset.zip"
PHIUSIIL_ZIP = os.path.join(RAW_DIR, "phiusiil_phishing_url_dataset.zip")
PHIUSIIL_CSV_NAME = "PhiUSIIL_Phishing_URL_Dataset.csv"

MAX_URLHAUS_SHARE = 0.15
MAX_POR_HOSTNAME = 5
RANDOM_STATE = 42

COLUMNS = ["url", "label", "fuente"]


def hostname_sin_www(url: str) -> str:
    host = (urllib.parse.urlparse(url).hostname or url).lower()
    return host[4:] if host.startswith("www.") else host


def descargar_phiusiil():
    if os.path.exists(PHIUSIIL_ZIP):
        print(f"[PhiUSIIL] usando cache local: {PHIUSIIL_ZIP}")
        return
    print("[PhiUSIIL] descargando dataset desde UCI...")
    os.makedirs(RAW_DIR, exist_ok=True)
    resp = requests.get(PHIUSIIL_URL, timeout=120)
    resp.raise_for_status()
    with open(PHIUSIIL_ZIP, "wb") as f:
        f.write(resp.content)
    print(f"[PhiUSIIL] guardado en cache: {PHIUSIIL_ZIP}")


def leer_phiusiil():
    """Devuelve (phishing_urls, legit_urls) del PhiUSIIL (label=0 -> phishing,
    label=1 -> legitimo; verificado empiricamente: label=1 son dominios
    populares de una sola URL raiz, label=0 incluye dominios sospechosos y
    hosting gratuito abusado como ipfs.io/docs.google.com)."""
    import io
    import zipfile

    descargar_phiusiil()
    phishing, legitimo = [], []
    with zipfile.ZipFile(PHIUSIIL_ZIP) as zf:
        nombre = next(n for n in zf.namelist() if n.endswith(".csv"))
        with zf.open(nombre) as f:
            reader = csv.DictReader(io.TextIOWrapper(f, encoding="utf-8"))
            for row in reader:
                url = (row.get("URL") or "").strip()
                label = (row.get("label") or "").strip()
                if not url:
                    continue
                if label == "0":
                    phishing.append(url)
                elif label == "1":
                    legitimo.append(url)
    print(f"[PhiUSIIL] {len(phishing)} phishing / {len(legitimo)} legitimo disponibles.")
    return phishing, legitimo


def leer_csv_filtrado(path, filtro_fuente=None):
    filas = []
    with open(path, newline="", encoding="utf-8") as f:
        for row in csv.DictReader(f):
            url = (row.get("url") or "").strip()
            fuente = (row.get("fuente") or "").strip()
            if not url:
                continue
            if filtro_fuente is not None and fuente != filtro_fuente:
                continue
            filas.append((url, fuente))
    return filas


def limitar_por_hostname(urls_con_fuente, max_por_host, rng):
    """urls_con_fuente: lista de (url, fuente). Baraja y se queda con hasta
    max_por_host URLs por hostname (sin 'www.')."""
    items = list(urls_con_fuente)
    rng.shuffle(items)
    conteo = {}
    out = []
    for url, fuente in items:
        host = hostname_sin_www(url)
        if conteo.get(host, 0) >= max_por_host:
            continue
        conteo[host] = conteo.get(host, 0) + 1
        out.append((url, fuente))
    return out


def deduplicar(urls_con_fuente):
    vistos = set()
    out = []
    for url, fuente in urls_con_fuente:
        if url in vistos:
            continue
        vistos.add(url)
        out.append((url, fuente))
    return out


def main():
    rng = random.Random(RANDOM_STATE)

    for p in (V1_DATASET, PHISHING_GT_CSV, PHISHING_GLOBAL_CSV):
        if not os.path.exists(p):
            print(f"ERROR: no se encontro {p}", file=sys.stderr)
            sys.exit(1)

    phiusiil_phishing, phiusiil_legit = leer_phiusiil()

    # --- clase phishing: base (GT + manual), sin URLhaus todavia ----------
    base_gt = [(u, f) for u, f in leer_csv_filtrado(PHISHING_GT_CSV)]
    base_manual = leer_csv_filtrado(PHISHING_GLOBAL_CSV, filtro_fuente="Manual/urlscan.io")
    base_no_urlhaus = deduplicar(base_gt + base_manual)
    base_no_urlhaus = limitar_por_hostname(base_no_urlhaus, MAX_POR_HOSTNAME, rng)
    print(f"\nBase phishing (GT + manual, sin URLhaus): {len(base_no_urlhaus)} URLs unicas")

    urlhaus_pool = leer_csv_filtrado(PHISHING_GLOBAL_CSV, filtro_fuente="URLhaus")
    print(f"URLhaus disponible: {len(urlhaus_pool)} URLs")

    # --- clase legitimo: se reutiliza la v1 tal cual -----------------------
    legit_v1 = []
    with open(V1_DATASET, newline="", encoding="utf-8") as f:
        for row in csv.DictReader(f):
            if (row.get("etiqueta") or "").strip().lower() == "legitimo":
                legit_v1.append((row["url"].strip(), (row.get("fuente") or "").strip()))
    legit_v1 = deduplicar(legit_v1)
    legit_v1 = limitar_por_hostname(legit_v1, MAX_POR_HOSTNAME, rng)
    print(f"Legitimo reutilizado de v1 (tras dedupe + limite/hostname): {len(legit_v1)} URLs")

    # --- objetivo de tamano: igualar phishing al total de legitimo ---------
    objetivo_total = len(legit_v1)
    objetivo_urlhaus = int(round(objetivo_total * MAX_URLHAUS_SHARE))
    objetivo_resto = objetivo_total - objetivo_urlhaus  # base_no_urlhaus + PhiUSIIL

    urlhaus_urls_ya = {u for u, _ in base_no_urlhaus}
    urlhaus_pool = [(u, f) for u, f in urlhaus_pool if u not in urlhaus_urls_ya]
    rng.shuffle(urlhaus_pool)
    urlhaus_sample = urlhaus_pool[:objetivo_urlhaus]
    print(f"URLhaus muestreado (tope {MAX_URLHAUS_SHARE:.0%} del total phishing): "
          f"{len(urlhaus_sample)} URLs")

    faltan_de_phiusiil = max(0, objetivo_resto - len(base_no_urlhaus))
    urls_ya_en_base = {u for u, _ in base_no_urlhaus} | {u for u, _ in urlhaus_sample}
    phiusiil_candidatos = [(u, "PhiUSIIL") for u in phiusiil_phishing if u not in urls_ya_en_base]
    rng.shuffle(phiusiil_candidatos)

    phishing_total = deduplicar(base_no_urlhaus + urlhaus_sample)
    # se agregan PhiUSIIL respetando el limite por hostname de forma incremental
    conteo_host = {}
    for u, _ in phishing_total:
        h = hostname_sin_www(u)
        conteo_host[h] = conteo_host.get(h, 0) + 1
    agregados = 0
    for u, f in phiusiil_candidatos:
        if len(phishing_total) >= objetivo_total:
            break
        h = hostname_sin_www(u)
        if conteo_host.get(h, 0) >= MAX_POR_HOSTNAME:
            continue
        conteo_host[h] = conteo_host.get(h, 0) + 1
        phishing_total.append((u, f))
        agregados += 1
    print(f"PhiUSIIL agregado al phishing: {agregados} URLs "
          f"(se necesitaban ~{faltan_de_phiusiil} para llegar al objetivo)")

    total_final = len(phishing_total) + len(legit_v1)
    print(f"\nDataset v2: {len(phishing_total)} phishing / {len(legit_v1)} legitimo "
          f"= {total_final} filas ({len(phishing_total) / total_final * 100:.1f}% phishing)")

    if len(legit_v1) > len(phishing_total):
        print("AVISO: el phishing quedo por debajo del objetivo (no habia suficiente "
              "PhiUSIIL sin usar tras los filtros); el balance no es exactamente 50/50, "
              "ver el numero real arriba.")

    print("\nComposicion final por fuente:")
    from collections import Counter
    c_phish = Counter(f for _, f in phishing_total)
    c_legit = Counter(f for _, f in legit_v1)
    print("  Phishing:")
    for f, n in c_phish.most_common():
        print(f"    {f:20s} {n:6d}  ({n / len(phishing_total) * 100:.1f}%)")
    print("  Legitimo:")
    for f, n in c_legit.most_common():
        print(f"    {f:20s} {n:6d}  ({n / len(legit_v1) * 100:.1f}%)")

    os.makedirs(DATA_DIR, exist_ok=True)
    with open(OUT_PATH, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=COLUMNS)
        w.writeheader()
        for u, fuente in phishing_total:
            w.writerow({"url": u, "label": "phishing", "fuente": fuente})
        for u, fuente in legit_v1:
            w.writerow({"url": u, "label": "legitimo", "fuente": fuente})

    print(f"\n>> Guardado en {OUT_PATH}")


if __name__ == "__main__":
    main()
