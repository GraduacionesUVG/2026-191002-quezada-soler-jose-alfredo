"""
Construye evaluation/validacion_externa_v2.csv.

  - Phishing: feed actual de OpenPhish, excluyendo URLs y hostnames del
    entrenamiento v1 y v2, y las URLs ya usadas en validacion_externa.csv.
  - Legitimo: las mismas URLs .gt de validacion_externa.csv (v1).

Uso:
    cd evaluation
    python construir_validacion_externa_v2.py
    python evaluar_modelo_v2.py
"""

import csv
import os
import subprocess
import sys
import urllib.parse

import requests

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
PHISHING_AI_DIR = os.path.abspath(os.path.join(BASE_DIR, "..", "phishing_ai"))
DATA_DIR = os.path.join(PHISHING_AI_DIR, "data")

TRAIN_V1 = os.path.join(DATA_DIR, "dataset_entrenamiento.csv")
TRAIN_V2 = os.path.join(DATA_DIR, "dataset_entrenamiento_v2.csv")
EXTERNAL_V1 = os.path.join(BASE_DIR, "validacion_externa.csv")
OUT_PATH = os.path.join(BASE_DIR, "validacion_externa_v2.csv")

OPENPHISH_URL = "https://openphish.com/feed.txt"
HTTP_HEADERS = {"User-Agent": "Mozilla/5.0 (compatible; PhishingShieldThesisBot/1.0)"}
MAX_PHISHING = 300


def descargar_openphish_feed() -> str:
    """Descarga con curl: requests.get() se queda colgado al seguir la
    redireccion de openphish.com a raw.githubusercontent.com y su timeout
    no lo corta."""
    resultado = subprocess.run(
        ["curl", "-sL", "--max-time", "20", OPENPHISH_URL],
        capture_output=True, text=True, timeout=30, check=True,
    )
    return resultado.stdout


def hostname_sin_www(url: str) -> str:
    host = (urllib.parse.urlparse(url).hostname or "").lower()
    return host[4:] if host.startswith("www.") else host


def cargar_urls_y_hosts(path: str, col_url="url"):
    urls, hosts = set(), set()
    with open(path, newline="", encoding="utf-8") as f:
        for row in csv.DictReader(f):
            url = (row.get(col_url) or "").strip()
            if not url:
                continue
            urls.add(url)
            hosts.add(hostname_sin_www(url))
    return urls, hosts


def main():
    for p in (TRAIN_V1, TRAIN_V2, EXTERNAL_V1):
        if not os.path.exists(p):
            print(f"ERROR: no se encontro {p}", file=sys.stderr)
            sys.exit(1)

    urls_v1, hosts_v1 = cargar_urls_y_hosts(TRAIN_V1)
    urls_v2, hosts_v2 = cargar_urls_y_hosts(TRAIN_V2)
    urls_ext_v1, _ = cargar_urls_y_hosts(EXTERNAL_V1)

    urls_excluir = urls_v1 | urls_v2 | urls_ext_v1
    hosts_excluir = hosts_v1 | hosts_v2
    print(f"Excluyendo por URL: {len(urls_excluir)} (entrenamiento v1+v2 + validacion externa v1)")
    print(f"Excluyendo por hostname: {len(hosts_excluir)} (entrenamiento v1+v2)")

    print(f"\n[OpenPhish] descargando lote nuevo desde {OPENPHISH_URL} ...")
    texto = descargar_openphish_feed()
    lineas = [l.strip() for l in texto.splitlines() if l.strip()]
    print(f"[OpenPhish] {len(lineas)} URLs en el feed actual.")

    seleccion, vistos = [], set()
    excl_url = excl_host = 0
    for url in lineas:
        if len(seleccion) >= MAX_PHISHING:
            break
        if url in vistos:
            continue
        vistos.add(url)
        if url in urls_excluir:
            excl_url += 1
            continue
        if hostname_sin_www(url) in hosts_excluir:
            excl_host += 1
            continue
        seleccion.append(url)

    print(f"[OpenPhish] excluidas por URL: {excl_url}; por hostname: {excl_host}")
    print(f"[OpenPhish] {len(seleccion)} URLs nuevas de phishing para validacion externa v2.")

    fuente_phishing = "OpenPhish-live-v2"
    if not seleccion:
        # Si el feed no trae URLs nuevas respecto a v1, se reutilizan las de
        # phishing de v1: nunca estuvieron en ningun set de entrenamiento.
        print("\nAVISO: el feed de OpenPhish no trae URLs nuevas; se reutilizan las "
              "de phishing de validacion_externa.csv (v1).")
        with open(EXTERNAL_V1, newline="", encoding="utf-8") as f:
            for row in csv.DictReader(f):
                if (row.get("label") or "").strip().lower() == "phishing":
                    seleccion.append(row["url"].strip())
        fuente_phishing = "OpenPhish-live-v1-reused"
        print(f"[OpenPhish] {len(seleccion)} URLs de phishing reutilizadas de v1.")

    # --- legitimo: reutilizar tal cual las .gt de validacion_externa.csv (v1) ---
    legitimo = []
    with open(EXTERNAL_V1, newline="", encoding="utf-8") as f:
        for row in csv.DictReader(f):
            if (row.get("label") or "").strip().lower() == "legitimo":
                legitimo.append((row["url"].strip(), row.get("fuente", "GT-crawl-live")))
    print(f"[GT] {len(legitimo)} URLs legitimas reutilizadas de validacion_externa.csv (v1).")

    filas = (
        [{"url": u, "label": "phishing", "fuente": fuente_phishing} for u in seleccion]
        + [{"url": u, "label": "legitimo", "fuente": f} for u, f in legitimo]
    )
    with open(OUT_PATH, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=["url", "label", "fuente"])
        w.writeheader()
        w.writerows(filas)

    print(f"\n>> {len(filas)} filas guardadas en {OUT_PATH} "
          f"({len(seleccion)} phishing / {len(legitimo)} legitimo)")


if __name__ == "__main__":
    main()
