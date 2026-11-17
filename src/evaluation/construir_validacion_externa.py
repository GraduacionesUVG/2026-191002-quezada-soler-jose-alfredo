"""
Construye evaluation/validacion_externa.csv, un conjunto de prueba con URLs
que no estan en el dataset de entrenamiento.

  - Phishing: hasta 300 URLs del feed actual de OpenPhish, excluyendo las
    que coinciden por URL u hostname con dataset_entrenamiento.csv.
  - Legitimo: URLs .gt con ruta interna (no homepages), en cantidad similar.

Limitacion: el lado legitimo solo se excluye por URL exacta, no por
hostname. Tranco trae 236 dominios .gt y casi todos ya estan en el
entrenamiento, asi que excluir por hostname dejaria 0 URLs. Por eso este
lado mide generalizacion a paginas nuevas de sitios .gt conocidos; la
generalizacion a hosts nuevos se mide en evaluar_modelo.py con
StratifiedGroupKFold.

Uso:
    cd evaluation
    python construir_validacion_externa.py
    python evaluar_modelo.py
"""

import csv
import io
import os
import re
import sys
import time
import urllib.parse
import zipfile
from concurrent.futures import ThreadPoolExecutor, as_completed

import requests

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
PHISHING_AI_DIR = os.path.abspath(os.path.join(BASE_DIR, "..", "phishing_ai"))
DATA_DIR = os.path.join(PHISHING_AI_DIR, "data")
TRAIN_CSV = os.path.join(DATA_DIR, "dataset_entrenamiento.csv")
TRANCO_CACHE = os.path.join(DATA_DIR, "raw", "tranco_top1m.zip")
OUT_PATH = os.path.join(BASE_DIR, "validacion_externa.csv")

OPENPHISH_URL = "https://openphish.com/feed.txt"
MAX_PHISHING = 300

HTTP_HEADERS = {"User-Agent": "Mozilla/5.0 (compatible; PhishingShieldThesisBot/1.0)"}
STATIC_EXT = (".css", ".js", ".png", ".jpg", ".jpeg", ".gif", ".svg", ".ico",
              ".pdf", ".zip", ".woff", ".woff2", ".mp4", ".webp", ".json",
              ".xml", ".txt", ".rss")
MAX_LINKS_POR_DOMINIO = 4
CRAWL_WORKERS = 20
CRAWL_TIMEOUT = (2, 3)
CRAWL_DEADLINE = 4
MAX_BYTES_POR_PAGINA = 60_000


def cargar_entrenamiento_urls_y_hosts():
    urls, hosts = set(), set()
    with open(TRAIN_CSV, newline="", encoding="utf-8") as f:
        for row in csv.DictReader(f):
            url = (row.get("url") or "").strip()
            if not url:
                continue
            urls.add(url)
            host = (urllib.parse.urlparse(url).hostname or "").lower()
            if host.startswith("www."):
                host = host[4:]
            hosts.add(host)
    return urls, hosts


def hostname_sin_www(url: str) -> str:
    host = (urllib.parse.urlparse(url).hostname or "").lower()
    return host[4:] if host.startswith("www.") else host


# --- lado phishing: feed actual de OpenPhish ------------------------------

def obtener_phishing_openphish(urls_entrenamiento: set, hosts_entrenamiento: set, n: int):
    print(f"[OpenPhish] descargando feed actual desde {OPENPHISH_URL} ...")
    resp = requests.get(OPENPHISH_URL, timeout=30, headers=HTTP_HEADERS)
    resp.raise_for_status()
    lineas = [l.strip() for l in resp.text.splitlines() if l.strip()]
    print(f"[OpenPhish] {len(lineas)} URLs en el feed actual.")

    seleccion, vistos = [], set()
    excluidas_url = excluidas_host = 0
    for url in lineas:
        if len(seleccion) >= n:
            break
        if url in vistos:
            continue
        vistos.add(url)
        if url in urls_entrenamiento:
            excluidas_url += 1
            continue
        if hostname_sin_www(url) in hosts_entrenamiento:
            excluidas_host += 1
            continue
        seleccion.append(url)

    print(f"[OpenPhish] excluidas por URL ya en entrenamiento: {excluidas_url}; "
          f"por hostname ya en entrenamiento: {excluidas_host}")
    print(f"[OpenPhish] {len(seleccion)} URLs nuevas seleccionadas para validacion externa.")
    return seleccion


# --- lado legitimo: paginas internas reales de dominios .gt conocidos ----

def leer_dominios_gt_tranco():
    with zipfile.ZipFile(TRANCO_CACHE) as zf:
        with zf.open(zf.namelist()[0]) as f:
            reader = csv.reader(io.TextIOWrapper(f, encoding="utf-8"))
            return [d.strip().lower() for _, d in reader if d.strip().lower().endswith(".gt")]


def extraer_enlaces_internos(base_url: str, html: str, max_links: int) -> list:
    dominio = urllib.parse.urlparse(base_url).netloc
    hrefs = re.findall(r'href=["\']([^"\'#]+)', html, re.IGNORECASE)
    vistos, out = set(), []
    for href in hrefs:
        if href.startswith(("mailto:", "javascript:", "tel:")):
            continue
        full = urllib.parse.urljoin(base_url, href)
        p = urllib.parse.urlparse(full)
        if p.netloc != dominio or p.scheme not in ("http", "https"):
            continue
        if p.path in ("", "/") or p.path.lower().endswith(STATIC_EXT):
            continue
        if full in vistos:
            continue
        vistos.add(full)
        out.append(full)
        if len(out) >= max_links:
            break
    return out


def crawl_dominio(dominio: str):
    url = f"https://{dominio}/"
    limite = time.monotonic() + CRAWL_DEADLINE
    try:
        r = requests.get(url, timeout=CRAWL_TIMEOUT, headers=HTTP_HEADERS, stream=True)
        if r.status_code >= 400:
            r.close()
            return []
        partes, total = [], 0
        for trozo in r.iter_content(chunk_size=8192):
            if trozo:
                partes.append(trozo)
                total += len(trozo)
            if total >= MAX_BYTES_POR_PAGINA or time.monotonic() > limite:
                break
        r.close()
        texto = b"".join(partes).decode(r.encoding or "utf-8", errors="ignore")
    except requests.RequestException:
        return []
    return extraer_enlaces_internos(r.url, texto, MAX_LINKS_POR_DOMINIO)


def obtener_legitimo_gt(urls_entrenamiento: set, n: int):
    dominios = leer_dominios_gt_tranco()
    print(f"[GT] {len(dominios)} dominios .gt en el ranking Tranco cacheado "
          f"(se visitan en vivo para sacar paginas internas nuevas)...")

    seleccion, vistos = [], set()
    excluidas_url = 0
    with ThreadPoolExecutor(max_workers=CRAWL_WORKERS) as pool:
        futuros = {pool.submit(crawl_dominio, d): d for d in dominios}
        for fut in as_completed(futuros):
            if len(seleccion) >= n:
                break
            try:
                enlaces = fut.result()
            except Exception:
                enlaces = []
            for url in enlaces:
                if url in vistos:
                    continue
                vistos.add(url)
                if url in urls_entrenamiento:
                    excluidas_url += 1
                    continue
                seleccion.append(url)
                if len(seleccion) >= n:
                    break

    print(f"[GT] excluidas por ser URL exacta ya usada en entrenamiento: {excluidas_url}")
    print(f"[GT] {len(seleccion)} URLs legitimas .gt (con ruta interna, nunca homepage) "
          f"seleccionadas.")
    return seleccion


def main():
    urls_entrenamiento, hosts_entrenamiento = cargar_entrenamiento_urls_y_hosts()
    print(f"Dataset de entrenamiento: {len(urls_entrenamiento)} URLs, "
          f"{len(hosts_entrenamiento)} hostnames distintos.\n")

    phishing = obtener_phishing_openphish(urls_entrenamiento, hosts_entrenamiento, MAX_PHISHING)
    print()
    legitimo = obtener_legitimo_gt(urls_entrenamiento, len(phishing))

    filas = (
        [{"url": u, "label": "phishing", "fuente": "OpenPhish-live"} for u in phishing]
        + [{"url": u, "label": "legitimo", "fuente": "GT-crawl-live"} for u in legitimo]
    )
    if not filas:
        print("\nERROR: no se pudo construir ninguna fila (revisa la conexion a internet).",
              file=sys.stderr)
        sys.exit(1)

    with open(OUT_PATH, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=["url", "label", "fuente"])
        w.writeheader()
        w.writerows(filas)

    print(f"\n>> {len(filas)} filas guardadas en {OUT_PATH} "
          f"({len(phishing)} phishing / {len(legitimo)} legitimo)")
    print("   Corre 'python evaluar_modelo.py' para evaluar el modelo de produccion ahi.")


if __name__ == "__main__":
    main()
