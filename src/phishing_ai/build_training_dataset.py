"""
Construye data/dataset_entrenamiento.csv a partir de las fuentes en data/raw/.

  1. Phishing: todo data/raw/dataset_phishing_global.csv (16,943 URLs,
     incluye los 46 casos de Guatemala).
  2. Legitimo: data/raw/dataset_legitimo_gt.csv (254 URLs) como base.
  3. Para balancear, recorre el ranking Tranco, descarga la homepage de
     cada dominio y toma hasta 3 enlaces internos como URLs legitimas
     (fuente "Tranco+crawl"). Los dominios que fallan se descartan.

Se usan enlaces internos y no solo el dominio raiz porque con URLs raiz el
modelo aprendia a separar las clases por longitud de URL.

El CSV conserva el esquema url,fuente,fecha,detalle,coincide_gt,dominio,
etiqueta que espera model_trainer.py.

Uso:
    python build_training_dataset.py
"""

import csv
import io
import os
import re
import sys
import time
import zipfile
from concurrent.futures import ThreadPoolExecutor, as_completed
from urllib.parse import urljoin, urlparse

import requests

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
RAW_DIR = os.path.join(BASE_DIR, "data", "raw")
OUT_PATH = os.path.join(BASE_DIR, "data", "dataset_entrenamiento.csv")
TRANCO_CACHE = os.path.join(RAW_DIR, "tranco_top1m.zip")
TRANCO_URL = "https://tranco-list.eu/top-1m.csv.zip"

COLUMNS = ["url", "fuente", "fecha", "detalle", "coincide_gt", "dominio", "etiqueta"]

HTTP_HEADERS = {"User-Agent": "Mozilla/5.0 (compatible; PhishingShieldThesisBot/1.0)"}
STATIC_EXT = (".css", ".js", ".png", ".jpg", ".jpeg", ".gif", ".svg", ".ico",
              ".pdf", ".zip", ".woff", ".woff2", ".mp4", ".webp", ".json",
              ".xml", ".txt", ".rss")
MAX_LINKS_POR_DOMINIO = 3
CRAWL_WORKERS = 60
CRAWL_TIMEOUT = (2, 3)  # (connect, read) segundos por operacion de red individual
CRAWL_DEADLINE = 4  # segundos wall-clock maximos por dominio (ver crawl_dominio)
MAX_BYTES_POR_PAGINA = 60_000  # suficiente para <head>+nav; no necesitamos la pagina completa
MAX_DOMINIOS_A_INTENTAR = 40_000  # limite de seguridad (medido: ~42% de exito -> hacen falta ~1.75x dominios que URLs objetivo)


def read_csv(path):
    with open(path, newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def load_phishing():
    rows = read_csv(os.path.join(RAW_DIR, "dataset_phishing_global.csv"))
    seen, out = set(), []
    for r in rows:
        url = (r.get("url") or "").strip()
        if not url or url in seen:
            continue
        if (r.get("etiqueta") or "").strip().lower() != "phishing":
            continue
        seen.add(url)
        out.append(r)
    return out


def load_legitimo_base():
    rows = read_csv(os.path.join(RAW_DIR, "dataset_legitimo_gt.csv"))
    seen, out = set(), []
    for r in rows:
        url = (r.get("url") or "").strip()
        if not url or url in seen:
            continue
        seen.add(url)
        out.append(r)
    return out, seen


def download_tranco():
    if os.path.exists(TRANCO_CACHE):
        print(f"[Tranco] usando cache local: {TRANCO_CACHE}")
        return
    print("[Tranco] descargando top-1M...")
    resp = requests.get(TRANCO_URL, timeout=60)
    resp.raise_for_status()
    os.makedirs(RAW_DIR, exist_ok=True)
    with open(TRANCO_CACHE, "wb") as f:
        f.write(resp.content)
    print(f"[Tranco] guardado en cache: {TRANCO_CACHE}")


def leer_ranking_tranco(limite: int):
    download_tranco()
    ranking = []
    with zipfile.ZipFile(TRANCO_CACHE) as zf:
        nombre = zf.namelist()[0]
        with zf.open(nombre) as f:
            reader = csv.reader(io.TextIOWrapper(f, encoding="utf-8"))
            for rank, dominio in reader:
                ranking.append((rank, dominio.strip().lower()))
                if len(ranking) >= limite:
                    break
    return ranking


def extraer_enlaces_internos(base_url: str, html: str, max_links: int) -> list:
    dominio = urlparse(base_url).netloc
    hrefs = re.findall(r'href=["\']([^"\'#]+)', html, re.IGNORECASE)
    vistos, out = set(), []
    for href in hrefs:
        if href.startswith(("mailto:", "javascript:", "tel:")):
            continue
        full = urljoin(base_url, href)
        p = urlparse(full)
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
    """Visita la homepage real de un dominio y devuelve (homepage_url, [enlaces_internos]).

    Usa iter_content() con un limite de tiempo real (wall-clock) en vez de
    r.raw.read(N), porque un servidor que entrega datos "a goteo" (pocos
    bytes cada vez, cada uno dentro del timeout por-lectura) puede hacer que
    read(N) tarde mucho mas que el timeout configurado: requests solo limita
    el tiempo de cada operacion de red individual, no el tiempo total.
    """
    url = f"https://{dominio}/"
    limite = time.monotonic() + CRAWL_DEADLINE
    try:
        r = requests.get(url, timeout=CRAWL_TIMEOUT, headers=HTTP_HEADERS, stream=True)
        if r.status_code >= 400:
            r.close()
            return None, []
        partes, total = [], 0
        for trozo in r.iter_content(chunk_size=8192):
            if trozo:
                partes.append(trozo)
                total += len(trozo)
            if total >= MAX_BYTES_POR_PAGINA or time.monotonic() > limite:
                break
        r.close()
        crudo = b"".join(partes)
    except requests.RequestException:
        return None, []
    texto = crudo.decode(r.encoding or "utf-8", errors="ignore")
    enlaces = extraer_enlaces_internos(r.url, texto, MAX_LINKS_POR_DOMINIO)
    return r.url, enlaces


def extra_legitimos_de_tranco(target_count: int, urls_ya_usadas: set,
                               dominios_phishing: set):
    if target_count <= 0:
        return []

    candidatos = [
        (rank, dom) for rank, dom in leer_ranking_tranco(MAX_DOMINIOS_A_INTENTAR)
        if dom and dom not in dominios_phishing and f"https://{dom}/" not in urls_ya_usadas
    ]
    print(f"[Tranco+crawl] visitando hasta {len(candidatos)} dominios reales "
          f"(objetivo: ~{target_count} URLs legitimas con ruta real)...")

    extras = []
    intentados = exitosos = 0
    lote = CRAWL_WORKERS * 4  # enviar en lotes chicos: al llegar a la meta no
                               # dejamos miles de tareas ya encoladas de mas

    with ThreadPoolExecutor(max_workers=CRAWL_WORKERS) as pool:
        for i in range(0, len(candidatos), lote):
            if len(extras) >= target_count:
                break
            trozo = candidatos[i:i + lote]
            futuros = {pool.submit(crawl_dominio, dom): (rank, dom) for rank, dom in trozo}
            for fut in as_completed(futuros):
                rank, dominio = futuros[fut]
                intentados += 1
                try:
                    homepage, enlaces = fut.result()
                except Exception:
                    homepage, enlaces = None, []
                if homepage is None:
                    continue
                exitosos += 1

                if homepage not in urls_ya_usadas:
                    urls_ya_usadas.add(homepage)
                    extras.append({
                        "url": homepage, "fuente": "Tranco+crawl", "fecha": "",
                        "detalle": f"homepage real, rank global {rank}",
                        "coincide_gt": "False", "dominio": dominio, "etiqueta": "legitimo",
                    })
                for enlace in enlaces:
                    if enlace in urls_ya_usadas:
                        continue
                    urls_ya_usadas.add(enlace)
                    extras.append({
                        "url": enlace, "fuente": "Tranco+crawl", "fecha": "",
                        "detalle": f"enlace interno real, rank global {rank}",
                        "coincide_gt": "False", "dominio": dominio, "etiqueta": "legitimo",
                    })

            print(f"  ... {intentados} dominios intentados, {exitosos} respondieron, "
                  f"{len(extras)} URLs legitimas reales recolectadas", flush=True)

    print(f"[Tranco+crawl] {exitosos}/{intentados} dominios respondieron; "
          f"{len(extras)} URLs legitimas con ruta real obtenidas.")
    return extras


def write_csv(path, rows):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=COLUMNS)
        writer.writeheader()
        for r in rows:
            writer.writerow({c: r.get(c, "") for c in COLUMNS})


def main():
    print("Cargando fuentes crudas desde data/raw/ ...")
    phishing = load_phishing()
    legitimo, urls_legit = load_legitimo_base()
    dominios_phishing = {(r.get("dominio") or "").strip().lower() for r in phishing}

    print(f"  Phishing (global, deduplicado): {len(phishing)}")
    print(f"  Legitimo base (GT manual + Tranco .gt): {len(legitimo)}")

    target_extra = max(0, len(phishing) - len(legitimo))
    extras = extra_legitimos_de_tranco(target_extra, urls_legit, dominios_phishing)
    legitimo += extras

    total = len(phishing) + len(legitimo)
    print(f"\nDataset final: {total} filas "
          f"({len(phishing)} phishing / {len(legitimo)} legitimo, "
          f"{len(phishing) / total * 100:.1f}% phishing)")

    write_csv(OUT_PATH, phishing + legitimo)
    print(f"\n>> Guardado en {OUT_PATH}")
    print("   Ejecuta model_trainer.py para reentrenar con este dataset.")


if __name__ == "__main__":
    try:
        main()
    except requests.RequestException as e:
        print(f"ERROR de red al descargar Tranco: {e}", file=sys.stderr)
        sys.exit(1)
