"""
Envia solicitudes POST a /analyze (microservicio phishing_ai) usando URLs
variadas tomadas de phishing_ai/data/dataset_entrenamiento.csv, mide la
latencia de cada una y reporta promedio, mediana, p95 y maximo en ms.

Requiere que el microservicio ya este corriendo:
    cd phishing_ai && uvicorn main:app --port 8000

Uso:
    cd evaluation
    python prueba_latencia.py                        # 100 solicitudes a localhost:8000
    python prueba_latencia.py --n 200
    python prueba_latencia.py --url http://localhost:8000
"""

import argparse
import csv
import os
import random
import statistics
import sys
import time

import requests

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATASET_PATH = os.path.join(BASE_DIR, "..", "phishing_ai", "data", "dataset_entrenamiento.csv")
RESULTS_DIR = os.path.join(BASE_DIR, "resultados")


def cargar_urls_muestra(n: int, seed: int = 42) -> list:
    urls = []
    with open(DATASET_PATH, newline="", encoding="utf-8") as f:
        for row in csv.DictReader(f):
            u = (row.get("url") or "").strip()
            if u:
                urls.append(u)
    random.Random(seed).shuffle(urls)
    if len(urls) < n:
        urls = urls * (n // len(urls) + 1)
    return urls[:n]


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                  formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--n", type=int, default=100, help="numero de solicitudes (default: 100)")
    ap.add_argument("--url", default="http://127.0.0.1:8000", help="URL base del microservicio")
    ap.add_argument("--etiqueta", default="",
                    help="sufijo para los archivos de salida (default: sin sufijo, "
                         "latencia_detalle.csv / latencia_resumen.csv)")
    args = ap.parse_args()

    base = args.url.rstrip("/")
    # Se usa 127.0.0.1 por defecto (no "localhost"): en varias configuraciones
    # de Windows la resolucion DNS de "localhost" agrega ~2s por conexion, lo
    # que arruina la medicion de latencia real del microservicio.
    sesion = requests.Session()

    try:
        r = sesion.get(f"{base}/health", timeout=5)
        r.raise_for_status()
        if not r.json().get("model_loaded"):
            print("ADVERTENCIA: el microservicio responde pero el modelo no esta cargado "
                  "(corre model_trainer.py primero).", file=sys.stderr)
    except requests.RequestException as e:
        print(f"ERROR: no se pudo conectar a {base}/health ({e}).", file=sys.stderr)
        print("Asegurate de que el microservicio este corriendo: "
              "cd phishing_ai && uvicorn main:app --port 8000", file=sys.stderr)
        sys.exit(1)

    urls = cargar_urls_muestra(args.n)
    print(f"Enviando {len(urls)} solicitudes a {base}/analyze ...")

    resultados = []
    for i, url in enumerate(urls, 1):
        t0 = time.perf_counter()
        try:
            resp = sesion.post(f"{base}/analyze", json={"url": url}, timeout=15)
            ok = resp.status_code == 200
            status = resp.status_code
        except requests.RequestException as e:
            ok = False
            status = f"error: {e}"
        latencia_ms = (time.perf_counter() - t0) * 1000
        resultados.append({"n": i, "url": url, "status": status, "ok": ok,
                            "latencia_ms": round(latencia_ms, 2)})
        if i % 20 == 0:
            print(f"  ... {i}/{len(urls)}")

    os.makedirs(RESULTS_DIR, exist_ok=True)
    sufijo = f"_{args.etiqueta}" if args.etiqueta else ""
    detalle_path = os.path.join(RESULTS_DIR, f"latencia_detalle{sufijo}.csv")
    with open(detalle_path, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=["n", "url", "status", "ok", "latencia_ms"])
        w.writeheader()
        w.writerows(resultados)

    latencias = [r["latencia_ms"] for r in resultados if r["ok"]]
    fallidas = len(resultados) - len(latencias)
    if not latencias:
        print(f"\nNinguna de las {len(resultados)} solicitudes tuvo exito; "
              f"revisa el detalle en {detalle_path}.", file=sys.stderr)
        sys.exit(1)

    ordenadas = sorted(latencias)
    p95_idx = min(len(ordenadas) - 1, int(round(0.95 * (len(ordenadas) - 1))))
    resumen = {
        "n_solicitudes": len(resultados),
        "exitosas": len(latencias),
        "fallidas": fallidas,
        "promedio_ms": round(statistics.mean(latencias), 2),
        "mediana_ms": round(statistics.median(latencias), 2),
        "p95_ms": round(ordenadas[p95_idx], 2),
        "maximo_ms": round(max(latencias), 2),
    }

    resumen_path = os.path.join(RESULTS_DIR, f"latencia_resumen{sufijo}.csv")
    with open(resumen_path, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(resumen.keys()))
        w.writeheader()
        w.writerow(resumen)

    print("\n=== Resumen de latencia ===")
    for k, v in resumen.items():
        print(f"  {k}: {v}")
    print(f"\n>> Detalle guardado en {detalle_path}")
    print(f">> Resumen guardado en {resumen_path}")


if __name__ == "__main__":
    main()
