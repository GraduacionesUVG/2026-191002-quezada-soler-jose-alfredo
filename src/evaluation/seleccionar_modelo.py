"""
Paso final de la comparacion (se ejecuta despues de comparar_modelos.py).
Levanta el microservicio con cada modelo de phishing_ai/modelos_comparacion/,
mide la latencia HTTP de POST /analyze igual que prueba_latencia.py, aplica
el criterio de criterio_seleccion.json y guarda la decision en
resultados/comparacion_seleccion.json.

Uso:
    cd evaluation
    python seleccionar_modelo.py
    python seleccionar_modelo.py --n 200
"""

import argparse
import csv
import json
import os
import statistics
import subprocess
import sys
import time

import requests

from prueba_latencia import cargar_urls_muestra

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
PHISHING_AI_DIR = os.path.abspath(os.path.join(BASE_DIR, "..", "phishing_ai"))
MODELS_DIR = os.path.join(PHISHING_AI_DIR, "modelos_comparacion")
RESULTS_DIR = os.path.join(BASE_DIR, "resultados")
CRITERIO_JSON = os.path.join(BASE_DIR, "criterio_seleccion.json")
UMBRALES_JSON = os.path.join(RESULTS_DIR, "comparacion_umbrales.json")
EXTERNA_CSV = os.path.join(RESULTS_DIR, "comparacion_metricas_validacion_externa.csv")
AGRUPADO_CSV = os.path.join(RESULTS_DIR, "comparacion_metricas_agrupado_hostname.csv")


def leer_csv(path):
    with open(path, newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def esperar_servicio(base: str, proc, timeout_s=60):
    t0 = time.time()
    while time.time() - t0 < timeout_s:
        if proc.poll() is not None:
            raise RuntimeError("uvicorn termino antes de estar listo")
        try:
            r = requests.get(f"{base}/health", timeout=2)
            if r.ok and r.json().get("model_loaded"):
                return r.json()
        except requests.RequestException:
            pass
        time.sleep(0.5)
    raise RuntimeError("el microservicio no respondio a tiempo")


def medir_latencia_http(clave, umbral, urls, port) -> dict:
    """Misma medicion que prueba_latencia.py, contra una instancia propia."""
    base = f"http://127.0.0.1:{port}"
    env = dict(os.environ,
               MODEL_PATH=os.path.join(MODELS_DIR, f"{clave}.pkl"),
               DECISION_THRESHOLD=str(umbral),
               PYTHONWARNINGS="ignore")
    proc = subprocess.Popen(
        [sys.executable, "-m", "uvicorn", "main:app", "--host", "127.0.0.1",
         "--port", str(port), "--log-level", "warning"],
        cwd=PHISHING_AI_DIR, env=env,
        stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    try:
        health = esperar_servicio(base, proc)
        assert os.path.basename(health["model_path"]) == f"{clave}.pkl", health
        sesion = requests.Session()
        lat = []
        for url in urls:
            t0 = time.perf_counter()
            resp = sesion.post(f"{base}/analyze", json={"url": url}, timeout=15)
            ms = (time.perf_counter() - t0) * 1000
            if resp.status_code == 200:
                lat.append(ms)
    finally:
        proc.terminate()
        try:
            proc.wait(timeout=10)
        except subprocess.TimeoutExpired:
            proc.kill()

    orden = sorted(lat)
    p95_idx = min(len(orden) - 1, int(round(0.95 * (len(orden) - 1))))
    return {"n_solicitudes": len(urls), "exitosas": len(lat),
            "fallidas": len(urls) - len(lat),
            "promedio_ms": round(statistics.mean(lat), 2),
            "mediana_ms": round(statistics.median(lat), 2),
            "p95_ms": round(orden[p95_idx], 2),
            "maximo_ms": round(max(lat), 2)}


def aplicar_criterio(criterio, candidatos) -> dict:
    max_ms = criterio["restriccion"]["maximo_ms"]
    margen = criterio["desempate"]["margen_f1"]
    elegibles = [c for c in candidatos if c["p95_ms"] < max_ms and c["fallidas"] == 0]
    if not elegibles:
        return {"ganador": None, "motivo": f"ningun modelo cumple p95 < {max_ms} ms"}
    mejor_f1 = max(c["f1_externa"] for c in elegibles)
    empatados = [c for c in elegibles if mejor_f1 - c["f1_externa"] <= margen + 1e-9]
    empatados.sort(key=lambda c: (-c["f1_agrupado"], c["p95_ms"]))
    ganador = empatados[0]
    if len(empatados) == 1:
        motivo = "mayor F1 en validacion externa entre los elegibles"
    else:
        motivo = (f"{len(empatados)} modelos a <= {margen} del mejor F1 externo "
                  f"({', '.join(c['clave'] for c in empatados)}); desempate por F1 "
                  "agrupado por hostname (y luego menor p95)")
    return {"ganador": ganador["clave"], "motivo": motivo,
            "elegibles": [c["clave"] for c in elegibles],
            "empatados_en_margen": [c["clave"] for c in empatados]}


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--n", type=int, default=100)
    ap.add_argument("--port", type=int, default=8765)
    args = ap.parse_args()

    with open(CRITERIO_JSON, encoding="utf-8") as f:
        criterio = json.load(f)
    with open(UMBRALES_JSON, encoding="utf-8") as f:
        umbrales = json.load(f)
    externa = {r["clave"]: r for r in leer_csv(EXTERNA_CSV) if r["umbral"] == "elegido"}
    agrupado = {r["clave"]: r for r in leer_csv(AGRUPADO_CSV)}

    urls = cargar_urls_muestra(args.n)
    candidatos, filas_lat = [], []
    for clave in criterio["candidatos"]:
        umbral = umbrales[clave]["umbral_optimo"]
        print(f"Midiendo latencia HTTP: {clave} (umbral {umbral}) ...")
        lat = medir_latencia_http(clave, umbral, urls, args.port)
        print(f"  mediana={lat['mediana_ms']} ms  p95={lat['p95_ms']} ms  "
              f"max={lat['maximo_ms']} ms  fallidas={lat['fallidas']}")
        filas_lat.append({"clave": clave, "umbral": umbral, **lat})
        candidatos.append({
            "clave": clave, "umbral": umbral,
            "f1_externa": float(externa[clave]["f1"]),
            "f1_agrupado": float(agrupado[clave]["f1_media"]),
            "p95_ms": lat["p95_ms"], "fallidas": lat["fallidas"],
        })

    os.makedirs(RESULTS_DIR, exist_ok=True)
    with open(os.path.join(RESULTS_DIR, "comparacion_latencia_http.csv"), "w",
              newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(filas_lat[0].keys()))
        w.writeheader()
        w.writerows(filas_lat)

    decision = aplicar_criterio(criterio, candidatos)
    salida = {"criterio": criterio, "candidatos": candidatos, "decision": decision}
    if decision["ganador"]:
        g = decision["ganador"]
        salida["configuracion_produccion"] = {
            "MODEL_PATH": f"modelos_comparacion/{g}.pkl",
            "DECISION_THRESHOLD": umbrales[g]["umbral_optimo"],
        }
    with open(os.path.join(RESULTS_DIR, "comparacion_seleccion.json"), "w",
              encoding="utf-8") as f:
        json.dump(salida, f, ensure_ascii=False, indent=2)

    print("\n=== Seleccion (criterio fijado en criterio_seleccion.json) ===")
    print(f"  {'modelo':24s} {'F1 ext':>7s} {'F1 host':>8s} {'p95 ms':>7s}")
    for c in sorted(candidatos, key=lambda c: -c["f1_externa"]):
        print(f"  {c['clave']:24s} {c['f1_externa']:7.4f} {c['f1_agrupado']:8.4f} {c['p95_ms']:7.1f}")
    print(f"\n  GANADOR: {decision['ganador']}  ({decision['motivo']})")
    if decision["ganador"]:
        print(f"  Produccion: {salida['configuracion_produccion']}")
    print(f"\n>> Guardado en {os.path.join(RESULTS_DIR, 'comparacion_seleccion.json')}")


if __name__ == "__main__":
    main()
