"""
Recopila URLs de phishing desde tres feeds públicos de threat intelligence
(PhishTank, OpenPhish, URLhaus) y filtra los resultados que estén
relacionados con Guatemala: dominio .gt, bancos guatemaltecos y
entidades de gobierno.

Uso:
    pip install -r requirements.txt
    python recolector_phishing_gt.py

Salida:
    dataset_phishing_gt.csv     -> phishing que hizo match con Guatemala
    dataset_phishing_global.csv -> todo el phishing recolectado (sin filtrar)
    dataset_legitimo_gt.csv     -> URLs legítimas guatemaltecas (negativos)
    dataset_comparativo_gt.csv  -> phishing_gt + legitimo_gt con columna
                                   "etiqueta" (phishing/legitimo), lista para
                                   entrenar/evaluar el modelo de clasificación

PhishTank requiere una API key (PHISHTANK_API_KEY); si está vacía, esa
fuente se omite.
"""

import csv
import io
import re
import sys
import time
import zipfile
from dataclasses import dataclass, field
from urllib.parse import urlparse

import requests

# ---------------------------------------------------------------------------
# CONFIGURACIÓN
# ---------------------------------------------------------------------------

PHISHTANK_API_KEY = ""
PHISHTANK_URL = "https://data.phishtank.com/data/{key}/online-valid.csv"
OPENPHISH_URL = "https://openphish.com/feed.txt"
URLHAUS_URL = "https://urlhaus.abuse.ch/downloads/csv_recent/"
TRANCO_URL = "https://tranco-list.eu/top-1m.csv.zip"

TIMEOUT = 30  # segundos por request
TIMEOUT_TRANCO = 60  # el zip de Tranco pesa varios MB

# Palabras clave para detectar relación con Guatemala.
# Incluye el TLD .gt, entidades de gobierno y bancos del sistema.
KEYWORDS_GT = [
    # TLD y gobierno
    ".gob.gt", ".gt", "guatemala",
    # Bancos y entidades financieras guatemaltecas
    "banrural", "bantrab", "bi.com.gt", "bancoindustrial", "banco industrial",
    "bam.com.gt", "banco agromercantil", "agromercantil", "gtc.com.gt", "gyt continental",
    "g&t continental", "gytcontinental", "gtcontinental", "promerica", "ficohsa",
    "vivibanco", "banco de america central",
    "bac guatemala", "credomatic",
    # Entidades de gobierno / servicios comúnmente suplantados
    "sat.gob.gt", "sib.gob.gt", "mingob.gob.gt", "renap", "irtra",
    "muniguate", "min.finanzas", "minfin",
]

# Se va a usar un solo patrón único para que la búsqueda sea rápida
_PATTERN_GT = re.compile("|".join(re.escape(k) for k in KEYWORDS_GT), re.IGNORECASE)


@dataclass
class RegistroPhishing:
    url: str
    fuente: str
    fecha: str = ""
    detalle: str = ""
    coincide_gt: bool = False
    etiqueta: str = "phishing"  # "phishing" o "legitimo"


def es_relacionado_gt(url: str, extra_texto: str = "") -> bool:
    """Determina si una URL/registro parece estar relacionado con Guatemala."""
    texto = f"{url} {extra_texto}"
    return bool(_PATTERN_GT.search(texto))


# ---------------------------------------------------------------------------
# FUENTE 1: OpenPhish
# ---------------------------------------------------------------------------

def obtener_openphish() -> list[RegistroPhishing]:
    print("[OpenPhish] Descargando feed...")
    registros = []
    try:
        resp = requests.get(OPENPHISH_URL, timeout=TIMEOUT)
        resp.raise_for_status()
        lineas = [l.strip() for l in resp.text.splitlines() if l.strip()]
        for url in lineas:
            registros.append(RegistroPhishing(url=url, fuente="OpenPhish"))
        print(f"[OpenPhish] {len(registros)} URLs obtenidas.")
    except requests.RequestException as e:
        print(f"[OpenPhish] ERROR al descargar el feed: {e}", file=sys.stderr)
    return registros


# ---------------------------------------------------------------------------
# FUENTE 2: URLhaus
# ---------------------------------------------------------------------------

def obtener_urlhaus() -> list[RegistroPhishing]:
    print("[URLhaus] Descargando CSV reciente...")
    registros = []
    try:
        resp = requests.get(URLHAUS_URL, timeout=TIMEOUT)
        resp.raise_for_status()
        # El CSV de URLhaus trae varias líneas de comentario (#) antes del header real
        contenido = "\n".join(
            line for line in resp.text.splitlines() if not line.startswith("#")
        )
        lector = csv.reader(io.StringIO(contenido))
        for fila in lector:
            if len(fila) < 6:
                continue
            # columnas: id,dateadded,url,url_status,last_online,threat,tags,urlhaus_link,reporter
            fecha, url, estado, amenaza = fila[1], fila[2], fila[3], fila[5]
            registros.append(
                RegistroPhishing(url=url, fuente="URLhaus", fecha=fecha, detalle=amenaza)
            )
        print(f"[URLhaus] {len(registros)} URLs obtenidas.")
    except requests.RequestException as e:
        print(f"[URLhaus] ERROR al descargar el feed: {e}", file=sys.stderr)
    return registros


# ---------------------------------------------------------------------------
# FUENTE 3: PhishTank - requiere API key propia
# ---------------------------------------------------------------------------

def obtener_phishtank() -> list[RegistroPhishing]:
    registros = []
    if not PHISHTANK_API_KEY:
        print("[PhishTank] Sin API key configurada, se omite esta fuente. "
              "Regístrate en https://phishtank.org/register.php para habilitarla.")
        return registros

    print("[PhishTank] Descargando CSV...")
    try:
        url = PHISHTANK_URL.format(key=PHISHTANK_API_KEY)
        resp = requests.get(url, timeout=TIMEOUT, headers={"User-Agent": "phishtank/recolector-gt"})
        resp.raise_for_status()
        lector = csv.DictReader(io.StringIO(resp.text))
        for fila in lector:
            registros.append(
                RegistroPhishing(
                    url=fila.get("url", ""),
                    fuente="PhishTank",
                    fecha=fila.get("submission_time", ""),
                    detalle=fila.get("target", ""),
                )
            )
        print(f"[PhishTank] {len(registros)} URLs obtenidas.")
    except requests.RequestException as e:
        print(f"[PhishTank] ERROR al descargar el feed: {e}", file=sys.stderr)
    return registros


# ---------------------------------------------------------------------------
# Candidatos agregados a mano. No están verificados como phishing: se
# clasifican manualmente antes de entrar al dataset de entrenamiento.
# ---------------------------------------------------------------------------

CANDIDATOS_MANUALES: list[RegistroPhishing] = [
    # Encontrados en urlscan.io con búsquedas domain:<marca>* (28-jul-2026).
    # Algunos ya no responden; se conservan como ejemplos históricos.

    # --- Banrural ---
    RegistroPhishing(url="https://banruralvirtualgt.biz.site/", fuente="Manual/urlscan.io",
                      fecha="2026-07-25", detalle="typosquat banca virtual falsa"),
    RegistroPhishing(url="https://banruralbancavirtualgt.biz.site/", fuente="Manual/urlscan.io",
                      fecha="2026-07-25", detalle="typosquat banca virtual falsa"),
    RegistroPhishing(url="https://banrural-en-linea.biz.site/", fuente="Manual/urlscan.io",
                      fecha="2026-07-25", detalle="typosquat banca en linea falsa"),
    RegistroPhishing(url="https://banruralgtaccess.netlify.app/", fuente="Manual/urlscan.io",
                      fecha="2026-07-13", detalle="typosquat hosteado en netlify"),
    RegistroPhishing(url="http://banca-virtual.banrural.gt.zya.me/", fuente="Manual/urlscan.io",
                      fecha="2025-12-08", detalle="subdominio falso simulando banrural.gt"),
    RegistroPhishing(url="http://bancavirtual.banrural.gt.zya.me/", fuente="Manual/urlscan.io",
                      fecha="2025-10-30", detalle="subdominio falso simulando banrural.gt"),
    RegistroPhishing(url="http://banrural.gt.iceiy.com/", fuente="Manual/urlscan.io",
                      fecha="2023-09-11", detalle="subdominio falso simulando banrural.gt"),
    RegistroPhishing(url="https://banrural.gtbanrural.repl.co/", fuente="Manual/urlscan.io",
                      fecha="2023-09-07", detalle="typosquat hosteado en repl.co"),
    RegistroPhishing(url="https://www.banrural.gt-cb.login-cons.pages.ccntinue.vvwvvbanrural.com",
                      fuente="Manual/urlscan.io", fecha="2022-09-01",
                      detalle="dominio largo tipico de phishing de credenciales"),

    # --- Bantrab ---
    RegistroPhishing(url="https://bantrabenlineagtx.xo.je/", fuente="Manual/urlscan.io",
                      fecha="2026-07-28", detalle="typosquat banca en linea falsa (muy reciente)"),
    RegistroPhishing(url="http://www.bantrab00w.webcindario.com/", fuente="Manual/urlscan.io",
                      fecha="2026-07-15", detalle="typosquat hosteado en webcindario"),
    RegistroPhishing(url="https://actualizatecon.bantrab-gt.sbs/", fuente="Manual/urlscan.io",
                      fecha="2026-06-30", detalle="dominio .sbs, pretexto actualizacion de datos"),
    RegistroPhishing(url="https://bantrabapklngreso-gt.cyou/", fuente="Manual/urlscan.io",
                      fecha="2026-06-30", detalle="typosquat app/ingreso falso"),
    RegistroPhishing(url="https://bantrabactualizacionvirtualgt.com/", fuente="Manual/urlscan.io",
                      fecha="2026-05-28", detalle="pretexto actualizacion virtual de datos"),
    RegistroPhishing(url="http://bantrabajadoresdatosgt.com/", fuente="Manual/urlscan.io",
                      fecha="2026-05-28", detalle="pretexto actualizacion de datos de trabajadores"),
    RegistroPhishing(url="http://www.bantrabdatosgt.com/", fuente="Manual/urlscan.io",
                      fecha="2026-05-26", detalle="pretexto actualizacion de datos"),
    RegistroPhishing(url="http://bantrab-4e441.web.app/", fuente="Manual/urlscan.io",
                      fecha="2026-05-26", detalle="typosquat hosteado en firebase web.app"),
    RegistroPhishing(url="http://bantrabl-lngres0.lat/", fuente="Manual/urlscan.io",
                      fecha="2026-05-22", detalle="typosquat con caracteres homoglifo (l por i)"),
    RegistroPhishing(url="https://bantrab-gtww.pages.dev/", fuente="Manual/urlscan.io",
                      fecha="2026-03-20", detalle="typosquat hosteado en cloudflare pages"),
    RegistroPhishing(url="https://bantrabenlinea0.webcindario.com/", fuente="Manual/urlscan.io",
                      fecha="2026-03-20", detalle="tag 'phishing-thesis' en urlscan: otra tesis lo marco"),
    RegistroPhishing(url="http://enlineabantrab.sbs/", fuente="Manual/urlscan.io",
                      fecha="2026-03-09", detalle="typosquat banca en linea falsa"),
    RegistroPhishing(url="https://www.bantrabregalo.cc/gt", fuente="Manual/urlscan.io",
                      fecha="2026-02-08", detalle="pretexto de regalo/promocion falsa"),

    # --- G&T Continental ---
    RegistroPhishing(url="http://gtcontinentalbank.wixsite.com/", fuente="Manual/urlscan.io",
                      fecha="2026-06-29", detalle="typosquat hosteado en wixsite"),
    RegistroPhishing(url="https://gtcontinentaldesbloque.gtcotinental.repl.co/",
                      fuente="Manual/urlscan.io", fecha="2023-01-11",
                      detalle="pretexto desbloqueo de cuenta"),
    RegistroPhishing(url="http://gytcontinentalonline922.c1.biz/DEFINITIVO/",
                      fuente="Manual/urlscan.io", fecha="2023-06-23",
                      detalle="tag @phish_report en urlscan"),
    RegistroPhishing(url="http://gytcontinentalhechodeti.com/", fuente="Manual/urlscan.io",
                      fecha="2023-08-08", detalle="typosquat dominio propio"),
    RegistroPhishing(url="https://bancogytcontinental.gytcontinental9.repl.co/",
                      fuente="Manual/urlscan.io", fecha="2023-01-12",
                      detalle="typosquat hosteado en repl.co"),

    # --- Banco Industrial ---
    RegistroPhishing(url="https://bancoindustrial-gt.com/login_up.php", fuente="Manual/urlscan.io",
                      fecha="2025-12-11", detalle="typosquat con formulario de login"),
    RegistroPhishing(url="https://bancoindustrial-gt.com/login.php", fuente="Manual/urlscan.io",
                      fecha="2025-12-11", detalle="typosquat con formulario de login"),
    RegistroPhishing(url="https://bancoindustrialdeguatemala1698250274.zendesk.com/",
                      fuente="Manual/urlscan.io", fecha="2026-02-05",
                      detalle="pagina de soporte falsa abusando de zendesk"),

    # --- Vivibanco ---
    RegistroPhishing(url="https://vivibanco-gt.com/", fuente="Manual/urlscan.io",
                      fecha="2024-05-03", detalle="typosquat dominio propio, con subdominios cpanel/webmail"),
    RegistroPhishing(url="https://vivibanco-ebanca-gt.servlc0s.site/login.php",
                      fuente="Manual/urlscan.io", fecha="2024-08-10",
                      detalle="tag @phish_report en urlscan, formulario de login"),

    # --- BAM / Banco Agromercantil ---
    RegistroPhishing(url="https://bam.agromercantil.top/gtm/login.html", fuente="Manual/urlscan.io",
                      fecha="2025-09-29", detalle="subdominio falso con ruta /gtm/login"),

    # --- Credomatic / BAC (regional, incluye variante GT) ---
    RegistroPhishing(url="https://credomatic-gt.kinposgo.com/Login", fuente="Manual/urlscan.io",
                      fecha="2025-09-16", detalle="variante especifica GT de kit de phishing regional"),
    RegistroPhishing(url="https://bac.credomatic.cc/", fuente="Manual/urlscan.io",
                      fecha="2026-04-29", detalle="tag 'phishing-thesis' en urlscan"),

    # --- RENAP ---
    RegistroPhishing(url="https://renapgobgt.com/", fuente="Manual/urlscan.io",
                      fecha="2026-07-22", detalle="typosquat de renap.gob.gt (muy reciente)"),

    # --- Municipalidad de Guatemala (muniguate) ---
    RegistroPhishing(url="http://especiales.muniguate.cfd/", fuente="Manual/urlscan.io",
                      fecha="2026-02-16", detalle="typosquat con ruta /gt de pago falso"),
    RegistroPhishing(url="https://especiales.muniguate.one/gt", fuente="Manual/urlscan.io",
                      fecha="2026-02-14", detalle="typosquat con ruta /gt de pago falso"),
    RegistroPhishing(url="https://especiales.muniguate.sbs/gt/welcome.html",
                      fuente="Manual/urlscan.io", fecha="2026-02-13",
                      detalle="tag @phish_report en urlscan"),
    RegistroPhishing(url="https://especiales.muniguate.icu/gt/card.html",
                      fuente="Manual/urlscan.io", fecha="2026-02-09",
                      detalle="tag @phish_report en urlscan, pide datos de tarjeta"),

    # --- Ficohsa (banco regional CA, opera tambien en GT) ---
    RegistroPhishing(url="https://inicio-ficohsa.webcindario.com/", fuente="Manual/urlscan.io",
                      fecha="2026-04-19", detalle="tag 'phishing-thesis' en urlscan"),
    RegistroPhishing(url="https://ficohsahninter.webcindario.com/", fuente="Manual/urlscan.io",
                      fecha="2026-03-21", detalle="tag 'phishing-thesis' en urlscan"),
]


# ---------------------------------------------------------------------------
# FUENTE 4: URLs legítimas (negativos) - dominios oficiales verificados a mano.
# Sitio oficial de cada marca suplantada arriba, para tener positivos y
# negativos de las mismas marcas.
# ---------------------------------------------------------------------------

LEGITIMOS_MANUALES: list[RegistroPhishing] = [
    RegistroPhishing(url="https://www.banrural.com.gt/", fuente="Manual/oficial",
                      detalle="sitio oficial Banrural", etiqueta="legitimo"),
    RegistroPhishing(url="https://www.bantrab.com.gt/", fuente="Manual/oficial",
                      detalle="sitio oficial Bantrab", etiqueta="legitimo"),
    RegistroPhishing(url="https://www.gytcontinental.com.gt/", fuente="Manual/oficial",
                      detalle="sitio oficial G&T Continental", etiqueta="legitimo"),
    RegistroPhishing(url="https://www.bi.com.gt/", fuente="Manual/oficial",
                      detalle="sitio oficial Banco Industrial", etiqueta="legitimo"),
    RegistroPhishing(url="https://www.bam.com.gt/", fuente="Manual/oficial",
                      detalle="sitio oficial Banco Agromercantil (BAM)", etiqueta="legitimo"),
    RegistroPhishing(url="https://www.vivibanco.com.gt/", fuente="Manual/oficial",
                      detalle="sitio oficial Vivibanco", etiqueta="legitimo"),
    RegistroPhishing(url="https://www.ficohsa.com/gt/", fuente="Manual/oficial",
                      detalle="sitio oficial Ficohsa Guatemala", etiqueta="legitimo"),
    RegistroPhishing(url="https://www.promerica.com.gt/", fuente="Manual/oficial",
                      detalle="sitio oficial Promerica Guatemala", etiqueta="legitimo"),
    RegistroPhishing(url="https://www.credomatic.com/", fuente="Manual/oficial",
                      detalle="sitio oficial BAC Credomatic", etiqueta="legitimo"),
    RegistroPhishing(url="https://portal.sat.gob.gt/", fuente="Manual/oficial",
                      detalle="sitio oficial SAT Guatemala", etiqueta="legitimo"),
    RegistroPhishing(url="https://www.renap.gob.gt/", fuente="Manual/oficial",
                      detalle="sitio oficial RENAP", etiqueta="legitimo"),
    RegistroPhishing(url="https://www.muniguate.com/", fuente="Manual/oficial",
                      detalle="sitio oficial Municipalidad de Guatemala", etiqueta="legitimo"),
    RegistroPhishing(url="https://www.mingob.gob.gt/", fuente="Manual/oficial",
                      detalle="sitio oficial Ministerio de Gobernación", etiqueta="legitimo"),
    RegistroPhishing(url="https://www.minfin.gob.gt/", fuente="Manual/oficial",
                      detalle="sitio oficial Ministerio de Finanzas Públicas", etiqueta="legitimo"),
    RegistroPhishing(url="https://www.sib.gob.gt/", fuente="Manual/oficial",
                      detalle="sitio oficial Superintendencia de Bancos", etiqueta="legitimo"),
    RegistroPhishing(url="https://www.irtra.org.gt/", fuente="Manual/oficial",
                      detalle="sitio oficial IRTRA", etiqueta="legitimo"),
]


def obtener_urls_legitimas_tranco() -> list[RegistroPhishing]:
    """Descarga el ranking Tranco (top 1M dominios globales, usado en
    investigación académica de seguridad) y se queda solo con los dominios
    .gt: son sitios legítimos con tráfico real, útiles como negativos
    adicionales más allá de la lista curada a mano."""
    print("[Tranco] Descargando ranking y filtrando dominios .gt...")
    registros = []
    try:
        resp = requests.get(TRANCO_URL, timeout=TIMEOUT_TRANCO)
        resp.raise_for_status()
        with zipfile.ZipFile(io.BytesIO(resp.content)) as zf:
            nombre_csv = zf.namelist()[0]
            with zf.open(nombre_csv) as f:
                contenido = io.TextIOWrapper(f, encoding="utf-8")
                lector = csv.reader(contenido)
                for fila in lector:
                    if len(fila) < 2:
                        continue
                    _rank, dominio = fila[0], fila[1].strip().lower()
                    if dominio.endswith(".gt"):
                        registros.append(
                            RegistroPhishing(
                                url=f"https://{dominio}/",
                                fuente="Tranco",
                                detalle=f"rank global {_rank}",
                                etiqueta="legitimo",
                            )
                        )
        print(f"[Tranco] {len(registros)} dominios .gt obtenidos.")
    except (requests.RequestException, zipfile.BadZipFile) as e:
        print(f"[Tranco] ERROR al descargar/leer el ranking: {e}", file=sys.stderr)
    return registros


# ---------------------------------------------------------------------------
# PROCESO PRINCIPAL
# ---------------------------------------------------------------------------

def main():
    todos: list[RegistroPhishing] = []

    todos += obtener_openphish()
    time.sleep(1)
    todos += obtener_urlhaus()
    time.sleep(1)
    todos += obtener_phishtank()
    todos += CANDIDATOS_MANUALES

    print(f"\nTotal de registros de phishing recolectados: {len(todos)}")

    # Marcar coincidencias con Guatemala
    for r in todos:
        r.coincide_gt = es_relacionado_gt(r.url, r.detalle)

    relacionados_gt = [r for r in todos if r.coincide_gt]
    print(f"Registros de phishing relacionados con Guatemala: {len(relacionados_gt)}")

    # --- Negativos: URLs legítimas para poder hacer la comparativa ---
    legitimos: list[RegistroPhishing] = []
    legitimos += LEGITIMOS_MANUALES
    legitimos += obtener_urls_legitimas_tranco()
    for r in legitimos:
        r.coincide_gt = es_relacionado_gt(r.url, r.detalle) or r.url.rstrip("/").endswith(".gt")

    print(f"Registros legítimos (negativos) recolectados: {len(legitimos)}")

    # Guardar dataset completo de phishing (útil como negativos/contexto)
    guardar_csv("dataset_phishing_global.csv", todos)

    # Guardar solo el subconjunto de phishing de Guatemala
    guardar_csv("dataset_phishing_gt.csv", relacionados_gt)

    # Guardar los legítimos
    guardar_csv("dataset_legitimo_gt.csv", legitimos)

    # Guardar el comparativo phishing_gt + legitimo_gt, con columna "etiqueta",
    # listo para entrenar/evaluar un modelo de clasificación
    guardar_csv("dataset_comparativo_gt.csv", relacionados_gt + legitimos)

    print("\nListo. Archivos generados:")
    print("  - dataset_phishing_global.csv")
    print("  - dataset_phishing_gt.csv    (phishing relacionado con Guatemala)")
    print("  - dataset_legitimo_gt.csv    (negativos: URLs legítimas GT)")
    print("  - dataset_comparativo_gt.csv (phishing + legítimo, columna 'etiqueta')")
    print("\nNota: dataset_phishing_gt.csv requiere revisión manual,")
    print("ya que el filtro por palabras clave puede traer falsos positivos")
    print("(por ejemplo, la palabra 'guatemala' en un texto no relacionado).")


def guardar_csv(nombre_archivo: str, registros: list[RegistroPhishing]) -> None:
    with open(nombre_archivo, "w", newline="", encoding="utf-8") as f:
        escritor = csv.writer(f)
        escritor.writerow(
            ["url", "fuente", "fecha", "detalle", "coincide_gt", "dominio", "etiqueta"]
        )
        for r in registros:
            dominio = ""
            try:
                dominio = urlparse(r.url).netloc
            except Exception:
                pass
            escritor.writerow(
                [r.url, r.fuente, r.fecha, r.detalle, r.coincide_gt, dominio, r.etiqueta]
            )


if __name__ == "__main__":
    main()
