# 🛡️ PhishingShield

Detección de phishing en tiempo real mediante Machine Learning, con un backend en **ASP.NET Core MVC** y un microservicio de inferencia en **FastAPI + scikit-learn**. Proyecto de trabajo de graduación enfocado en Guatemala.

![.NET 8](https://img.shields.io/badge/.NET-8.0-512BD4?logo=dotnet&logoColor=white)
![Python 3.12](https://img.shields.io/badge/Python-3.12-3776AB?logo=python&logoColor=white)
![FastAPI](https://img.shields.io/badge/FastAPI-009688?logo=fastapi&logoColor=white)
![scikit-learn](https://img.shields.io/badge/scikit--learn-F7931E?logo=scikitlearn&logoColor=white)
![PostgreSQL](https://img.shields.io/badge/PostgreSQL-16-4169E1?logo=postgresql&logoColor=white)
![Docker](https://img.shields.io/badge/Docker-Compose-2496ED?logo=docker&logoColor=white)

---

## Tabla de contenidos

- [¿Qué hace?](#qué-hace)
- [Arquitectura](#arquitectura)
- [Estructura del repositorio](#estructura-del-repositorio)
- [Puesta en marcha](#puesta-en-marcha)
  - [Opción A: Docker Compose (recomendado)](#opción-a-docker-compose-recomendado)
  - [Opción B: Ejecución local](#opción-b-ejecución-local)
- [El modelo de Machine Learning](#el-modelo-de-machine-learning)
  - [Dataset](#dataset)
  - [Cómo se construyó el dataset de entrenamiento](#cómo-se-construyó-el-dataset-de-entrenamiento)
  - [Características extraídas](#características-extraídas)
  - [Entrenamiento](#entrenamiento)
  - [Reporte visual de entrenamiento](#reporte-visual-de-entrenamiento)
  - [Verificación de sobreajuste (v1)](#verificación-de-sobreajuste-v1)
  - [Corrección del sesgo de longitud de URL (v1)](#corrección-del-sesgo-de-longitud-de-url-v1)
  - [Dataset v2: por qué y cómo](#dataset-v2-por-qué-y-cómo)
  - [Comparación de modelos y modelo de producción](#comparación-de-modelos-y-modelo-de-producción)
- [API](#api)
- [Base de datos](#base-de-datos)
- [Configuración y variables de entorno](#configuración-y-variables-de-entorno)
- [Roadmap](#roadmap)

---

## ¿Qué hace?

1. El usuario introduce una URL en la interfaz web (`PhishingDetector`).
2. El backend ASP.NET envía la URL al microservicio de IA (`phishing_ai`).
3. El microservicio extrae 15 características léxicas de la URL y las pasa por un modelo **HistGradientBoosting** (gradient boosting), elegido entre seis modelos evaluados con el mismo protocolo (ver [Comparación de modelos](#comparación-de-modelos-y-modelo-de-producción)) y entrenado sobre el [dataset v2](#dataset-v2-por-qué-y-cómo) (34,006 URLs reales de phishing/legítimo).
4. El resultado (predicción, probabilidad, nivel de riesgo) se guarda en PostgreSQL y se muestra al usuario, junto con un dashboard histórico.

> Nota: el análisis se basa solo en el texto de la URL; no se visita el sitio ni se consultan listas de reputación.

## Arquitectura

```
┌─────────────────────┐        HTTP/JSON        ┌───────────────────────┐
│   PhishingDetector   │ ───────────────────────▶│      phishing_ai       │
│   (ASP.NET Core MVC) │ ◀─────────────────────── │   (FastAPI + sklearn)  │
│   puerto 5000 / 80    │      /analyze /health    │   puerto 8000           │
└──────────┬───────────┘                          └───────────┬───────────┘
           │                                                   │
           │ EF Core (Npgsql)                                  │ joblib
           ▼                                                   ▼
   ┌───────────────┐                         modelos_comparacion/
   │  PostgreSQL 16 │                         hist_gradient_boosting.pkl
   │  phishing_db   │                         (generado por
   └───────────────┘                          evaluation/comparar_modelos.py)
```

- **`PhishingDetector`** — Aplicación web MVC (.NET 8): formulario de análisis, historial y dashboard de métricas. Persiste cada análisis en PostgreSQL vía EF Core.
- **`phishing_ai`** — Microservicio FastAPI stateless que carga el modelo `.pkl` en memoria (con una predicción de calentamiento al arrancar) y expone `/analyze` y `/health`.
- **`postgres`** — Almacena el historial de análisis (tabla `analyses`).

Los tres servicios se orquestan con `docker-compose.yml`.

## Estructura del repositorio

```
Phishing Detector/
├── docker-compose.yml
├── init.sql
├── ptf.sln
├── recolector_phishing_gt.py     # Recolecta phishing/legítimo desde feeds públicos
│
├── PhishingDetector/              # App web ASP.NET Core MVC
│   ├── Controllers/                #   Home, Dashboard, PhishingApi (REST)
│   ├── Services/                   #   PhishingService, FastApiClient (HttpClient)
│   ├── Data/                       #   AppDbContext (EF Core)
│   ├── Models/                     #   Analysis, AnalyzeRequest, PhishingResult
│   ├── Views/                      #   Razor views (Home, Dashboard)
│   └── Dockerfile
│
└── phishing_ai/                    # Microservicio de inferencia
    ├── main.py                     #   API FastAPI (/analyze, /health)
    ├── feature_extractor.py        #   Extracción de 15 features léxicas de URL
    ├── build_training_dataset.py   #   v1: fusiona/balancea las fuentes crudas -> dataset_entrenamiento.csv
    ├── build_training_dataset_v2.py#   v2: PhiUSIIL + URLhaus (≤15%) + manual -> dataset_entrenamiento_v2.csv
    ├── model_trainer.py            #   v1: entrena RF + LR y genera el reporte
    ├── model_trainer_v2.py         #   v2: entrena RF + LR (phishing_model_v2.pkl) y genera el reporte v2
    ├── elegir_umbral_v2.py         #   Umbral óptimo del RF v2 (split 85/15 agrupado por hostname)
    ├── report_generator.py         #   Genera reports/training_report*.{json,html}
    ├── data/
    │   ├── raw/                          # Fuentes crudas (ver tabla abajo)
    │   ├── dataset_comparativo_gt.csv    # Dataset legado, solo-Guatemala (300 filas)
    │   ├── dataset_entrenamiento.csv     # Dataset v1 (34,031 filas) — histórico
    │   └── dataset_entrenamiento_v2.csv  # Dataset v2 (34,006 filas) — usado por el modelo de producción
    ├── reports/                    #   Reportes de entrenamiento v1 y v2, umbral_v2.json
    ├── modelos_comparacion/        #   Los 6 modelos comparados; hist_gradient_boosting.pkl = producción
    ├── phishing_model_v2.pkl       #   Random Forest v2 (modelo de producción anterior)
    ├── phishing_model.pkl          #   Random Forest v1 (histórico)
    └── Dockerfile

evaluation/                          # Evaluación y comparación de modelos
├── criterio_seleccion.json          #   Criterio de selección, fijado antes de la comparación
├── evaluar_modelo.py                #   v1: CV 5x5, agrupada por hostname, escenario GT, validación externa
├── evaluar_modelo_v2.py             #   v2: mismo protocolo sobre dataset_entrenamiento_v2.csv
├── construir_validacion_externa.py  #   Arma validacion_externa.csv (OpenPhish + legítimas .gt)
├── construir_validacion_externa_v2.py # Arma validacion_externa_v2.csv
├── comparar_modelos.py              #   6 modelos: CV 5x5, por hostname, umbral, validación externa
├── seleccionar_modelo.py            #   Latencia HTTP por modelo + aplica el criterio
├── importancia_permutacion.py       #   Importancia por permutación (HGB y RF)
├── analisis_solapamiento_externa.py #   Hostnames compartidos externa vs. entrenamiento
├── analisis_sobreajuste_mcnemar_youden.py # McNemar todos vs todos (Holm), Cochran Q, Youden, curvas
├── analisis_exploratorio.py         #   EDA de los datasets (v1, v2, GT, externa)
├── youden_gt.py                     #   Umbral de Youden en el escenario Guatemala
├── figura_f1_externa.py             #   Figura de F1 en validación externa
├── prueba_latencia.py               #   Latencia de POST /analyze
└── resultados/                      #   CSV, PNG y JSON de todas las corridas
```

> `analisis_sobreajuste_mcnemar_youden.py` y `analisis_exploratorio.py` escriben en `resultados/analisis_extra/` y `resultados/eda/`; esas carpetas no están en el repositorio hasta que se ejecutan los scripts.

## Puesta en marcha

### Opción A: Docker Compose (recomendado)

Levanta los tres servicios (Postgres, microservicio de IA y app web) con un solo comando. La imagen `phishing_ai` **no entrena** el modelo: copia el `.pkl` ya entrenado (`modelos_comparacion/hist_gradient_boosting.pkl`). Las versiones de scikit-learn, numpy y joblib están fijadas en `requirements.txt` para que coincidan con las que generaron el `.pkl`.

```bash
docker compose up --build
```

| Servicio | URL |
|---|---|
| App web | http://localhost:5000 |
| API de IA (FastAPI) | http://localhost:8000/docs |
| PostgreSQL | localhost:5432 (`postgres` / `postgres`) |

```bash
docker compose down          # detiene los contenedores
docker compose down -v       # además elimina el volumen de datos de Postgres
```

### Opción B: Ejecución local

**1. Microservicio de IA**

```bash
cd phishing_ai
python -m venv .venv && .venv\Scripts\activate   # Windows

pip install -r requirements.txt

uvicorn main:app --reload --port 8000   # carga modelos_comparacion/hist_gradient_boosting.pkl

# Para reentrenar y volver a comparar los 6 modelos:
#   cd ../evaluation && python comparar_modelos.py && python seleccionar_modelo.py
```

**2. Base de datos**

```bash
docker run --name phishing-db -e POSTGRES_PASSWORD=postgres -e POSTGRES_DB=phishing_db -p 5432:5432 -d postgres:16-alpine
```

**3. App web**

Ajusta `PhishingDetector/appsettings.Development.json` con tu cadena de conexión y la URL del microservicio (por defecto `http://localhost:8000`), luego:

```bash
cd PhishingDetector
dotnet run
```

## El modelo de Machine Learning

### Dataset

Hay **tres** datasets disponibles en `phishing_ai/data/`:

| Archivo | Filas | Alcance |
|---|---|---|
| `dataset_comparativo_gt.csv` | 300 | Legado: solo phishing filtrado por palabras clave de Guatemala (46) + legítimos GT (254). Se usa como escenario de evaluación "solo Guatemala". |
| `dataset_entrenamiento.csv` (v1, histórico) | 34,031 | Phishing global (16,943; 98% URLhaus) + legítimos GT (254) + 16,834 URLs legítimas reales (homepage + enlaces internos) recolectadas visitando en vivo 7,138 dominios del ranking Tranco. Balance final: 49.8% phishing / 50.2% legítimo. |
| **`dataset_entrenamiento_v2.csv`** (v2, **producción**) | **34,006** | 17,003 phishing (PhiUSIIL 14,407 + URLhaus 2,554 + manual/urlscan.io 42) / 17,003 legítimas (las de v1, máx. 5 por hostname). Ver [Dataset v2](#dataset-v2-por-qué-y-cómo). |

El modelo de producción (HistGradientBoosting) se entrenó con **v2** mediante `evaluation/comparar_modelos.py`. Las secciones siguientes sobre construcción, entrenamiento, sobreajuste y sesgo de longitud describen el trabajo hecho sobre **v1**, que es la base de v2 (las legítimas de v2 son las mismas de v1).

`model_trainer.py` (v1) usa `dataset_entrenamiento.csv` automáticamente si existe; si no, cae de vuelta al dataset legado. Para forzar uno específico:

```bash
DATASET_PATH=data/dataset_comparativo_gt.csv python model_trainer.py
```

### Cómo se construyó el dataset de entrenamiento

> Esta sección describe la construcción de **v1**. Para v2, ver [Dataset v2](#dataset-v2-por-qué-y-cómo).

`recolector_phishing_gt.py` recolecta de tres feeds de threat intelligence (OpenPhish, URLhaus, PhishTank\*) más una lista curada manualmente de typosquats verificados en urlscan.io, y produce:

- `data/raw/dataset_phishing_global.csv` — **todo** el phishing recolectado, sin filtrar (16,943 URLs).
- `data/raw/dataset_phishing_gt.csv` — subconjunto filtrado por palabras clave relacionadas con Guatemala (bancos, `.gob.gt`, etc.) — 46 URLs, ya incluido dentro del anterior.
- `data/raw/dataset_legitimo_gt.csv` — sitios oficiales guatemaltecos + dominios `.gt` del ranking Tranco (254 URLs).
- `data/raw/candidatos_urlscan_gt.csv` — 411 URLs candidatas encontradas por búsqueda de palabras clave en urlscan.io, **sin verificar/etiquetar** (pendiente de revisión manual, no se usan todavía para entrenar).

\* PhishTank requiere una API key gratuita (ver comentarios en `recolector_phishing_gt.py`); sin ella, esa fuente se omite.

El dataset original de comparación (`dataset_comparativo_gt.csv`) solo combinaba `dataset_phishing_gt.csv` + `dataset_legitimo_gt.csv` (300 filas), descartando los 16,943 registros de `dataset_phishing_global.csv`. Como `feature_extractor.py` es puramente léxico (no usa ninguna señal específica de Guatemala), se amplió el alcance de "phishing dirigido a Guatemala" a "phishing en general" — los casos GT siguen siendo un subconjunto — para aprovechar todo el volumen de datos disponible.

`build_training_dataset.py` genera `dataset_entrenamiento.csv` así:

```bash
cd phishing_ai
python build_training_dataset.py
```

1. Toma las 16,943 URLs de `dataset_phishing_global.csv` como clase `phishing`.
2. Toma las 254 URLs de `dataset_legitimo_gt.csv` como base de `legitimo`.
3. Para balancear las clases, recorre el ranking global de **Tranco top-1M** (con caché local en `data/raw/tranco_top1m.zip`) y **visita en vivo la homepage real** de cada dominio candidato, extrayendo hasta 3 enlaces internos reales (rutas reales del sitio, no solo el dominio) por dominio que responde. Esto corrige un sesgo importante — ver la siguiente sección — y no solo balancea las clases, sino que las hace comparables en estructura.
4. Guarda todo en `data/dataset_entrenamiento.csv` con el mismo esquema (`url,fuente,fecha,detalle,coincide_gt,dominio,etiqueta`) que ya consume `model_trainer.py`.

Recolectar así ~16,800 URLs reales toma cerca de una hora (visita ~14,000 dominios uno por uno, con 60 hilos concurrentes y timeouts cortos; solo ~51% de los dominios responden a tiempo). El resultado queda cacheado en `data/dataset_entrenamiento.csv`, así que no hace falta repetirlo salvo que quieras ampliar el dataset.

> **¿El entrenamiento usa todos los datos que tengo?** Con `dataset_entrenamiento.csv`, sí: las 34,031 filas se usan (0 se descartan; verificable en la consola y en el reporte de cada corrida). Si tienes datos adicionales que aún no están en `data/raw/`, agrégalos ahí siguiendo el mismo esquema de columnas y vuelve a correr `build_training_dataset.py`.

### Características extraídas

`feature_extractor.py` convierte cada URL en 15 características numéricas (sin necesidad de visitar la URL):

`length_url`, `count_dots`, `has_https`, `has_at_symbol`, `has_ip_address`, `count_hyphens`, `suspicious_words_count`, `count_subdomains`, `count_slashes`, `count_question_marks`, `count_equal_signs`, `count_underscores`, `count_percent`, `length_domain`, `count_digits_url`.

### Entrenamiento

Hay tres scripts de entrenamiento, uno por etapa del proyecto:

| Script | Dataset | Genera |
|---|---|---|
| `phishing_ai/model_trainer.py` | v1 | `phishing_model.pkl` (RF) + `phishing_model_lr.pkl`, `reports/training_report.*` |
| `phishing_ai/model_trainer_v2.py` | v2 | `phishing_model_v2.pkl` (RF) + `phishing_model_lr_v2.pkl`, `reports/training_report_v2.*` |
| **`evaluation/comparar_modelos.py`** | v2 | Los 6 modelos en `modelos_comparacion/`, incluido **`hist_gradient_boosting.pkl` (producción)** |

Para regenerar el modelo de producción:

```bash
cd evaluation
python comparar_modelos.py      # entrena y evalúa los 6 modelos (~12 min, SVM es el más lento)
python seleccionar_modelo.py    # requiere poder levantar uvicorn; mide latencia y aplica el criterio
```

El flujo de `model_trainer.py` (v1; `model_trainer_v2.py` es igual pero sobre v2 y con `min_samples_leaf=5`):

```bash
cd phishing_ai
python model_trainer.py
```

Esto:

1. Carga y valida el dataset completo (ver estadísticas en consola).
2. Extrae las 15 características de cada URL.
3. Divide 80/20 train/test (estratificado por clase, `random_state=42`).
4. Entrena un `RandomForestClassifier` (100 árboles, `class_weight="balanced"`). Fue el modelo de producción hasta la [comparación de modelos](#comparación-de-modelos-y-modelo-de-producción); hoy la API usa HistGradientBoosting.
5. Entrena una `LogisticRegression` como comparación (`phishing_model_lr.pkl`).
6. Imprime accuracy/precision/recall/F1 y el top de características más importantes.
7. Genera `reports/training_report.json` y `reports/training_report.html` con todos los resultados.

### Reporte visual de entrenamiento

Cada corrida de `model_trainer.py` genera automáticamente un **dashboard HTML autocontenido** en `phishing_ai/reports/training_report.html` — ábrelo con doble clic, no necesita servidor ni conexión a internet:

- KPIs del dataset (total de URLs, phishing, legítimas, % phishing, tamaño train/test).
- Comparación de Accuracy / Precision / Recall / F1 entre Random Forest y Logistic Regression.
- Importancia de características (top 10, Random Forest).
- Distribución de clases (donut chart).
- Matrices de confusión de ambos modelos.
- Curvas de aprendizaje (entrenamiento vs. validación) de ambos modelos — ver [Verificación de sobreajuste](#verificación-de-sobreajuste-v1).

Es la forma más rápida de ver "cómo le fue" al modelo después de cada reentrenamiento, sin tener que leer la salida de consola.

### Verificación de sobreajuste (v1)

`model_trainer.py` calcula curvas de aprendizaje (F1 de entrenamiento vs. validación al usar 10%…100% de los datos, 5-fold CV) para Random Forest y Logistic Regression, y las incluye en `reports/training_report.{json,html}`. Se calculan con dos esquemas de validación:

- **K-fold estratificado aleatorio** — el mismo criterio que el split train/test principal.
- **GroupKFold por dominio** — ~81% de las URLs del dataset comparten dominio con otra (13,927 dominios para 34,031 URLs), así que un split aleatorio deja URLs "hermanas" a ambos lados y puede sobreestimar el desempeño. Este esquema mide cómo rinde el modelo sobre **dominios que nunca vio**.

**Diagnóstico inicial.** Con el Random Forest sin restricciones (`RandomForestClassifier(n_estimators=100, class_weight="balanced")`), la brecha train/validación era:

| Modelo | Validación | F1 train | F1 validación | Brecha |
|---|---|---|---|---|
| Random Forest | K-fold aleatorio | 99.8% | 98.6% ±0.1 | 1.1 pts |
| Random Forest | Agrupado por dominio | 99.8% | 97.1% ±1.4 | **2.6 pts** |
| Logistic Regression | K-fold aleatorio | 96.6% | 96.7% ±0.0 | ≈0 |
| Logistic Regression | Agrupado por dominio | 96.7% | 96.5% ±1.1 | 0.2 pts |

Random Forest es el que sobreajusta, no Logistic Regression: un F1 de entrenamiento de 99.8% junto con un F1 de validación notablemente menor (97.1% sobre dominios nunca vistos) es la firma clásica de sobreajuste. Logistic Regression tiene menor F1 absoluto, pero su brecha train/validación es casi cero — eso es un modelo simple con menos capacidad, no uno sobreajustado. (Es fácil confundir "rinde peor" con "está sobreajustado": son cosas distintas y, de hecho, contrarias.)

**Corrección aplicada.** El Random Forest de producción se entrena sin límite de profundidad (`n_estimators=100` sin `max_depth`), lo que le permite crear hojas hiper-específicas que memorizan particularidades de dominios individuales — justo lo que expone el esquema agrupado por dominio. Se probaron variantes de regularización (`max_depth`, `min_samples_leaf`) evaluadas con `GroupKFold` por dominio (la métrica que importa: desempeño sobre sitios nuevos, no sobre el split aleatorio con fuga de dominios hermanos). `min_samples_leaf=5` fue la que dio mejor resultado — no es un trade-off, **mejora en ambos frentes a la vez**:

| Modelo (Random Forest) | F1 train | F1 validación (por dominio) | Brecha |
|---|---|---|---|
| Sin regularizar | 99.8% | 97.1% | 2.6 pts |
| `min_samples_leaf=5` | 98.6% | **97.3%** | **1.4 pts** |

`model_trainer.py` ya usa `min_samples_leaf=5` por defecto (ver el comentario junto a la instanciación del `RandomForestClassifier`).

**Lectura.** El F1 de validación por dominio sube ligeramente (97.1% → 97.3%) y la brecha se reduce a poco más de la mitad (2.6 → 1.4 pts), simplemente evitando que el árbol memorice hojas de una sola URL/dominio. Sigue quedando una brecha pequeña (1.4 pts) entre entrenamiento y dominios nuevos — esperable y razonable para un ensamble de árboles — que vale la pena mencionar como limitación conocida en la tesis, pero ya no es la señal de alerta que era antes. Las métricas del split aleatorio (ver tabla de métricas más abajo) siguen siendo una cota ligeramente optimista del desempeño sobre sitios nuevos; el número mejor fundamentado para "qué tan bien detecta phishing en un sitio que nunca vio" es el F1 agrupado por dominio (97.3%), no el del split aleatorio.

### Corrección del sesgo de longitud de URL (v1)

**El problema.** La primera versión de `dataset_entrenamiento.csv` (33,886 filas) tomaba los dominios de Tranco tal cual (`https://google.com/`) como clase `legitimo`: URLs raíz, cortas, sin ruta. Contra phishing real (URLs largas de URLhaus/OpenPhish, con rutas y parámetros), el Random Forest obtenía **Accuracy/F1 ≈ 99.8%** — una señal de alerta, no un éxito. Las características más importantes eran `length_url`, `length_domain` y `count_dots` (70% de la importancia combinada): el modelo estaba aprendiendo "¿es una URL larga?", no "¿es phishing?". Un sitio legítimo con una URL larga (un artículo de noticias, un enlace de tienda con parámetros) se habría clasificado como phishing por esta sola razón.

**La corrección.** `build_training_dataset.py` ahora **visita en vivo la homepage real** de cada dominio candidato de Tranco y extrae hasta 3 enlaces internos reales (rutas reales del sitio, ej. `/productos/123`, `/es/soporte`), en vez de usar solo el dominio raíz. Así la clase `legitimo` también tiene URLs largas y con estructura real, y la longitud deja de ser una señal útil para separar las clases.

**Resultado, verificado reentrenando sobre el dataset corregido (34,031 filas, 49.8%/50.2%):**

| Métrica (Random Forest) | Antes (dominios raíz) | Después (rutas reales) |
|---|---|---|
| Accuracy | 99.84% | **98.56%** |
| F1 | 99.84% | **98.54%** |
| Importancia de `length_url` | 36.5% (#1) | **6.2% (#6)** |
| Importancia de `length_domain` | 32.9% (#2) | **5.7% (#7)** |
| Feature #1 | `length_url` | **`has_https` (24.6%)** |
| Feature #2 | `length_domain` | **`count_digits_url` (24.6%)** |
| Feature #3 | `count_hyphens` | **`has_ip_address` (12.2%)** |

La caída de accuracy (99.8% → 98.6%) **es la señal de que la corrección funcionó**: el problema se volvió genuinamente más difícil porque ambas clases ahora tienen URLs de longitud/complejidad comparable, y el modelo tuvo que apoyarse en señales reales de phishing (ausencia de HTTPS, uso de IP en vez de dominio, cantidad de dígitos, subdominios) en vez de la forma de la URL. La brecha entre Random Forest (98.56%) y Logistic Regression (96.44%, antes 99.65%) también se amplió, como se espera cuando el problema deja de ser trivialmente separable.

> Nota: la tabla de arriba (98.56%/98.54%) corresponde al Random Forest **sin regularizar**, para aislar el efecto de la corrección de dataset. El Random Forest v1 regularizado además tiene `min_samples_leaf=5` (ver [Verificación de sobreajuste](#verificación-de-sobreajuste-v1)), que lo deja en Accuracy 98.27% / F1 98.24% sobre el split aleatorio, pero con mejor F1 (97.3% vs 97.1%) y mucha menor brecha (1.4 vs 2.6 pts) sobre dominios nuevos — el número que de verdad importa para sitios nuevos.

**Limitaciones que persisten (para documentar en la tesis):**

1. Los enlaces internos se extraen del HTML crudo con una expresión regular simple (no un parser HTML completo) y sin ejecutar JavaScript — sitios que renderizan su navegación por JS pueden aportar solo la homepage, no subpáginas.
2. Solo ~51% de los dominios de Tranco visitados respondieron a tiempo (timeout corto, deliberado, para mantener el tiempo de recolección en ~1 hora); esto podría sesgar levemente la muestra hacia sitios más rápidos/disponibles.
3. El escenario exclusivamente Guatemala (`dataset_comparativo_gt.csv`, 300 URLs) se evalúa en `evaluation/evaluar_modelo.py` (`resultados/metricas_gt.csv`: F1 0.766 ± 0.079 con RF y 0.832 ± 0.072 con LR, validación cruzada 5×5), pero con solo 46 phishing la varianza es alta y no se reportan aún métricas GT para el modelo de producción.

### Dataset v2: por qué y cómo

**El hallazgo que lo motivó.** Al medir el Random Forest v1 contra phishing real y nuevo de OpenPhish (`evaluation/validacion_externa.csv`, 562 URLs), la precisión fue 0.986 pero el **recall solo 0.488** (F1 0.652): detectaba menos de la mitad del phishing real. Causa: el 98% de la clase phishing de v1 venía de URLhaus, que es mayormente distribución de malware (payloads en IP cruda), no phishing de credenciales. El modelo v1 era muy bueno en su propia distribución, pero esa distribución no representaba el phishing que llega a los usuarios.

**Composición de v2** (`phishing_ai/build_training_dataset_v2.py`):

| Clase | Fuente | URLs |
|---|---|---|
| Phishing | PhiUSIIL Phishing URL Dataset (UCI) | 14,407 |
| Phishing | URLhaus (limitado a ≤15% de la clase, muestreo aleatorio) | 2,554 |
| Phishing | Manual/urlscan.io (typosquats de bancos GT) | 42 |
| Legítimo | Las legítimas de v1 (Tranco+crawl, Tranco `.gt`, oficiales) | 17,003 |

Reglas: deduplicado por URL, **máximo 5 URLs por hostname** en cada clase (reduce la memorización de un mismo sitio), balance 50/50, `random_state=42`. **OpenPhish se excluyó a propósito del entrenamiento** para reservarlo a la validación externa. En v2, 42% del phishing usa HTTPS y 12.4% usa IP en el hostname, contra 99.6% y 0% en las legítimas (`resultados/composicion_features_v2.csv`).

**Resultado (Random Forest, misma configuración):**

| Medición | v1 | v2 |
|---|---|---|
| F1 validación cruzada 5×5 | 0.983 | 0.938 |
| F1 agrupado por hostname | 0.969 | 0.935 |
| Recall en validación externa | 0.488 | 0.758 (umbral 0.5) / 0.822 (umbral 0.40) |
| F1 en validación externa | 0.652 | 0.852 (umbral 0.5) / 0.873 (umbral 0.40) |

El F1 interno baja porque el problema es más realista; el F1 contra phishing real nuevo sube 22 puntos.

**Validación externa v2** (`evaluation/construir_validacion_externa_v2.py`): se intentó descargar un lote nuevo de OpenPhish, pero el feed no había rotado (0 URLs nuevas). Como fallback documentado se reutilizaron las mismas 281 URLs de phishing y 281 legítimas `.gt` de la validación v1; ninguna estuvo en el entrenamiento de v1 ni de v2. Esto además permite comparar v1 y v2 sobre exactamente los mismos ítems.

### Comparación de modelos y modelo de producción

Se compararon seis modelos sobre el dataset v2 (34,006 URLs) con las mismas 15 características y el mismo protocolo (`evaluation/comparar_modelos.py`):

1. Validación cruzada repetida 5×5.
2. Validación agrupada por hostname (StratifiedGroupKFold 5).
3. Umbral de decisión propio de cada modelo, elegido maximizando F1 en un split 85/15 **del dataset de entrenamiento** agrupado por hostname. La validación externa nunca se usa para elegir umbrales.
4. Validación externa con 562 URLs nuevas (`evaluation/validacion_externa_v2.csv`) y prueba de McNemar contra Random Forest.
5. Latencia HTTP de `POST /analyze`.

El criterio se fijó antes de correr nada (`evaluation/criterio_seleccion.json`): mayor F1 en validación externa con p95 de latencia < 200 ms. Si dos modelos quedan a 0.01 o menos, desempata el F1 agrupado por hostname.

| Modelo | F1 CV 5×5 | F1 por hostname | F1 externa | Umbral | p95 HTTP |
|---|---|---|---|---|---|
| **HistGradientBoosting** | **0.941** | 0.935 | **0.897** | 0.31 | 9 ms |
| SVM (RBF) | 0.925 | 0.924 | 0.880 | 0.29 | 7 ms |
| Random Forest | 0.938 | 0.935 | 0.873 | 0.40 | 61 ms |
| Regresión logística | 0.912 | 0.912 | 0.845 | 0.38 | 5 ms |
| MLP (64, 32) | 0.934 | 0.933 | 0.832 | 0.46 | 5 ms |
| KNN (k=15) | 0.871 | 0.868 | 0.795 | 0.41 | 7 ms |

Gana **HistGradientBoosting** (McNemar contra Random Forest: p = 0.029). Ningún otro modelo quedó a 0.01 o menos de su F1 externo (SVM queda a 0.017), así que no se aplicó desempate. Resultados completos en `evaluation/resultados/comparacion_*`.

En validación externa, con su umbral de 0.31, HGB obtiene precisión 0.945, recall 0.854 y ROC-AUC 0.966 (TN 267 · FP 14 · FN 41 · TP 240). Con el umbral por defecto 0.5 su F1 sería 0.864 y la diferencia con RF no sería significativa (p = 0.42), así que el umbral elegido aporta parte de la ganancia.

**Sobreajuste del modelo de producción** (F1 de entrenamiento vs. validación, StratifiedGroupKFold 5 por hostname sobre v2):

| Modelo | F1 entrenamiento | F1 validación | Brecha |
|---|---|---|---|
| HistGradientBoosting | 0.951 | 0.935 | 1.6 pts |
| Random Forest (`min_samples_leaf=5`) | 0.947 | 0.935 | 1.2 pts |
| Regresión logística | 0.912 | 0.912 | 0.1 pts |

La caída de 0.935 (interno) a 0.897 (externo) no es sobreajuste sino cambio de distribución: el phishing externo viene de otra fuente (OpenPhish).

**Importancia de características** (por permutación, porque HGB no expone `feature_importances_`; `evaluation/importancia_permutacion.py`): las que más pesan son `length_domain` (caída de F1 de 0.183), `has_https` (0.162), `count_digits_url` (0.068) y `count_slashes` (0.044); `suspicious_words_count` aporta casi nada (0.001).

**Limitaciones de la validación externa:**

- Ninguna URL es idéntica a una del entrenamiento. Del lado phishing, 2 de las 281 URLs comparten hostname con el entrenamiento; del lado legítimo, 259 de 281 sí lo comparten (son páginas distintas de sitios `.gt` ya vistos, porque Tranco solo tiene ~236 dominios `.gt` y casi todos están en entrenamiento). Ver `analisis_solapamiento_externa.py`. El recall es una medida limpia; la precisión puede ser optimista.
- La validación externa se usó para **elegir** el modelo entre seis, así que su F1 (0.897) ya no es una estimación totalmente insesgada. Se mitigó fijando el criterio antes de ver resultados y no usándola nunca para elegir umbrales; lo ideal sería confirmar con un lote nuevo de OpenPhish.

## API

### Microservicio FastAPI (`phishing_ai`)

Base URL: `http://localhost:8000` · Docs interactivas: `/docs`

| Método | Ruta | Descripción |
|---|---|---|
| `POST` | `/analyze` | Recibe `{ "url": "..." }`, devuelve `is_phishing`, `probability`, `risk_level` y el vector de `features`. `LOW`: p < 0.31 (legítima); `MEDIUM`: 0.31 ≤ p < 0.70; `HIGH`: p ≥ 0.70. |
| `GET` | `/health` | Estado del servicio y si el modelo está cargado. |

### Backend ASP.NET (`PhishingDetector`)

Base URL: `http://localhost:5000`

| Método | Ruta | Descripción |
|---|---|---|
| `POST` | `/api/phishing/analyze` | Analiza una URL (llama internamente al microservicio de IA) y persiste el resultado. |
| `GET` | `/api/phishing/history?limit=50` | Historial de análisis. |
| `GET` | `/api/dashboard/metrics` | Métricas agregadas (totales, riesgo, últimos 7 días). |
| `GET` | `/` | Interfaz web de análisis de URL. |
| `GET` | `/Dashboard` | Dashboard visual con Chart.js. |

## Base de datos

Tabla `analyses` (ver `init.sql`):

| Columna | Tipo | Descripción |
|---|---|---|
| `Id` | `SERIAL` | PK |
| `Url` | `VARCHAR(2048)` | URL analizada |
| `Result` | `VARCHAR(20)` | `PHISHING` / `LEGITIMATE` |
| `Probability` | `DOUBLE PRECISION` | Probabilidad de phishing (0-1) |
| `RiskLevel` | `VARCHAR(10)` | `LOW` / `MEDIUM` / `HIGH` |
| `CreatedAt` | `TIMESTAMPTZ` | Fecha del análisis |

En desarrollo local, `Program.cs` llama a `db.Database.EnsureCreated()` al arrancar, así que `init.sql` es opcional salvo cuando se provisiona la base directamente vía Docker.

## Configuración y variables de entorno

| Variable | Dónde | Por defecto | Descripción |
|---|---|---|---|
| `ConnectionStrings__DefaultConnection` | `PhishingDetector` | ver `appsettings.json` | Cadena de conexión a PostgreSQL |
| `FastApi__BaseUrl` | `PhishingDetector` | `http://localhost:8000` | URL del microservicio de IA |
| `MODEL_PATH` | `phishing_ai` | `modelos_comparacion/hist_gradient_boosting.pkl` | Ruta del modelo `.pkl` a cargar |
| `DECISION_THRESHOLD` | `phishing_ai` | `0.31` | Probabilidad desde la que una URL se clasifica como phishing (y riesgo `MEDIUM`) |
| `DATASET_PATH` | `phishing_ai` | `data/dataset_entrenamiento.csv` (o legado si no existe) | CSV a usar en `model_trainer.py` (v1). `model_trainer_v2.py` y `evaluation/comparar_modelos.py` usan siempre `dataset_entrenamiento_v2.csv` e ignoran esta variable |

## Roadmap

- [x] Mitigar la fuga de "forma de URL" (negativos con rutas reales, no solo dominio raíz) — ver [Corrección del sesgo de longitud de URL](#corrección-del-sesgo-de-longitud-de-url-v1).
- [x] Diagnosticar y reducir el sobreajuste del Random Forest sobre dominios nuevos — ver [Verificación de sobreajuste](#verificación-de-sobreajuste-v1).
- [x] Recomponer la clase phishing para que represente phishing de credenciales (dataset v2) — ver [Dataset v2](#dataset-v2-por-qué-y-cómo).
- [x] Comparar 6 modelos con un criterio fijado de antemano y validación externa — ver [Comparación de modelos](#comparación-de-modelos-y-modelo-de-producción).
- [ ] Revisar manualmente `data/raw/candidatos_urlscan_gt.csv` (411 URLs) e incorporarlas al dataset.
- [ ] Añadir features de red (edad del dominio, certificado TLS, reputación de IP).
- [ ] Reportar métricas del subconjunto Guatemala para el modelo de producción (hoy solo existen para RF/LR v1 en `resultados/metricas_gt.csv`).
- [ ] Confirmar el F1 externo con un lote nuevo de OpenPhish (el actual se usó para elegir el modelo).
- [ ] Autenticación/roles para el dashboard.
