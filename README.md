# PatagonIA 

> Sistema de predicción de riesgo de incendios forestales en la Patagonia argentina  
> usando datos satelitales públicos y reanálisis climático.

**Argumento central:** alternativa interpretable sobre datos tabulares de libre acceso frente a enfoques de deep learning con imágenes satelitales costosas. Replicable sin infraestructura costosa, orientada a decisiones a escala regional en contexto argentino.

---

## Descripción

PatagonIA construye un dataset geoespacial de celdas 25km × 25km para la región patagónica (lat [-55, -37], lon [-73, -62]) integrando cinco fuentes de datos públicas. Sobre ese dataset entrena modelos de regresión (predicción de focos a 7 días) y clasificación binaria (riesgo alto/bajo), con énfasis en interpretabilidad y reproducibilidad.

El proyecto fue desarrollado como trabajo final de la materia **IA y Aprendizaje Automático 1** — Licenciatura en Ciencias de Datos, UCA Rosario (2026).

---

## Fuentes de datos

| Fuente | Contenido | Acceso |
|--------|-----------|--------|
| NASA FIRMS VIIRS | Focos de calor históricos (375m, 2010–2023) | CSV gratuito, confidence ≥ 50% |
| ERA5 / Copernicus CDS | Temperatura, humedad, viento, precipitación diaria | API `cdsapi`, NetCDF |
| SRTM / NASADEM | Elevación del terreno (30m) | GeoTIFF, earthdata.nasa.gov |
| ESA WorldCover 2021 | Cobertura vegetal dominante por celda | GeoTIFF, esa-worldcover.org |
| IGN Argentina | Red vial nacional | Shapefile, datos.gob.ar |

---

## Estructura del repositorio

```
PatagonIA/
├── data/
│   ├── raw/          # FIRMS CSV por año, ERA5 NetCDF por bloque (no versionados)
│   ├── processed/    # Dataset final en parquet (no versionado)
│   └── static/       # GeoTIFF elevación/cobertura, shapefile rutas (no versionados)
├── notebooks/
│   ├── 01_descarga_firms.ipynb
│   ├── 02_descarga_era5.ipynb
│   ├── 03_join_grilla.ipynb
│   ├── 04_features_engineering.ipynb
│   ├── 05_eda_tp1.ipynb
│   ├── 06_regresion_tp2.ipynb
│   └── 07_clasificacion_tp3.ipynb
├── app/
│   ├── app.py
│   ├── model_pipeline.joblib
│   └── stats_historicas.parquet
├── models/           # Pipelines serializados (.joblib)
├── reports/          # Informes PDF de cada TP
├── requirements.txt
├── CONTRIBUTING.md
└── README.md
```

---

## Instalación y reproducción

### 1. Clonar el repositorio

```bash
git clone https://github.com/TU_USUARIO/PatagonIA.git
cd PatagonIA
```

### 2. Crear entorno virtual e instalar dependencias

```bash
python -m venv .venv
source .venv/bin/activate        # Linux / macOS
.venv\Scripts\activate           # Windows

pip install -r requirements.txt
```

### 3. Configurar credenciales de APIs

**ERA5 / Copernicus CDS:**  
Crear cuenta en https://cds.climate.copernicus.eu y generar una API key.  
Guardar en `~/.cdsapirc`:

```
url: https://cds.climate.copernicus.eu/api/v2
key: TU_UID:TU_API_KEY
```

**NASA Earthdata (SRTM):**  
Crear cuenta en https://urs.earthdata.nasa.gov y guardar credenciales en `~/.netrc`:

```
machine urs.earthdata.nasa.gov login TU_USUARIO password TU_PASSWORD
```

### 4. Ejecutar notebooks en orden

```
01 → 02 → 03 → 04 → 05 → 06 → 07
```

Los notebooks 01 y 02 descargan los datos crudos. El notebook 03 construye la grilla y hace los joins. Los siguientes realizan feature engineering, EDA y modelado.

>  Los archivos de datos (`data/`) no están versionados por su tamaño. Ver `.gitignore`.

---

## Decisiones metodológicas clave

- **Split temporal estricto:** train < 2022, test ≥ 2022. Nunca aleatorio.
- **Anti-leakage en rolling windows:** todas las ventanas temporales aplican `shift(1)` sobre la celda antes del cálculo.
- **Desbalance de clases (~90/10):** tratado con `class_weight='balanced'`. Métricas: F1, AUC-ROC, AUC-PR. Accuracy descartada.
- **Pipeline sklearn obligatorio:** scaler + encoder + modelo en un único objeto serializado con joblib.
- **Grilla 25km:** justificada por la resolución nativa de ERA5 (~28km).

---

## Estructura de TPs

| TP | Período | Contenido |
|----|---------|-----------|
| TP1 | 2–15 jun | EDA y preparación de datos |
| TP2 | 16–22 jun | Regresión (Ridge, RF, XGBoost) |
| TP3 | 23–29 jun | Clasificación + Clustering + Ensamble |
| TP4 | 30 jun–6 jul | Deploy Streamlit + Paper IEEE |

---

## App

La app está disponible en: `https://TU_APP.streamlit.app` *(disponible desde julio 2026)*

Permite ingresar condiciones climáticas y topográficas de una celda patagónica y obtener una predicción de riesgo de incendio con los factores más influyentes.

---

## Autores

- Juan — UCA Rosario, Licenciatura en Ciencias de Datos
- [Integrante 2]
- [Integrante 3]

---

## Licencia

MIT License — libre uso con atribución.
