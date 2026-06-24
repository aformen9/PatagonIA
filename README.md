# PatagonIA

> Análisis de patrones ambientales e incendios forestales en la Patagonia Argentina  
> usando datos satelitales públicos y reanálisis climático.

**Licenciatura en Ciencias de Datos · UCA Rosario · 2026**  
Materia A: *Minería de Datos y Big Data* · Materia B: *IA y Aprendizaje Automático 1*

---

## Descripción general

PatagonIA construye un dataset geoespacial de celdas 25km × 25km sobre la Patagonia Argentina
(lat [-55, -37], lon [-73, -62]) integrando cinco fuentes de datos públicas.

Ese dataset es el punto de partida compartido para dos trabajos académicos con enfoques distintos:

| | Minería de Datos y Big Data | IA y Aprendizaje Automático 1 |
|---|---|---|
| **Pregunta** | ¿Qué perfiles ambientales caracterizan zonas con focos? | ¿Puedo predecir si habrá un foco en 7 días? |
| **Enfoque** | No supervisado — descubrimiento de patrones | Supervisado — regresión y clasificación |
| **Técnicas** | K-Means + Apriori (KNIME) | Ridge, RF, XGBoost, ensamble (sklearn) |
| **Variable target** | No entra como input del modelo | `foco_presente` es el label |
| **Entregable** | `.knwf` + paper académico PDF | Notebooks + app Streamlit + paper IEEE |
| **Carpeta** | `mineria/` | `ia/` |

El pipeline de construcción del dataset (`pipeline/`) es neutral — no pertenece a ninguna materia.

---

## Fuentes de datos

| Fuente | Contenido | Acceso |
|--------|-----------|--------|
| [NASA FIRMS VIIRS](https://firms.modaps.eosdis.nasa.gov/) | Focos de calor históricos (375m) | CSV gratuito |
| [ERA5 / Copernicus CDS](https://cds.climate.copernicus.eu/) | Temperatura, humedad, viento, precipitación | API `cdsapi` |
| [SRTM / NASADEM](https://earthdata.nasa.gov/) | Elevación del terreno (30m) | GeoTIFF |
| [ESA WorldCover 2021](https://esa-worldcover.org/) | Cobertura vegetal dominante | GeoTIFF |
| [IGN Argentina](https://datos.gob.ar/) | Red vial nacional | Shapefile |

---

## Estructura del repositorio

```
PatagonIA/
├── pipeline/               ← construcción del dataset (compartido entre materias)
│   ├── 01_descarga_firms.ipynb
│   ├── 02_descarga_era5.ipynb
│   ├── 03_join_grilla.ipynb
│   └── 04_features_engineering.ipynb
│
├── mineria/                ← Minería de Datos y Big Data (entrega: junio 2026)
│   ├── README.md
│   ├── knime/              ← workflow .knwf (entregable oficial)
│   ├── notebooks/          ← EDA exploratorio previo a KNIME
│   └── report/             ← paper académico PDF + figuras
│
├── ia/                     ← IA y Aprendizaje Automático 1 (entrega: julio 2026)
│   ├── README.md
│   ├── notebooks/          ← TPs 1-4
│   ├── app/                ← Streamlit deploy
│   ├── models/             ← pipelines serializados (.joblib)
│   └── reports/            ← informes PDF por TP
│
├── data/
│   ├── raw/                ← datos crudos por fuente (no versionados)
│   ├── processed/          ← patagonia_dataset.csv (versionado)
│   └── static/             ← GeoTIFFs y shapefiles estáticos (no versionados)
│
├── requirements.txt
├── .gitignore
└── README.md
```

---

## Instalación

```bash
git clone https://github.com/TU_USUARIO/PatagonIA.git
cd PatagonIA
python -m venv .venv
source .venv/bin/activate        # Linux / macOS
.venv\Scripts\activate           # Windows
pip install -r requirements.txt
```

### Credenciales necesarias

**ERA5 / Copernicus CDS** — crear cuenta en https://cds.climate.copernicus.eu  
Guardar en `~/.cdsapirc`:
```
url: https://cds.climate.copernicus.eu/api/v2
key: TU_UID:TU_API_KEY
```

**NASA Earthdata (SRTM)** — crear cuenta en https://urs.earthdata.nasa.gov  
Guardar en `~/.netrc`:
```
machine urs.earthdata.nasa.gov login TU_USUARIO password TU_PASSWORD
```

---

## Ejecutar el pipeline de datos

```bash
# Ejecutar en orden — cada notebook depende del anterior
jupyter lab pipeline/01_descarga_firms.ipynb
jupyter lab pipeline/02_descarga_era5.ipynb
jupyter lab pipeline/03_join_grilla.ipynb
jupyter lab pipeline/04_features_engineering.ipynb
```

Output final: `data/processed/patagonia_dataset.csv`

---

## Decisiones metodológicas del dataset

- **Grilla 25km:** justificada por la resolución nativa de ERA5 (~28km)
- **Región:** Patagonia Argentina — lat [-55, -37], lon [-73, -62]
- **Período:** 2019–2023 (5 años, ~1.100–1.400 celdas)
- **Confianza FIRMS:** filtro confidence ≥ 50% para reducir falsos positivos
- **Desbalance de clases:** ~90% celdas sin foco, ~10% con foco — tratado distinto en cada materia

---

## Autores

| Nombre | GitHub |
|--------|--------|
| Agustín Formenti | [@TU_USUARIO](https://github.com/TU_USUARIO) |
| Juan Chocobares | [@choco721](https://github.com/choco721) |
| Javier (integrante 3) | — |

---

## Licencia

MIT License — libre uso con atribución.
