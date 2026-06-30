# PatagonIA

> Forecasting de riesgo de incendios forestales en la Patagonia Argentina  
> usando datos satelitales públicos y pronóstico meteorológico GFS.

**Licenciatura en Ciencias de Datos · UCA Rosario · 2026**  
Materia A: *Minería de Datos y Big Data* · Materia B: *IA y Aprendizaje Automático I*

---

## Descripción general

PatagonIA construye un dataset geoespacial sobre una grilla hexagonal H3 (resolución 5, ~253 km² por celda, ~16km de lado) sobre la Patagonia Argentina integrando seis fuentes de datos públicas.

Ese dataset es el punto de partida compartido para dos trabajos académicos con enfoques distintos:

| | Minería de Datos y Big Data | IA y Aprendizaje Automático I |
|---|---|---|
| **Pregunta** | ¿Qué perfiles ambientales caracterizan zonas con focos? | ¿Cuánto riesgo de incendio habrá en los próximos 7 días? |
| **Enfoque** | No supervisado — descubrimiento de patrones | Supervisado — regresión y clasificación |
| **Técnicas** | K-Means + Apriori (KNIME) | RF, XGBoost → TFT (sklearn + xgboost) |
| **Rol del dataset** | Entra como caja negra ya construida | Se construye desde cero y se modela |
| **Entregable** | `.knwf` + paper académico PDF 6-8 páginas | Notebooks + app Streamlit + paper IEEE 10-15 páginas |
| **Carpeta** | `knime/` + `reports/` | `notebooks/` + `app/` |

El pipeline de construcción del dataset (`notebooks/01` a `05`) es neutral — no pertenece a ninguna materia.

---

## Fuentes de datos

| Fuente | Contenido | Acceso |
|--------|-----------|--------|
| [NASA FIRMS VIIRS](https://firms.modaps.eosdis.nasa.gov/) | Focos de calor históricos 2012-2023 (375m) | API key gratuita |
| [GFS hindcast (NOAA)](https://nomads.ncep.noaa.gov/) | Temperatura, humedad, viento, precipitación — training Y anomalías | Librería `herbie-data` |
| [ECMWF IFS](https://open-meteo.com/en/docs/ecmwf-api) | Variables meteorológicas para producción (9km) | API REST sin cuenta |
| [SRTM / NASADEM](https://earthdata.nasa.gov/) | Elevación del terreno (30m) | GeoTIFF — cuenta NASA Earthdata |
| [NASA POWER Climatology](https://power.larc.nasa.gov/) | Normales climáticas 1981-2020 (temp, precip, viento, humedad) | API REST sin cuenta |
| [ESA WorldCover 2021](https://esa-worldcover.org/) | Cobertura vegetal dominante | GeoTIFF descarga directa |
| [IGN Argentina](https://datos.gob.ar/dataset/ign) | Rutas y asentamientos | Shapefile descarga directa |

> **Nota**: ERA5 fue descartado del pipeline. GFS hindcast y ERA5 tienen sesgos sistemáticos distintos (~1-3°C en Patagonia). Mezclarlos introduce ruido en las anomalías. Una sola fuente meteorológica: GFS todo el pipeline.

---

## Estructura del repositorio

```
PatagonIA/
├── notebooks/                  ← construcción del dataset + TPs de IA
│   ├── 01_firms_a_h3.py        ← focos VIIRS 2012-2023 → agregación H3 res 5
│   ├── 02_elevacion_clima.py   ← elevación SRTM + clima NASA POWER
│   ├── 03_vegetacion_esa.py    ← cobertura vegetal ESA WorldCover 2021
│   ├── 04_distancias_ign.py    ← distancias a asentamientos y rutas (IGN)
│   ├── 05_join_final.py        ← integración del dataset final
│   ├── 06_eda.ipynb            ← TP1: análisis exploratorio (planeado)
│   ├── 07_regresion_tp2.ipynb  ← TP2: regresión D+1..D+7 (planeado)
│   └── 08_clasificacion_tp3.ipynb ← TP3: clasificación riesgo_alto (planeado)
│
├── knime/                      ← Minería de Datos
│   ├── PatagonIA_Mineria.knwf  ← workflow K-Means + Apriori
│   ├── results_knime/          ← figuras del workflow
│   └── informe_mdbd/           ← artículo académico (PDF)
│
├── app/                        ← TP4: app Streamlit (planeado)
│   ├── app.py
│   └── models/                 ← pipelines serializados (.joblib)
│
├── data/
│   ├── raw/                    ← datos crudos por fuente (no versionados)
│   │   ├── firms/
│   │   ├── esa_worldcover/
│   │   └── ign/
│   ├── processed/              ← patagonia_dataset.csv (entregable compartido)
│   └── static/                 ← grilla H3, variables estáticas
│
├── src/                        ← funciones reutilizables Python
├── model_context/              ← documentación del proyecto y consignas
├── reports/                    ← informes PDF de ambas materias
├── requirements.txt
├── .gitignore
└── README.md
```

---

## Instalación

```bash
git clone https://github.com/aformen9/PatagonIA.git
cd PatagonIA
pip install -r requirements.txt
```

### Credenciales necesarias

**NASA Earthdata** (FIRMS + SRTM) — crear cuenta en https://urs.earthdata.nasa.gov  
Guardar en `~/.netrc`:
```
machine urs.earthdata.nasa.gov login TU_USUARIO password TU_PASSWORD
```

El resto de las fuentes no requieren cuenta.

---

## Ejecutar el pipeline de datos

```bash
# Ejecutar en orden — cada script depende del anterior
python notebooks/01_firms_a_h3.py
python notebooks/02_elevacion_clima.py
python notebooks/03_vegetacion_esa.py
python notebooks/04_distancias_ign.py
python notebooks/05_join_final.py
```

Output final: `data/processed/patagonia_dataset.csv`

---

## Decisiones de diseño clave

| Decisión | Elegido | Descartado | Por qué |
|---|---|---|---|
| Tipo de sistema | Forecasting 7 días | Nowcasting | GeoAlertAR-ML ya hace nowcasting con F1=97.6% en Patagonia |
| Grilla espacial | H3 resolución 5 (~253 km²) | Grilla cuadrada / res 4 | Distancia uniforme a 6 vecinos, alineada con resolución nativa GFS |
| Fuente meteorológica | GFS hindcast (training + anomalías) | ERA5 | Elimina training-serving skew. Una fuente, cero sesgo |
| Producción | ECMWF IFS via Open-Meteo | GFS forecast | 15-20% más preciso en 3-7 días, 9km, open data desde oct 2025 |
| Features temporales | Anomalías vs histórico GFS | Mes/día del año | Con temporalidad directa el modelo aprende estacionalidad, no anomalías |
| Arquitectura ML | 7 modelos RF/XGBoost por horizonte | Un solo modelo | Feature importance por horizonte D+1..D+7 es resultado publicable |

---

## Dataset

Cada fila = hexágono H3 × fecha. ~15 millones de filas (2012-2023).

- **Región**: Patagonia Argentina — lat [-56, -38], lon [-76, -62]
- **Período**: 2012-2023
- **Grilla**: H3 resolución 5, ~3.200 hexágonos burnables
- **Confianza FIRMS**: filtro confidence ≥ 50%
- **Desbalance de clases**: ~90% sin foco, ~10% con foco

---

## Autores

| Nombre | Rol |
|--------|-----|
| Agustín Formenti | IA y ML + Minería de Datos |
| Juan Cruz Chocobares | IA y ML + Minería de Datos |
| Andrés Morenico | IA y ML + Minería de Datos |
| Lorenzo Mendes | Minería de Datos |

---

## Licencia

MIT License — libre uso con atribución.
