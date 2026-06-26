# Construcción del dataset PatagonIA (pipeline neutral)

Estos scripts construyen **una sola vez** el dataset que usan las dos materias
(Minería de Datos en KNIME e IA en Python). Cada fila es un **hexágono H3
(resolución 5, ~253 km²)** de la Patagonia con actividad de incendio 2012-2023;
cada columna es una característica ambiental o del régimen de fuego de esa zona.

## Cómo correrlo

Desde la raíz del repo, con la venv:

```powershell
.\.venv\Scripts\python.exe notebooks\01_firms_a_h3.py
.\.venv\Scripts\python.exe notebooks\02_elevacion_clima.py
.\.venv\Scripts\python.exe notebooks\03_vegetacion_esa.py
.\.venv\Scripts\python.exe notebooks\04_distancias_ign.py
.\.venv\Scripts\python.exe notebooks\05_join_final.py
```

Cada paso lee el parquet intermedio del anterior (`data/processed/_intermedios/`)
y el último genera el entregable: **`data/processed/patagonia_dataset.csv`**.

## Qué hace cada paso y por qué

| Paso | Script | Aporta | Fuente | Método / justificación |
|---|---|---|---|---|
| 1 | `01_firms_a_h3.py` | Régimen de fuego por zona (n_focos, brillo, FRP, estacionalidad, recurrencia) | NASA FIRMS VIIRS (subido a `data/raw/`) | Recorte a Patagonia + asignación a H3 res 5 + agregación por hexágono |
| 2 | `02_elevacion_clima.py` | Elevación + clima (temp, precip, viento, humedad relativa) | OpenTopoData SRTM 30m (elevación); NASA POWER Climatology (clima 1981-2020) | Elevación por hexágono; clima por celda madre res 4 (~28 km) para evitar pseudo-replicación |
| 3 | `03_vegetacion_esa.py` | Cobertura vegetal dominante (combustible) | ESA WorldCover 2021 (10 m) | Streaming COG vía `/vsicurl/`; moda de 7 puntos por hexágono |
| 4 | `04_distancias_ign.py` | Distancia a asentamiento y a ruta nacional (presión humana) | IGN Argentina (WFS) | Reproyección a Azimutal Equidistante + vecino más cercano (`sjoin_nearest`) |
| 5 | `05_join_final.py` | Integración, limpieza, dataset final | — | Une todo, quita faltantes, ordena y guarda CSV + parquet |

## Variables del dataset final

- **Identificación**: `hex`, `lat`, `lon`
- **Régimen de fuego** (FIRMS): `n_focos`, `brillo_medio`, `brillo_max`,
  `frp_medio`, `frp_max`, `brillo_t31_medio`, `pct_noche`, `pct_verano`,
  `pct_conf_alta`, `n_anios_activo`, `mes_pico`
- **Terreno y clima**: `elevacion`, `temp_media`, `precip_anual`,
  `viento_medio`, `humedad_relativa`
- **Vegetación**: `cobertura_veg` (categórica: bosque/matorral/pastizal/…)
- **Presión humana** (IGN): `dist_asentamiento_km`, `dist_ruta_km`

## Notas de método (para defender el trabajo)

- **¿Por qué hexágonos H3?** Distancia uniforme a los 6 vecinos (la grilla
  cuadrada tiene los vecinos diagonales 41% más lejos). Es el estándar emergente
  en estudios de incendios geoespaciales.
- **¿Por qué OpenTopoData para elevación?** Open-Meteo tiene cuota horaria; OpenTopoData
  SRTM 30m (NASA SRTM v3) no la tiene. Solo necesitamos 20 llamadas para 1.980
  hexágonos, dentro de cualquier límite.
- **¿Por qué NASA POWER y no ERA5/Open-Meteo Archive?** Para caracterizar el clima
  de cada zona alcanzan las normales climáticas (promedios 1981-2020). NASA POWER
  Climatology API las devuelve en una sola llamada por punto, sin cuota horaria, y
  es citable (Stackhouse et al. 2019). El pipeline pesado de GFS (Herbie) queda para
  el modelado temporal de IA.
- **¿Por qué ESA por streaming?** Los tiles pesan ~1,5 GB pero son COG; leemos solo
  los píxeles necesarios. Reproducible desde el bucket público de ESA.
- **¿Por qué la distancia a asentamientos?** El 95% de los incendios patagónicos
  son de origen humano (SNMF); Kitzberger et al. la señalan como predictor estático
  clave.
