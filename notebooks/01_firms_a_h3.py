"""
============================================================================
PatagonIA · Construcción del dataset (pipeline NEUTRAL, compartido) — PASO 1
FIRMS (focos de calor) -> agregación por hexágono H3
============================================================================

Este pipeline construye UNA sola vez el dataset que usan las dos materias:
Minería de Datos (clustering + reglas de asociación en KNIME) e IA (modelado en
Python). No se construye nada dos veces.

QUÉ HACE ESTE PASO
------------------
Toma los focos de calor crudos de NASA FIRMS (satélite VIIRS) de `data/raw/firms/`,
los recorta a la Patagonia, le asigna a cada foco la celda hexagonal H3 que lo
contiene, y agrega todos los focos de cada hexágono en una fila con variables que
describen el "régimen de incendio" histórico (2012-2023) de esa zona.

POR QUÉ H3 Y RESOLUCIÓN 5
-------------------------
H3 (grilla hexagonal de Uber): la distancia del centro de un hexágono a sus 6
vecinos es siempre igual (en una grilla cuadrada los vecinos diagonales están 41%
más lejos). Resolución 5 ≈ 253 km² por celda, alineada con la resolución de los
datos meteorológicos. Es la resolución que fija el proyecto.

POR QUÉ AGREGAR POR HEXÁGONO (una fila = una zona)
--------------------------------------------------
Minería busca DESCUBRIR PATRONES (no supervisado): agrupar zonas parecidas
(K-Means) y encontrar reglas de asociación (Apriori). Por eso cada fila representa
una ZONA y resume su comportamiento histórico, no un foco individual.

ENTRADA : data/raw/firms/fire_archive_SV-C2_767267.csv   (subido por el equipo)
SALIDA  : data/processed/_intermedios/01_fuego.parquet
============================================================================
"""
from pathlib import Path
import pandas as pd
import h3

# --- Parámetros del recorte espacial ---------------------------------------
# Bounding box de la Patagonia argentina (al sur de -38°, entre Atlántico y
# cordillera). Límites tomados del Plan Maestro del proyecto.
RES_H3 = 5
LAT_MIN, LAT_MAX = -56, -38
LON_MIN, LON_MAX = -76, -62

RAW = Path("data/raw/firms/fire_archive_SV-C2_767267.csv")
OUT = Path("data/processed/_intermedios/01_fuego.parquet")
OUT.parent.mkdir(parents=True, exist_ok=True)

# --- 1. Cargar focos crudos -------------------------------------------------
# El CSV de FIRMS trae un foco por fila con columnas satelitales: ubicación,
# brillo (temperatura del píxel en Kelvin), FRP (Fire Radiative Power = energía
# liberada, en MW), confianza de la detección y si fue de día o de noche.
df = pd.read_csv(RAW)
print(f"Focos crudos (toda Argentina): {len(df):,}")

# --- 2. Recortar a la Patagonia --------------------------------------------
df = df[(df.latitude < LAT_MAX) & (df.latitude > LAT_MIN) &
        (df.longitude > LON_MIN) & (df.longitude < LON_MAX)].copy()
print(f"Focos en Patagonia: {len(df):,}")

# --- 3. Asignar cada foco a su hexágono H3 ---------------------------------
# h3.latlng_to_cell devuelve el id del hexágono (res 5) que contiene la coord.
# Es la operación que discretiza el espacio continuo en zonas.
df["hex"] = [h3.latlng_to_cell(la, lo, RES_H3)
             for la, lo in zip(df.latitude, df.longitude)]

# --- 4. Variables temporales derivadas de la fecha -------------------------
df["acq_date"] = pd.to_datetime(df["acq_date"])
df["year"]  = df.acq_date.dt.year
df["month"] = df.acq_date.dt.month
# Verano austral (dic-ene-feb): temporada de incendios en Patagonia.
df["is_summer"] = df.month.isin([12, 1, 2]).astype(int)
# daynight == 'N': detección nocturna (suele indicar incendio más establecido).
df["is_night"]  = (df.daynight == "N").astype(int)
# Confianza VIIRS categórica l/n/h; marcamos las de alta confianza.
df["conf_high"] = (df.confidence == "h").astype(int)

# --- 5. Agregar por hexágono -----------------------------------------------
g = df.groupby("hex")
agg = pd.DataFrame({
    "n_focos":          g.size(),                 # focos totales en 12 años
    "brillo_medio":     g.brightness.mean(),      # intensidad térmica promedio (K)
    "brillo_max":       g.brightness.max(),       # foco más intenso registrado
    "frp_medio":        g.frp.mean(),             # energía radiativa promedio (MW)
    "frp_max":          g.frp.max(),              # pico de energía radiativa
    "brillo_t31_medio": g.bright_t31.mean(),      # temp. de fondo (canal 31)
    "pct_noche":        g.is_night.mean(),        # fracción de focos nocturnos
    "pct_verano":       g.is_summer.mean(),       # estacionalidad (fracción en verano)
    "pct_conf_alta":    g.conf_high.mean(),       # fracción de detecciones de alta confianza
    "n_anios_activo":   g.year.nunique(),         # años distintos con focos (recurrencia)
    "mes_pico":         g.month.agg(lambda s: s.mode().iloc[0]),  # mes más frecuente
}).reset_index()

# --- 6. Centroide del hexágono (para elevación/clima y mapas) ---------------
centroides = [h3.cell_to_latlng(hx) for hx in agg.hex]
agg["lat"] = [c[0] for c in centroides]
agg["lon"] = [c[1] for c in centroides]

# --- 7. Guardar -------------------------------------------------------------
agg.to_parquet(OUT, index=False)
print(f"\nHexágonos con actividad de fuego: {len(agg):,}")
print(f"Guardado en: {OUT}")
print("\nResumen de variables de fuego:")
print(agg.describe().round(2).T[["mean", "min", "max"]])
