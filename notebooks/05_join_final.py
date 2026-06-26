"""
============================================================================
PatagonIA · Construcción del dataset — PASO 5
Integración final -> data/processed/patagonia_dataset.csv
============================================================================

QUÉ HACE
--------
Toma el resultado de los pasos anteriores (fuego + terreno + clima + vegetación +
distancias ya unidos en el paso 4), hace una limpieza final y guarda el DATASET
COMPARTIDO que consumen las dos materias:
  - Minería de Datos: lo carga en KNIME para clustering (K-Means) y reglas (Apriori).
  - IA: lo usa como base para los modelos supervisados.

Cada fila = un hexágono H3 de la Patagonia con actividad de incendio 2012-2023.
Cada columna = una característica ambiental o del régimen de fuego de esa zona.

DECISIONES DE LIMPIEZA
----------------------
- Eliminamos filas con datos faltantes en variables ambientales (muy pocas: zonas
  costeras donde la API meteo no devuelve valor). Documentar el conteo.
- Redondeamos a 2-3 decimales para legibilidad (no afecta el análisis).
- 'mes_pico' y 'cobertura_veg' son categóricas; el resto numéricas.

ENTRADA : data/processed/_intermedios/04_fuego_completo.parquet
SALIDA  : data/processed/patagonia_dataset.csv      (entregable compartido, se versiona)
          data/processed/patagonia_dataset.parquet  (mismo dato, formato eficiente)
============================================================================
"""
from pathlib import Path
import pandas as pd

IN   = Path("data/processed/_intermedios/04_fuego_completo.parquet")
CSV  = Path("data/processed/patagonia_dataset.csv")
PARQ = Path("data/processed/patagonia_dataset.parquet")

df = pd.read_parquet(IN)
print(f"Hexágonos antes de limpiar: {len(df):,}")

# Orden lógico de columnas: identificación, fuego, terreno/clima, vegetación, humano.
columnas = [
    "hex", "lat", "lon",
    # --- régimen de fuego (FIRMS) ---
    "n_focos", "brillo_medio", "brillo_max", "frp_medio", "frp_max",
    "brillo_t31_medio", "pct_noche", "pct_verano", "pct_conf_alta",
    "n_anios_activo", "mes_pico",
    # --- terreno y clima ---
    "elevacion", "temp_media", "precip_anual", "viento_medio", "humedad_relativa",
    # --- vegetación (combustible) ---
    "cobertura_veg",
    # --- presión humana ---
    "dist_asentamiento_km", "dist_ruta_km",
]
df = df[columnas]

# Quitar filas con faltantes en variables ambientales (costa sin dato meteo).
antes = len(df)
df = df.dropna(subset=["elevacion", "temp_media", "precip_anual",
                       "viento_medio", "humedad_relativa"])
print(f"Filas eliminadas por faltantes: {antes - len(df)}")

# Redondeo para legibilidad.
num = df.select_dtypes("number").columns
df[num] = df[num].round(3)

CSV.parent.mkdir(parents=True, exist_ok=True)
df.to_csv(CSV, index=False, encoding="utf-8")
df.to_parquet(PARQ, index=False)

print(f"\n=== DATASET FINAL ===")
print(f"Filas (hexágonos): {len(df):,}   Columnas: {df.shape[1]}")
print(f"CSV para KNIME: {CSV}")
print("\nColumnas:", list(df.columns))
print("\nPrimeras filas:")
print(df.head())
print("\nEstadísticos:")
print(df.describe().round(2).T[["mean", "min", "max"]])
