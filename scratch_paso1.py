# -*- coding: utf-8 -*-
"""PASO 1 — Verificación del crudo FIRMS 775078 + comparación contra Minería."""
import sys
from pathlib import Path
import pandas as pd
import h3

pd.set_option("display.width", 160)
pd.set_option("display.max_columns", 40)

RAW = Path("data/raw/firms/fire_archive_SV-C2_775078.csv")
RES_H3 = 5
LAT_MIN, LAT_MAX = -56, -38
LON_MIN, LON_MAX = -76, -62

print("=" * 70)
print("PASO 1 — VERIFICACIÓN DEL CRUDO")
print("=" * 70)

df = pd.read_csv(RAW)
print(f"\n[1] Shape crudo: {df.shape[0]:,} filas x {df.shape[1]} columnas")
print(f"\n[2] Columnas: {list(df.columns)}")
print("\n[3] Dtypes:")
print(df.dtypes.to_string())

# Rango de fechas
d = pd.to_datetime(df["acq_date"])
print(f"\n[4] Rango acq_date: {d.min().date()}  ->  {d.max().date()}")
print(f"    Años presentes: {sorted(d.dt.year.unique().tolist())}")

# Confidence
print("\n[5] Distribución confidence:")
print(df["confidence"].value_counts(dropna=False).to_string())
print("    (proporción):")
print((df["confidence"].value_counts(normalize=True, dropna=False) * 100).round(2).to_string())

# Daynight
print("\n[6] Distribución daynight:")
print(df["daynight"].value_counts(dropna=False).to_string())
print((df["daynight"].value_counts(normalize=True, dropna=False) * 100).round(2).to_string())

# Nulos
print("\n[7] Nulos por columna:")
nul = pd.DataFrame({"n_nulos": df.isna().sum(), "pct": (100 * df.isna().mean()).round(3)})
print(nul.to_string())

# ------------------------------------------------------------------
# COMPARACIÓN vs MINERÍA — replica exacta del groupby de 01_firms_a_h3.py
# ------------------------------------------------------------------
print("\n" + "=" * 70)
print("COMPARACIÓN CONTRA MINERÍA (agregación 01 sobre este crudo)")
print("=" * 70)

f = df[(df.latitude < LAT_MAX) & (df.latitude > LAT_MIN) &
       (df.longitude > LON_MIN) & (df.longitude < LON_MAX)].copy()
print(f"\nFocos crudos (todo el archivo): {len(df):,}")
print(f"Focos dentro del bbox Patagonia: {len(f):,}")
print(f"Focos fuera del bbox (descartados): {len(df) - len(f):,}")

f["hex"] = [h3.latlng_to_cell(la, lo, RES_H3) for la, lo in zip(f.latitude, f.longitude)]
g = f.groupby("hex")
agg = pd.DataFrame({"n_focos": g.size()}).reset_index()

n_hex = len(agg)
suma_focos = int(agg.n_focos.sum())

MIN_HEX = 1980
MIN_FOCOS = 174371

print(f"\n--- RESULTADO NUEVO CRUDO (775078) ---")
print(f"Hexágonos con foco : {n_hex:,}")
print(f"Suma total n_focos : {suma_focos:,}")

print(f"\n--- REFERENCIA MINERÍA (dataset 767267 en repo) ---")
print(f"Hexágonos con foco : {MIN_HEX:,}")
print(f"Suma total n_focos : {MIN_FOCOS:,}")

print(f"\n--- DIFERENCIA (nuevo - minería) ---")
print(f"Δ hexágonos : {n_hex - MIN_HEX:+,}  ({100*(n_hex-MIN_HEX)/MIN_HEX:+.2f}%)")
print(f"Δ n_focos   : {suma_focos - MIN_FOCOS:+,}  ({100*(suma_focos-MIN_FOCOS)/MIN_FOCOS:+.2f}%)")

# Cross-check contra el dataset final versionado
FINAL = Path("data/processed/patagonia_dataset.csv")
if FINAL.exists():
    dsf = pd.read_csv(FINAL)
    print(f"\n--- CROSS-CHECK contra data/processed/patagonia_dataset.csv ---")
    print(f"Filas (hexágonos) dataset final : {len(dsf):,}")
    print(f"Suma n_focos dataset final       : {int(dsf.n_focos.sum()):,}")
    # overlap de hexágonos
    inter = set(agg.hex) & set(dsf.hex)
    print(f"Hexágonos en común (nuevo ∩ final): {len(inter):,}")
    print(f"Solo en nuevo crudo               : {len(set(agg.hex) - set(dsf.hex)):,}")
    print(f"Solo en dataset final             : {len(set(dsf.hex) - set(agg.hex)):,}")

print("\n[Fin Paso 1 — NO se corrige ninguna diferencia, solo se reporta.]")
