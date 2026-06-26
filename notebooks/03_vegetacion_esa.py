"""
============================================================================
PatagonIA · Construcción del dataset — PASO 3
Cobertura vegetal (ESA WorldCover 2021) por hexágono
============================================================================

QUÉ HACE
--------
A cada hexágono le asigna su TIPO DE COBERTURA VEGETAL dominante (bosque,
matorral, pastizal, etc.) a partir de ESA WorldCover 2021, un mapa global de
cobertura del suelo a 10 m de resolución. La vegetación es el COMBUSTIBLE del
incendio: no es lo mismo un bosque andino húmedo que una estepa de matorral seco.

CÓMO SE MUESTREA (mode de 7 puntos por hexágono)
------------------------------------------------
Un hexágono de res 5 (~253 km²) contiene millones de píxeles de 10 m. Calcular la
clase exacta mayoritaria sobre todos sería costosísimo. Como aproximación robusta,
muestreamos la clase en 7 puntos del hexágono (su centroide + los centroides de
sus 7 sub-hexágonos de res 6) y nos quedamos con la clase más frecuente (la moda).

POR QUÉ SE LEE POR STREAMING Y NO SE DESCARGA
---------------------------------------------
Los tiles de ESA que cubren la Patagonia pesan ~1,5 GB. Pero son COG
(Cloud-Optimized GeoTIFF): GDAL puede leer SOLO los píxeles de los puntos que nos
interesan vía /vsicurl/ (HTTP range requests), sin bajar el archivo entero.
Leemos unos pocos MB en vez de 1,5 GB. Es reproducible: el script vuelve a leer
los mismos tiles del bucket público de ESA (sin credenciales).

ENTRADA : data/processed/_intermedios/02_fuego_elev_clima.parquet
SALIDA  : data/processed/_intermedios/03_fuego_elev_clima_veg.parquet
============================================================================
"""
import os
from pathlib import Path
import math
from collections import Counter

# Config GDAL para streaming eficiente de COG (antes de importar rasterio)
os.environ["GDAL_DISABLE_READDIR_ON_OPEN"] = "EMPTY_DIR"
os.environ["CPL_VSIL_CURL_ALLOWED_EXTENSIONS"] = ".tif"

import pandas as pd
import h3
import rasterio

IN  = Path("data/processed/_intermedios/02_fuego_elev_clima.parquet")
OUT = Path("data/processed/_intermedios/03_fuego_elev_clima_veg.parquet")

# Códigos de clase de ESA WorldCover -> etiqueta legible (en español).
CLASES = {
    10: "bosque", 20: "matorral", 30: "pastizal", 40: "cultivo",
    50: "urbano", 60: "suelo_desnudo", 70: "nieve_hielo", 80: "agua",
    90: "humedal", 95: "manglar", 100: "musgo_liquen",
}
BASE = ("https://esa-worldcover.s3.eu-central-1.amazonaws.com/"
        "v200/2021/map/ESA_WorldCover_10m_2021_v200_{tile}_Map.tif")


def tile_de(lat, lon):
    """Nombre del tile 3°x3° de ESA que contiene la coordenada (esquina SO)."""
    sla = math.floor(lat / 3) * 3
    slo = math.floor(lon / 3) * 3
    return f"S{abs(sla):02d}W{abs(slo):03d}"


df = pd.read_parquet(IN)
print(f"Hexágonos: {len(df):,}")

# Para cada hexágono: 7 puntos de muestreo (centroide + sub-hexágonos res 6).
puntos_por_hex = {}
for hx in df.hex:
    hijos = h3.cell_to_children(hx, 6)            # 7 sub-hexágonos
    puntos_por_hex[hx] = [h3.cell_to_latlng(c) for c in hijos]

# Agrupar TODOS los puntos por tile, para abrir cada tile una sola vez.
df["tile"] = [tile_de(la, lo) for la, lo in zip(df.lat, df.lon)]
clase_hex = {}   # hex -> etiqueta de cobertura dominante

for tile in sorted(df.tile.unique()):
    hexes = df.loc[df.tile == tile, "hex"].tolist()
    # (lon, lat) de todos los puntos de los hexágonos de este tile
    coords, idx = [], []
    for hx in hexes:
        for (la, lo) in puntos_por_hex[hx]:
            coords.append((lo, la)); idx.append(hx)
    url = "/vsicurl/" + BASE.format(tile=tile)
    with rasterio.open(url) as src:
        vals = [v[0] for v in src.sample(coords)]   # lee solo esos píxeles
    # moda de la clase por hexágono
    por_hex = {}
    for hx, v in zip(idx, vals):
        por_hex.setdefault(hx, []).append(int(v))
    for hx, vs in por_hex.items():
        cod = Counter(vs).most_common(1)[0][0]
        clase_hex[hx] = CLASES.get(cod, f"otro_{cod}")
    print(f"  tile {tile}: {len(hexes)} hexágonos muestreados")

df["cobertura_veg"] = df.hex.map(clase_hex)
df = df.drop(columns="tile")

df.to_parquet(OUT, index=False)
print(f"\nGuardado en: {OUT}")
print("Distribución de cobertura vegetal:")
print(df.cobertura_veg.value_counts())
