"""
============================================================================
PatagonIA · Construcción del dataset — PASO 4
Distancias a infraestructura humana (IGN Argentina)
============================================================================

QUÉ HACE
--------
A cada hexágono le agrega la distancia (en km) al asentamiento humano más cercano
y a la ruta nacional más cercana, usando datos del Instituto Geográfico Nacional
(IGN) servidos vía WFS (Web Feature Service, sin cuenta).

POR QUÉ ESTAS VARIABLES
-----------------------
El 95% de los incendios en Patagonia son de origen humano (SNMF). Kitzberger et al.
identifican la DISTANCIA A ASENTAMIENTOS como el segundo predictor estático más
fuerte de incendios en la región: cuanto más cerca de la actividad humana, mayor
probabilidad de ignición. La distancia a rutas captura la accesibilidad/uso del
territorio.

CÓMO SE CALCULA LA DISTANCIA
----------------------------
Las distancias geográficas (en grados lat/lon) no son métricas. Reproyectamos todo
a una proyección Azimutal Equidistante centrada en la Patagonia (-45, -69), donde
las distancias al centro están en metros y son fieles para toda la región. Luego
usamos `sjoin_nearest` de geopandas para hallar, por hexágono, el elemento más
cercano y su distancia.

ENTRADA : data/processed/_intermedios/03_fuego_elev_clima_veg.parquet
SALIDA  : data/processed/_intermedios/04_fuego_completo.parquet
============================================================================
"""
from pathlib import Path
import requests
import pandas as pd
import geopandas as gpd

IN  = Path("data/processed/_intermedios/03_fuego_elev_clima_veg.parquet")
OUT = Path("data/processed/_intermedios/04_fuego_completo.parquet")

# Bounding box Patagonia para pedirle a IGN solo lo de la región (lon,lat).
BBOX = "-76,-56,-62,-38,EPSG:4326"
WFS = "https://wms.ign.gob.ar/geoserver/ows"
# Proyección métrica local (Azimutal Equidistante centrada en Patagonia).
AEQD = "+proj=aeqd +lat_0=-45 +lon_0=-69 +datum=WGS84 +units=m"


def bajar_wfs(capa):
    """Descarga una capa IGN dentro del bbox de Patagonia como GeoDataFrame.

    Pagina de a 5000 features por las dudas (algunos servidores limitan).
    """
    partes, start = [], 0
    while True:
        params = {
            "service": "WFS", "version": "2.0.0", "request": "GetFeature",
            "typeName": capa, "outputFormat": "application/json",
            "srsName": "EPSG:4326", "bbox": BBOX,
            "count": 5000, "startIndex": start,
        }
        gj = requests.get(WFS, params=params, timeout=90).json()
        feats = gj.get("features", [])
        if not feats:
            break
        partes.append(gpd.GeoDataFrame.from_features(feats, crs="EPSG:4326"))
        if len(feats) < 5000:
            break
        start += 5000
    return pd.concat(partes, ignore_index=True) if partes else gpd.GeoDataFrame()


# --- 1. Cargar hexágonos como puntos (centroides) --------------------------
df = pd.read_parquet(IN)
hexes = gpd.GeoDataFrame(
    df, geometry=gpd.points_from_xy(df.lon, df.lat), crs="EPSG:4326"
).to_crs(AEQD)

# --- 2. Asentamientos (puntos) y rutas nacionales (líneas) -----------------
print("Bajando asentamientos IGN...")
asent = bajar_wfs("ign:puntos_de_asentamientos_y_edificios_020101").to_crs(AEQD)
print(f"  {len(asent):,} asentamientos")

print("Bajando rutas nacionales IGN...")
rutas = bajar_wfs("ign:vial_nacional").to_crs(AEQD)
print(f"  {len(rutas):,} tramos de ruta")

# --- 3. Distancia al más cercano (en km) -----------------------------------
# sjoin_nearest agrega la distancia (en la unidad del CRS = metros) al vecino
# más próximo. Lo pasamos a km.
d_as = gpd.sjoin_nearest(hexes[["hex", "geometry"]], asent[["geometry"]],
                         distance_col="d")
df["dist_asentamiento_km"] = (d_as.groupby("hex").d.min() / 1000).reindex(df.hex).values

d_ru = gpd.sjoin_nearest(hexes[["hex", "geometry"]], rutas[["geometry"]],
                         distance_col="d")
df["dist_ruta_km"] = (d_ru.groupby("hex").d.min() / 1000).reindex(df.hex).values

# --- 4. Guardar -------------------------------------------------------------
df.to_parquet(OUT, index=False)
print(f"\nGuardado en: {OUT}")
print(df[["dist_asentamiento_km", "dist_ruta_km"]].describe().round(2).T[["mean", "min", "max"]])
