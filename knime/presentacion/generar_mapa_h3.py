"""
============================================================================
PatagonIA · Figura para la presentación — Mapa de la grilla H3
============================================================================

QUÉ HACE
--------
Genera el mapa de la grilla hexagonal H3 (resolución 5) de la Patagonia,
coloreando cada hexágono según su actividad histórica de fuego (`n_focos`),
sobre una **base geográfica difuminada** (silueta de Argentina y Chile) que da
contexto: se reconoce la costa atlántica, el límite con Chile y la punta sur.
Es la figura de la presentación de defensa (portada y diapositiva del dataset).

POR QUÉ ESTA FIGURA
-------------------
Las diapositivas necesitaban mostrar de un vistazo la unidad de análisis (zona =
hexágono H3). Colorear por `n_focos` comunica el hallazgo central (pocas zonas
concentran el fuego); la base geográfica tenue evita la confusión de "¿qué parte
del país es esto?" y deja claro que es la Patagonia.

DECISIONES DE DISEÑO
--------------------
- **Base geográfica difuminada (Natural Earth):** silueta de Argentina + Chile en
  un navy apenas más claro que el fondo, con la costa en línea tenue. Va DEBAJO de
  los hexágonos para dar contexto sin competir visualmente.
- **Color por `n_focos` en escala logarítmica:** la distribución está muy sesgada
  (media 88, máx 5.406); el log resalta el gradiente real de actividad.
- **Paleta crema → naranja:** coincide con el sistema visual del deck (acento
  naranja = fuego).
- **Fondo navy, sin ejes y recorte a la región:** la figura se integra como
  elemento gráfico de las diapositivas.
- **Corrección de aspecto geográfico:** a la latitud de la Patagonia un grado de
  longitud mide menos que uno de latitud; se ajusta el aspecto por 1/cos(lat).

ENTRADA : data/processed/patagonia_dataset.csv
          Natural Earth 50m (se descarga y cachea en data/static/)
SALIDA  : knime/presentacion/mapa_h3.png
============================================================================
"""
from pathlib import Path

import numpy as np
import pandas as pd
import requests
import geopandas as gpd
from shapely.geometry import Polygon
import h3
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm, LinearSegmentedColormap

CSV = Path("data/processed/patagonia_dataset.csv")
OUT = Path("knime/presentacion/mapa_h3.png")
# Base geográfica cacheada (data/static/ no se versiona; se descarga si falta).
BASE_CACHE = Path("data/static/ne_50m_admin_0_countries.geojson")
BASE_URL = ("https://raw.githubusercontent.com/nvkelso/natural-earth-vector/"
            "master/geojson/ne_50m_admin_0_countries.geojson")

# Paleta del deck.
NAVY = "#0F1E3D"
NAVY_CLARO = "#1B3057"   # silueta de tierra: navy apenas más claro
CREMA = "#F4EFE6"
NARANJA = "#E8843C"

# --- 1. Base geográfica (descarga + cache) -----------------------------------
if not BASE_CACHE.exists():
    print("Descargando límites Natural Earth (una sola vez)...")
    BASE_CACHE.parent.mkdir(parents=True, exist_ok=True)
    BASE_CACHE.write_bytes(requests.get(BASE_URL, timeout=120).content)

mundo = gpd.read_file(BASE_CACHE)
# El campo de nombre varía según la versión de Natural Earth.
col = next(c for c in ("ADMIN", "NAME", "SOVEREIGNT") if c in mundo.columns)
base = mundo[mundo[col].isin(["Argentina", "Chile"])]

# --- 2. Dataset y polígono de cada hexágono ----------------------------------
df = pd.read_csv(CSV)
print(f"Hexágonos a dibujar: {len(df):,}")

# h3.cell_to_boundary devuelve (lat, lon); shapely necesita (lon, lat).
def hex_a_poligono(h):
    return Polygon([(lon, lat) for lat, lon in h3.cell_to_boundary(h)])

gdf = gpd.GeoDataFrame(
    df[["hex", "n_focos"]],
    geometry=[hex_a_poligono(h) for h in df.hex],
    crs="EPSG:4326",
)

# --- 3. Dibujar --------------------------------------------------------------
cmap = LinearSegmentedColormap.from_list(
    "patagonia_fuego", [CREMA, NARANJA, "#9E3B16"]
)

fig, ax = plt.subplots(figsize=(8, 11))
fig.patch.set_facecolor(NAVY)
ax.set_facecolor(NAVY)

# Base geográfica difuminada DEBAJO (silueta + costa tenue).
base.plot(ax=ax, facecolor=NAVY_CLARO, edgecolor=CREMA, linewidth=0.5,
          alpha=0.55, zorder=1)

# Hexágonos ENCIMA, coloreados por actividad de fuego.
gdf.plot(
    column="n_focos",
    cmap=cmap,
    norm=LogNorm(vmin=1, vmax=df.n_focos.max()),
    linewidth=0.15,
    edgecolor=NAVY,
    zorder=2,
    ax=ax,
)

# Ventana geográfica fija: se extiende al oeste para que aparezca Chile y al este
# hasta la costa atlántica, de modo que se lea como un mapa completo del sur del
# continente (no recortado al borde de los hexágonos).
ax.set_xlim(-76, -60)
ax.set_ylim(-56.5, -36.5)

ax.set_aspect(1 / np.cos(np.radians(45)))
ax.axis("off")

OUT.parent.mkdir(parents=True, exist_ok=True)
fig.savefig(OUT, dpi=150, bbox_inches="tight", pad_inches=0.1, facecolor=NAVY)
print(f"Guardado en: {OUT}")
