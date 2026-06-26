"""
============================================================================
PatagonIA · Construcción del dataset — PASO 2
Elevación + clima por hexágono
============================================================================

QUÉ HACE
--------
A cada hexágono del paso 1 le agrega variables del TERRENO y del CLIMA, que son
drivers ambientales del riesgo de incendio (Kitzberger et al.):
  - elevación (m)           -> el terreno modula temperatura y tipo de vegetación
  - temperatura media (°C)  -> predictor dominante de incendios en Patagonia
  - precipitación anual (mm)-> a menor precipitación, más combustible seco
  - viento medio (m/s)      -> propaga el fuego
  - humedad relativa (%)    -> proxy de aridez: menor HR = ambiente más seco y
                               propenso a ignición

FUENTES DE DATOS
----------------
  Elevación: OpenTopoData SRTM 30m (https://api.opentopodata.org/v1/srtm30m),
             sin cuenta, hasta 100 puntos por llamada (pipe-separated).
             Dato base: NASA SRTM v3 (~30 m de resolución horizontal).
  Clima: NASA POWER Climatology API (https://power.larc.nasa.gov/api/temporal/
         climatology/point), datos 1981-2020. Sin cuenta, sin límite horario.
         Variables: T2M, PRECTOTCORR, WS10M, RH2M (comunidad AG = agricultura).

POR QUÉ OPENTOPODATA Y NO OPEN-METEO ELEVATION
----------------------------------------------
Open-Meteo tiene cuota horaria en el nivel gratuito (tanto el endpoint de
elevación como el de archivo histórico). OpenTopoData SRTM no tiene cuota horaria
estricta y sirve directamente datos NASA SRTM v3, que es la fuente de referencia
estándar para elevación en estudios de incendios (mayor resolución que SRTM 90m
del CGIAR y sin requisito de cuenta). Límite: 1 request/segundo → 20 lotes para
nuestros 1.980 hexágonos = ~30 segundos en total.

POR QUÉ NASA POWER Y NO ERA5/OPEN-METEO-ARCHIVE
------------------------------------------------
Open-Meteo Archive tiene cuota horaria en el nivel gratuito, lo que haría fallar
el pipeline si se superan ~500 llamadas rápidas. NASA POWER Climatology no tiene
esa restricción: devuelve los promedios 1981-2020 en una sola llamada por punto,
sin necesidad de pedir ni procesar 10 años de datos diarios. La fuente es la misma
(reanálisis ERA5/MERRA-2) y es citable: Stackhouse et al. 2019, NASA.

POR QUÉ EL CLIMA SE MUESTREA A RESOLUCIÓN 4 (celda "madre")
-----------------------------------------------------------
Los datos meteorológicos tienen resolución nativa ~28 km. Pedir clima para cada
hexágono de res 5 (~9 km de lado) sería PSEUDO-REPLICACIÓN: estaríamos copiando el
mismo dato de ~28 km en varios hexágonos vecinos con ruido. Por eso agrupamos los
hexágonos en su celda madre H3 de res 4 (~28 km, alineada con la resolución meteo),
pedimos el clima una vez por celda madre y se lo asignamos a sus hexágonos hijos.
Esto reduce las llamadas a la API ~4x y es metodológicamente correcto.
La elevación SÍ se pide por hexágono porque el terreno cambia rápido (montañas).

ENTRADA : data/processed/_intermedios/01_fuego.parquet
SALIDA  : data/processed/_intermedios/02_fuego_elev_clima.parquet
============================================================================
"""
from pathlib import Path
import time
import requests
import pandas as pd
import h3

IN  = Path("data/processed/_intermedios/01_fuego.parquet")
OUT = Path("data/processed/_intermedios/02_fuego_elev_clima.parquet")

df = pd.read_parquet(IN)
n = len(df)
print(f"Hexágonos a enriquecer: {n:,}")

sess = requests.Session()


def get_json(url, params=None, n_intentos=6):
    """GET robusto con backoff exponencial (2s → 4s → 8s ... 60s).

    Loguea el código de respuesta en cada error para facilitar el diagnóstico.
    """
    espera = 2
    for intento in range(n_intentos):
        try:
            r = sess.get(url, params=params, timeout=60)
            if r.status_code == 200:
                return r.json()
            print(f"    [intento {intento+1}] HTTP {r.status_code}: {r.text[:120]}")
        except Exception as e:
            print(f"    [intento {intento+1}] excepción: {e}")
        time.sleep(espera)
        espera = min(espera * 2, 60)
    raise RuntimeError(f"API no respondió tras {n_intentos} intentos: {url}")

# ===========================================================================
# 1. ELEVACIÓN — por hexágono (terreno cambia rápido, no se puede agrupar)
# ===========================================================================
# OpenTopoData acepta hasta 100 puntos por llamada, separados por '|'.
# Respeta 1 req/seg para no exceder el límite de la API pública.
OTOPO_URL = "https://api.opentopodata.org/v1/srtm30m"
elev = []
for i in range(0, n, 100):
    lote = df.iloc[i:i+100]
    locations = "|".join(f"{la:.5f},{lo:.5f}"
                         for la, lo in zip(lote.lat, lote.lon))
    j = get_json(OTOPO_URL, {"locations": locations})
    elev.extend(r["elevation"] for r in j["results"])
    time.sleep(1.1)   # OpenTopoData: máx 1 req/seg en el tier gratuito
    if (i // 100) % 5 == 0:
        print(f"  elevación {i}/{n}")
df["elevacion"] = elev
print(f"Elevación lista (rango {min(elev):.0f}–{max(elev):.0f} m)")

# ===========================================================================
# 2. CLIMA — NASA POWER Climatology, una llamada por celda madre (res 4)
# ===========================================================================
# Agrupamos en celdas H3 res 4 (~28 km), alineadas con la resolución meteo.
df["hex_madre"] = [h3.cell_to_parent(hx, 4) for hx in df.hex]
madres = df.hex_madre.unique()
print(f"Celdas madre (res 4) para clima: {len(madres):,}  "
      f"(en vez de {n:,} llamadas)")

# Variables pedidas a NASA POWER:
#   T2M          -> temperatura media anual (°C)
#   PRECTOTCORR  -> precipitación corregida (mm/día, ANN = promedio diario anual)
#   WS10M        -> velocidad del viento a 10m (m/s)
#   RH2M         -> humedad relativa a 2m (%)
POWER_URL = "https://power.larc.nasa.gov/api/temporal/climatology/point"
PARAMS_BASE = {
    "parameters": "T2M,PRECTOTCORR,WS10M,RH2M",
    "community": "AG",     # comunidad agrícola: clima de superficie
    "format": "JSON",
}

clima = {}   # hex_madre -> dict con variables climáticas
for j, m in enumerate(madres):
    la, lo = h3.cell_to_latlng(m)
    params = {**PARAMS_BASE, "latitude": f"{la:.5f}", "longitude": f"{lo:.5f}"}
    data = get_json(POWER_URL, params)["properties"]["parameter"]
    clima[m] = {
        # ANN = promedio anual provisto directamente por NASA POWER
        "temp_media":        data["T2M"]["ANN"],
        "precip_anual":      data["PRECTOTCORR"]["ANN"] * 365,  # mm/día → mm/año
        "viento_medio":      data["WS10M"]["ANN"],
        "humedad_relativa":  data["RH2M"]["ANN"],
    }
    time.sleep(0.3)    # espaciado mínimo (NASA POWER no tiene cuota horaria estricta)
    if j % 50 == 0:
        print(f"  clima {j}/{len(madres)}")

# Asignar a cada hexágono el clima de su celda madre.
clima_df = pd.DataFrame(clima).T
df = df.join(clima_df, on="hex_madre").drop(columns="hex_madre")

df.to_parquet(OUT, index=False)
print(f"\nGuardado en: {OUT}")
print("Nulos por variable:",
      df[["elevacion", "temp_media", "precip_anual",
          "viento_medio", "humedad_relativa"]].isna().sum().to_dict())
print(df[["elevacion", "temp_media", "precip_anual",
          "viento_medio", "humedad_relativa"]]
      .describe().round(2).T[["mean", "min", "max"]])
