# -*- coding: utf-8 -*-
"""
PASO 2 — Re-ejecuta el pipeline 01->05 sobre el crudo FIRMS 775078,
SIN el dropna de 05_join_final.py:58. NO edita los scripts del repo.

Cada paso es resumible: si su intermedio ya existe, se salta (las APIs son lentas).
Salida final: data/processed/patagonia_ia_con_nulos.csv  (+ .parquet)
"""
import os, time, math, sys
from pathlib import Path
from collections import Counter

os.environ["GDAL_DISABLE_READDIR_ON_OPEN"] = "EMPTY_DIR"
os.environ["CPL_VSIL_CURL_ALLOWED_EXTENSIONS"] = ".tif"

import pandas as pd
import h3

RAW = Path("data/raw/firms/fire_archive_SV-C2_775078.csv")
INT = Path("data/processed/_intermedios")
INT.mkdir(parents=True, exist_ok=True)
P01 = INT / "01_fuego.parquet"
P02 = INT / "02_fuego_elev_clima.parquet"
P03 = INT / "03_fuego_elev_clima_veg.parquet"
P04 = INT / "04_fuego_completo.parquet"

RES_H3 = 5
LAT_MIN, LAT_MAX = -56, -38
LON_MIN, LON_MAX = -76, -62


def log(msg): print(msg, flush=True)


# ===========================================================================
# PASO 01 — FIRMS -> H3 (idéntico a 01_firms_a_h3.py salvo el RAW)
# ===========================================================================
def paso01():
    if P01.exists():
        log(f"[01] existe, salto: {P01}"); return
    df = pd.read_csv(RAW)
    log(f"[01] Focos crudos: {len(df):,}")
    df = df[(df.latitude < LAT_MAX) & (df.latitude > LAT_MIN) &
            (df.longitude > LON_MIN) & (df.longitude < LON_MAX)].copy()
    log(f"[01] Focos en Patagonia: {len(df):,}")
    df["hex"] = [h3.latlng_to_cell(la, lo, RES_H3) for la, lo in zip(df.latitude, df.longitude)]
    df["acq_date"] = pd.to_datetime(df["acq_date"])
    df["year"] = df.acq_date.dt.year
    df["month"] = df.acq_date.dt.month
    df["is_summer"] = df.month.isin([12, 1, 2]).astype(int)
    df["is_night"] = (df.daynight == "N").astype(int)
    df["conf_high"] = (df.confidence == "h").astype(int)
    g = df.groupby("hex")
    agg = pd.DataFrame({
        "n_focos": g.size(),
        "brillo_medio": g.brightness.mean(),
        "brillo_max": g.brightness.max(),
        "frp_medio": g.frp.mean(),
        "frp_max": g.frp.max(),
        "brillo_t31_medio": g.bright_t31.mean(),
        "pct_noche": g.is_night.mean(),
        "pct_verano": g.is_summer.mean(),
        "pct_conf_alta": g.conf_high.mean(),
        "n_anios_activo": g.year.nunique(),
        "mes_pico": g.month.agg(lambda s: s.mode().iloc[0]),
    }).reset_index()
    cent = [h3.cell_to_latlng(hx) for hx in agg.hex]
    agg["lat"] = [c[0] for c in cent]
    agg["lon"] = [c[1] for c in cent]
    agg.to_parquet(P01, index=False)
    log(f"[01] Hexágonos con fuego: {len(agg):,} -> {P01}")


# ===========================================================================
# PASO 02 — elevación + clima (idéntico a 02_elevacion_clima.py)
# ===========================================================================
def paso02():
    if P02.exists():
        log(f"[02] existe, salto: {P02}"); return
    import json
    import requests
    df = pd.read_parquet(P01)
    n = len(df)
    log(f"[02] Hexágonos a enriquecer: {n:,}")
    sess = requests.Session()

    # Cachés granulares: un corte pierde una celda, no un lote entero.
    CACHE_CLIMA = INT / "cache_clima"   # {celda_res4}.json
    CACHE_ELEV = INT / "cache_elev"     # {hex}.json
    CACHE_CLIMA.mkdir(parents=True, exist_ok=True)
    CACHE_ELEV.mkdir(parents=True, exist_ok=True)

    def get_json(url, params=None, n_intentos=6):
        espera = 2
        for intento in range(n_intentos):
            try:
                r = sess.get(url, params=params, timeout=60)
                if r.status_code == 200:
                    return r.json()
                log(f"    [intento {intento+1}] HTTP {r.status_code}: {r.text[:100]}")
            except Exception as e:
                log(f"    [intento {intento+1}] excepción: {e}")
            time.sleep(espera); espera = min(espera * 2, 60)
        raise RuntimeError(f"API no respondió: {url}")

    # elevación — cacheada por hexágono; sólo se piden los que faltan
    OTOPO = "https://api.opentopodata.org/v1/srtm30m"
    faltan = df[~df.hex.map(lambda h: (CACHE_ELEV / f"{h}.json").exists())]
    log(f"[02] Elevación cacheada: {n - len(faltan):,}/{n:,}; faltan {len(faltan):,}")
    for i in range(0, len(faltan), 100):
        lote = faltan.iloc[i:i+100]
        loc = "|".join(f"{la:.5f},{lo:.5f}" for la, lo in zip(lote.lat, lote.lon))
        j = get_json(OTOPO, {"locations": loc})
        for hx, r in zip(lote.hex, j["results"]):
            (CACHE_ELEV / f"{hx}.json").write_text(
                json.dumps({"elevacion": r["elevation"]}), encoding="utf-8")
        time.sleep(1.1)
        if (i // 100) % 5 == 0: log(f"  elevación {i}/{len(faltan)}")
    elev = [json.loads((CACHE_ELEV / f"{hx}.json").read_text(encoding="utf-8"))["elevacion"]
            for hx in df.hex]
    df["elevacion"] = elev
    log(f"[02] Elevación lista (rango {min(x for x in elev if x is not None):.0f}"
        f"–{max(x for x in elev if x is not None):.0f} m)")

    # clima por celda madre res 4 — cacheado por celda madre
    df["hex_madre"] = [h3.cell_to_parent(hx, 4) for hx in df.hex]
    madres = df.hex_madre.unique()
    log(f"[02] Celdas madre (res4) para clima: {len(madres):,}")
    POWER = "https://power.larc.nasa.gov/api/temporal/climatology/point"
    BASE = {"parameters": "T2M,PRECTOTCORR,WS10M,RH2M", "community": "AG", "format": "JSON"}
    clima = {}
    for j, m in enumerate(madres):
        cache_f = CACHE_CLIMA / f"{m}.json"
        if cache_f.exists():
            clima[m] = json.loads(cache_f.read_text(encoding="utf-8"))
            if j % 50 == 0: log(f"  clima {j}/{len(madres)} (cache)")
            continue
        la, lo = h3.cell_to_latlng(m)
        p = {**BASE, "latitude": f"{la:.5f}", "longitude": f"{lo:.5f}"}
        data = get_json(POWER, p)["properties"]["parameter"]
        clima[m] = {
            "temp_media": data["T2M"]["ANN"],
            "precip_anual": data["PRECTOTCORR"]["ANN"] * 365,
            "viento_medio": data["WS10M"]["ANN"],
            "humedad_relativa": data["RH2M"]["ANN"],
        }
        cache_f.write_text(json.dumps(clima[m]), encoding="utf-8")
        time.sleep(0.3)
        if j % 50 == 0: log(f"  clima {j}/{len(madres)}")
    clima_df = pd.DataFrame(clima).T
    df = df.join(clima_df, on="hex_madre").drop(columns="hex_madre")
    # NASA POWER usa -999 como fill: convertir a NaN (para que queden como nulos)
    for c in ["temp_media", "precip_anual", "viento_medio", "humedad_relativa"]:
        df.loc[df[c] <= -900, c] = pd.NA
    df.to_parquet(P02, index=False)
    log(f"[02] -> {P02}")
    log("[02] Nulos ambientales: " + str(df[["elevacion","temp_media","precip_anual",
        "viento_medio","humedad_relativa"]].isna().sum().to_dict()))


# ===========================================================================
# PASO 03 — vegetación ESA (idéntico a 03_vegetacion_esa.py)
# ===========================================================================
def paso03():
    if P03.exists():
        log(f"[03] existe, salto: {P03}"); return
    import json
    import rasterio
    CLASES = {10:"bosque",20:"matorral",30:"pastizal",40:"cultivo",50:"urbano",
              60:"suelo_desnudo",70:"nieve_hielo",80:"agua",90:"humedal",95:"manglar",100:"musgo_liquen"}
    BASE = ("https://esa-worldcover.s3.eu-central-1.amazonaws.com/"
            "v200/2021/map/ESA_WorldCover_10m_2021_v200_{tile}_Map.tif")
    CACHE_ESA = INT / "cache_esa"   # {hex}.json  (un corte pierde un tile, no todo)
    CACHE_ESA.mkdir(parents=True, exist_ok=True)

    def tile_de(lat, lon):
        sla = math.floor(lat/3)*3; slo = math.floor(lon/3)*3
        return f"S{abs(sla):02d}W{abs(slo):03d}"

    df = pd.read_parquet(P02)
    log(f"[03] Hexágonos: {len(df):,}")
    df["tile"] = [tile_de(la, lo) for la, lo in zip(df.lat, df.lon)]
    # sólo se muestrean los hexágonos sin caché
    cacheados = df.hex.map(lambda h: (CACHE_ESA / f"{h}.json").exists())
    faltan = df[~cacheados]
    log(f"[03] ESA cacheada: {int(cacheados.sum()):,}/{len(df):,}; faltan {len(faltan):,}")
    puntos = {hx: [h3.cell_to_latlng(c) for c in h3.cell_to_children(hx, 6)] for hx in faltan.hex}
    for tile in sorted(faltan.tile.unique()):
        hexes = faltan.loc[faltan.tile == tile, "hex"].tolist()
        coords, idx = [], []
        for hx in hexes:
            for (la, lo) in puntos[hx]:
                coords.append((lo, la)); idx.append(hx)
        url = "/vsicurl/" + BASE.format(tile=tile)
        try:
            with rasterio.open(url) as src:
                vals = [v[0] for v in src.sample(coords)]
        except Exception as e:
            log(f"  tile {tile}: ERROR {e} -> nulos (no se cachea, se reintenta)")
            continue
        por_hex = {}
        for hx, v in zip(idx, vals): por_hex.setdefault(hx, []).append(int(v))
        for hx, vs in por_hex.items():
            cod = Counter(vs).most_common(1)[0][0]
            (CACHE_ESA / f"{hx}.json").write_text(
                json.dumps({"cobertura_veg": CLASES.get(cod, f"otro_{cod}")}), encoding="utf-8")
        log(f"  tile {tile}: {len(hexes)} hex")
    def leer_esa(h):
        f = CACHE_ESA / f"{h}.json"
        return json.loads(f.read_text(encoding="utf-8"))["cobertura_veg"] if f.exists() else None
    df["cobertura_veg"] = df.hex.map(leer_esa)
    df = df.drop(columns="tile")
    df.to_parquet(P03, index=False)
    log(f"[03] -> {P03}")
    log("[03] Cobertura:\n" + df.cobertura_veg.value_counts(dropna=False).to_string())


# ===========================================================================
# PASO 04 — distancias IGN (idéntico a 04_distancias_ign.py)
# ===========================================================================
def paso04():
    if P04.exists():
        log(f"[04] existe, salto: {P04}"); return
    import requests
    import geopandas as gpd
    BBOX = "-76,-56,-62,-38,EPSG:4326"
    WFS = "https://wms.ign.gob.ar/geoserver/ows"
    AEQD = "+proj=aeqd +lat_0=-45 +lon_0=-69 +datum=WGS84 +units=m"

    def bajar_wfs(capa):
        partes, start = [], 0
        while True:
            params = {"service":"WFS","version":"2.0.0","request":"GetFeature",
                      "typeName":capa,"outputFormat":"application/json","srsName":"EPSG:4326",
                      "bbox":BBOX,"count":5000,"startIndex":start}
            gj = requests.get(WFS, params=params, timeout=120).json()
            feats = gj.get("features", [])
            if not feats: break
            partes.append(gpd.GeoDataFrame.from_features(feats, crs="EPSG:4326"))
            if len(feats) < 5000: break
            start += 5000
        return pd.concat(partes, ignore_index=True) if partes else gpd.GeoDataFrame()

    import json
    CACHE_IGN = INT / "cache_ign"   # {hex}.json  (evita re-bajar el WFS al reanudar)
    CACHE_IGN.mkdir(parents=True, exist_ok=True)

    df = pd.read_parquet(P03)
    faltan = df[~df.hex.map(lambda h: (CACHE_IGN / f"{h}.json").exists())]
    log(f"[04] IGN cacheada: {len(df)-len(faltan):,}/{len(df):,}; faltan {len(faltan):,}")
    if len(faltan):
        hexes = gpd.GeoDataFrame(faltan, geometry=gpd.points_from_xy(faltan.lon, faltan.lat),
                                 crs="EPSG:4326").to_crs(AEQD)
        log("[04] Bajando asentamientos IGN...")
        asent = bajar_wfs("ign:puntos_de_asentamientos_y_edificios_020101").to_crs(AEQD)
        log(f"  {len(asent):,} asentamientos")
        log("[04] Bajando rutas nacionales IGN...")
        rutas = bajar_wfs("ign:vial_nacional").to_crs(AEQD)
        log(f"  {len(rutas):,} tramos de ruta")
        d_as = gpd.sjoin_nearest(hexes[["hex","geometry"]], asent[["geometry"]], distance_col="d")
        d_as = d_as.groupby("hex").d.min() / 1000
        d_ru = gpd.sjoin_nearest(hexes[["hex","geometry"]], rutas[["geometry"]], distance_col="d")
        d_ru = d_ru.groupby("hex").d.min() / 1000
        for hx in faltan.hex:
            va, vr = d_as.get(hx), d_ru.get(hx)
            (CACHE_IGN / f"{hx}.json").write_text(json.dumps({
                "dist_asentamiento_km": float(va) if va is not None else None,
                "dist_ruta_km": float(vr) if vr is not None else None,
            }), encoding="utf-8")
    def leer_ign(h, k):
        return json.loads((CACHE_IGN / f"{h}.json").read_text(encoding="utf-8"))[k]
    df["dist_asentamiento_km"] = pd.to_numeric(df.hex.map(lambda h: leer_ign(h, "dist_asentamiento_km")), errors="coerce")
    df["dist_ruta_km"] = pd.to_numeric(df.hex.map(lambda h: leer_ign(h, "dist_ruta_km")), errors="coerce")
    df.to_parquet(P04, index=False)
    log(f"[04] -> {P04}")


# ===========================================================================
# PASO 05 — join final SIN dropna (05_join_final.py sin la línea 58)
# ===========================================================================
def paso05():
    COLS = ["hex","lat","lon","n_focos","brillo_medio","brillo_max","frp_medio","frp_max",
            "brillo_t31_medio","pct_noche","pct_verano","pct_conf_alta","n_anios_activo","mes_pico",
            "elevacion","temp_media","precip_anual","viento_medio","humedad_relativa",
            "cobertura_veg","dist_asentamiento_km","dist_ruta_km"]
    df = pd.read_parquet(P04)
    log(f"[05] Hexágonos (SIN dropna): {len(df):,}")
    df = df[COLS]
    ENV = ["elevacion","temp_media","precip_anual","viento_medio","humedad_relativa"]
    n_drop = int(df[ENV].isna().any(axis=1).sum())
    log(f"[05] Filas que el dropna original habría eliminado: {n_drop}")
    log(f"[05] Shape CON dropna sería: {len(df)-n_drop} x {df.shape[1]}")
    num = df.select_dtypes("number").columns
    # cast a float para poder redondear con NaN presentes
    df[num] = df[num].astype("float64").round(3)
    CSV = Path("data/processed/patagonia_ia_con_nulos.csv")
    PARQ = Path("data/processed/patagonia_ia_con_nulos.parquet")
    df.to_csv(CSV, index=False, encoding="utf-8")
    df.to_parquet(PARQ, index=False)
    log(f"[05] -> {CSV}  (shape {df.shape})")


if __name__ == "__main__":
    steps = sys.argv[1:] or ["01", "02", "03", "04", "05"]
    if "01" in steps: paso01()
    if "02" in steps: paso02()
    if "03" in steps: paso03()
    if "04" in steps: paso04()
    if "05" in steps: paso05()
    log("=== RUNNER FIN ===")
