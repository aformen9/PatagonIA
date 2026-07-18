# -*- coding: utf-8 -*-
"""Carga y preparación del dataset de riesgo de incendios PatagonIA.

Funciones para leer el dataset generado por el pipeline 01-05, aplicar el
filtro de flaring (zona Vaca Muerta) y construir la matriz de features para
los modelos del TP1.
"""
from __future__ import annotations

import pandas as pd

# Columnas predictoras usadas en el TP1. `cobertura_veg` es categórica y se
# entrega sin encodear (el encoding se decide en el notebook / pipeline de modelo).
FEATURES_NUM = [
    "elevacion",
    "temp_media",
    "precip_anual",
    "viento_medio",
    "humedad_relativa",
    "dist_asentamiento_km",
    "dist_ruta_km",
]
FEATURE_CAT = "cobertura_veg"
FEATURES = FEATURES_NUM + [FEATURE_CAT]

TARGET_CONT = "n_focos"
UMBRAL_RIESGO = 150


def cargar_dataset(path: str) -> pd.DataFrame:
    """Carga el dataset de hexágonos H3 desde un CSV.

    Args:
        path: Ruta al archivo CSV (típicamente
            ``data/processed/patagonia_ia_con_nulos.csv``).

    Returns:
        DataFrame con una fila por hexágono H3 y las 22 columnas del pipeline.
    """
    return pd.read_csv(path)


def filtrar_flaring(
    df: pd.DataFrame, lat_max: float = -39.5, lon_min: float = -70.0
) -> pd.DataFrame:
    """Excluye los hexágonos afectados por quema de gas (flaring), no incendios.

    Descarta las filas que cumplen simultáneamente ``lat > lat_max`` y
    ``lon > lon_min``. Ese cuadrante contiene 561 hexágonos que concentran
    actividad térmica persistente con una firma inconsistente con el régimen
    de incendios (alta ``pct_noche``, muy baja ``pct_conf_alta``, muchos años
    activos y ``n_focos`` extremos): corresponde a las antorchas de gas de la
    formación Vaca Muerta. Mantenerlos contamina la señal de riesgo de
    incendios que el modelo debe aprender.

    Args:
        df: DataFrame con columnas ``lat`` y ``lon``.
        lat_max: Latitud por encima de la cual (más al norte) se excluye.
        lon_min: Longitud por encima de la cual (más al este) se excluye.

    Returns:
        Copia del DataFrame sin los hexágonos del cuadrante de flaring.
    """
    bbox_flaring = (df["lat"] > lat_max) & (df["lon"] > lon_min)
    return df.loc[~bbox_flaring].copy()


def preparar_features(
    df: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.Series, pd.Series]:
    """Separa la matriz de features y los dos targets (continuo y binario).

    ``cobertura_veg`` se devuelve como columna categórica sin encodear.

    Args:
        df: DataFrame ya cargado (y normalmente filtrado por flaring).

    Returns:
        Tupla ``(X, y_cont, y_bin)`` donde:
            - ``X``: features (7 numéricas + ``cobertura_veg`` sin encodear).
            - ``y_cont``: ``n_focos`` (regresión).
            - ``y_bin``: ``(n_focos > 150)`` como entero (clasificación).
    """
    X = df[FEATURES].copy()
    y_cont = df[TARGET_CONT].copy()
    y_bin = (df[TARGET_CONT] > UMBRAL_RIESGO).astype(int)
    y_bin.name = "riesgo_alto"
    return X, y_cont, y_bin
