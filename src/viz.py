# -*- coding: utf-8 -*-
"""Gráficos reutilizables para el EDA del TP1.

Todas las funciones trabajan sobre matplotlib y aceptan (o crean) un ``Axes``,
de modo que puedan componerse en grillas de subplots dentro del notebook.
"""
from __future__ import annotations

from typing import Optional

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

# Etiquetas legibles (nombre + unidad) para rotular ejes y títulos.
ETIQUETAS = {
    "n_focos": "N.º de focos",
    "elevacion": "Elevación (m)",
    "temp_media": "Temp. media anual (°C)",
    "precip_anual": "Precipitación anual (mm)",
    "viento_medio": "Viento medio (m/s)",
    "humedad_relativa": "Humedad relativa (%)",
    "dist_asentamiento_km": "Dist. a asentamiento (km)",
    "dist_ruta_km": "Dist. a ruta (km)",
    "cobertura_veg": "Cobertura vegetal",
}


def _label(col: str) -> str:
    """Devuelve la etiqueta legible de una columna, o su nombre crudo."""
    return ETIQUETAS.get(col, col)


def plot_distribucion(
    df: pd.DataFrame, col: str, ax: Optional[plt.Axes] = None
) -> plt.Axes:
    """Dibuja un histograma de una variable numérica.

    Args:
        df: DataFrame con los datos.
        col: Nombre de la columna numérica a graficar.
        ax: Eje donde dibujar. Si es ``None`` se crea uno nuevo.

    Returns:
        El ``Axes`` con el histograma dibujado.
    """
    if ax is None:
        _, ax = plt.subplots(figsize=(6, 4))
    serie = df[col].dropna()
    ax.hist(serie, bins=40, color="#3b6ea5", edgecolor="white", alpha=0.85)
    ax.set_title(f"Distribución de {_label(col)}")
    ax.set_xlabel(_label(col))
    ax.set_ylabel("Frecuencia (hexágonos)")
    return ax


def plot_matriz_correlacion(df: pd.DataFrame) -> plt.Axes:
    """Dibuja la matriz de correlación (Pearson) de las columnas numéricas.

    Args:
        df: DataFrame; se usan solo sus columnas numéricas.

    Returns:
        El ``Axes`` con el heatmap de correlaciones anotado.
    """
    num = df.select_dtypes("number")
    corr = num.corr()

    fig, ax = plt.subplots(figsize=(10, 8))
    im = ax.imshow(corr, cmap="RdBu_r", vmin=-1, vmax=1, aspect="auto")
    ax.set_xticks(range(len(corr.columns)))
    ax.set_yticks(range(len(corr.columns)))
    ax.set_xticklabels(corr.columns, rotation=90)
    ax.set_yticklabels(corr.columns)
    ax.set_title("Matriz de correlación (Pearson)")

    # Anotar cada celda con el coeficiente.
    for i in range(len(corr.columns)):
        for j in range(len(corr.columns)):
            val = corr.iloc[i, j]
            color = "white" if abs(val) > 0.5 else "black"
            ax.text(j, i, f"{val:.2f}", ha="center", va="center",
                    color=color, fontsize=7)
    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04, label="Correlación")
    return ax


def plot_bivariado(
    df: pd.DataFrame, col: str, target: str, ax: Optional[plt.Axes] = None
) -> plt.Axes:
    """Grafica una predictora contra el target.

    Si ``col`` es numérica dibuja un scatter; si es categórica dibuja un
    boxplot del target por categoría.

    Args:
        df: DataFrame con los datos.
        col: Predictora (numérica o categórica).
        target: Columna objetivo (numérica) en el eje Y.
        ax: Eje donde dibujar. Si es ``None`` se crea uno nuevo.

    Returns:
        El ``Axes`` con el gráfico bivariado.
    """
    if ax is None:
        _, ax = plt.subplots(figsize=(6, 4))

    if pd.api.types.is_numeric_dtype(df[col]):
        ax.scatter(df[col], df[target], s=10, alpha=0.4, color="#c0504d")
        ax.set_xlabel(_label(col))
        ax.set_ylabel(_label(target))
        ax.set_title(f"{_label(target)} vs {_label(col)}")
    else:
        cats = [c for c in df[col].dropna().unique()]
        cats = sorted(cats, key=lambda c: df.loc[df[col] == c, target].median())
        datos = [df.loc[df[col] == c, target].dropna().values for c in cats]
        ax.boxplot(datos, labels=cats, vert=True, showfliers=True)
        ax.set_xlabel(_label(col))
        ax.set_ylabel(_label(target))
        ax.set_title(f"{_label(target)} por {_label(col)}")
        ax.tick_params(axis="x", rotation=45)
    return ax
