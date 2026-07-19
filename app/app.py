# -*- coding: utf-8 -*-
"""App Streamlit del TP4 — estimación de riesgo de incendio (PatagonIA).

Carga el `Pipeline` serializado en el Paso 1 (`app/models/`) y su metadata, y
expone una interfaz para estimar, a partir de 8 predictoras ambientales, si un
hexágono es de **riesgo alto** de incendio. No re-entrena ni necesita el dataset
crudo: sólo consume `clasificador_riesgo.joblib` y `metadata.json`.

Ejecutar:  ``streamlit run app/app.py``
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import streamlit as st

# --- Rutas de los artefactos (relativas a este archivo) ---------------------
DIR_APP = Path(__file__).resolve().parent
DIR_MODELOS = DIR_APP / "models"
RUTA_MODELO = DIR_MODELOS / "clasificador_riesgo.joblib"
RUTA_METADATA = DIR_MODELOS / "metadata.json"

# Paletas de color coherentes con la semántica riesgo alto / bajo.
COLOR_ALTO = "#c0392b"
COLOR_BAJO = "#1e8449"


# ---------------------------------------------------------------------------
# Carga de artefactos (cacheada para no releer en cada interacción)
# ---------------------------------------------------------------------------
@st.cache_resource(show_spinner=False)
def cargar_modelo(ruta: Path) -> Any:
    """Carga el Pipeline serializado con joblib.

    Args:
        ruta: Ruta al archivo ``.joblib`` con el Pipeline completo.

    Returns:
        El estimador scikit-learn recargado (preprocesamiento + modelo).
    """
    return joblib.load(ruta)


@st.cache_data(show_spinner=False)
def cargar_metadata(ruta: Path) -> dict[str, Any]:
    """Carga la metadata JSON del modelo.

    Args:
        ruta: Ruta al ``metadata.json`` generado en el Paso 1.

    Returns:
        Diccionario con rangos, categorías, umbral, métricas, importancias,
        envolvente y distribución de ``n_focos``.
    """
    with open(ruta, encoding="utf-8") as archivo:
        return json.load(archivo)


# ---------------------------------------------------------------------------
# Lógica de dominio
# ---------------------------------------------------------------------------
def recoger_entradas(meta: dict[str, Any]) -> pd.DataFrame:
    """Dibuja los controles de entrada y devuelve un DataFrame de una fila.

    Genera 7 sliders numéricos (acotados al rango observado, con la mediana
    como valor por defecto) y un ``selectbox`` para la cobertura vegetal. Las
    cotas de los sliders hacen **imposible** ingresar valores absurdos fuera del
    rango del dataset.

    Args:
        meta: Metadata del modelo.

    Returns:
        DataFrame de una fila con las 8 columnas que espera el Pipeline, en el
        orden correcto.
    """
    valores: dict[str, Any] = {}
    st.sidebar.header("🌱 Condiciones del hexágono")
    st.sidebar.caption(
        "Ajustá las condiciones ambientales. Los límites son los observados "
        "en la Patagonia (2012–2023): no se pueden ingresar valores imposibles."
    )

    for col in meta["features_numericas"]:
        r = meta["rangos"][col]
        lo, hi, med = float(r["min"]), float(r["max"]), float(r["mediana"])
        # Paso proporcional al rango, para un control fluido.
        paso = max((hi - lo) / 200.0, 0.01)
        valores[col] = st.sidebar.slider(
            r["etiqueta"], min_value=lo, max_value=hi, value=med, step=paso,
            help=f"Rango observado: {lo:.1f} – {hi:.1f} {r['unidad']}",
        )

    cat = meta["feature_categorica"]
    valores[cat] = st.sidebar.selectbox(
        "Cobertura vegetal", options=meta["categorias_cobertura"],
        help="Tipo de cobertura dominante del hexágono.",
    )

    columnas = meta["features_numericas"] + [cat]
    return pd.DataFrame([valores])[columnas]


def distancia_mahalanobis(fila: pd.DataFrame, meta: dict[str, Any]) -> float:
    """Distancia de Mahalanobis del input a la nube de datos observada.

    Mide cuán atípica es la **combinación** de valores numéricos respecto de la
    estructura de correlación del dataset. Un valor alto delata combinaciones
    que individualmente son válidas pero que **no coexisten** (p. ej.
    precipitación alta con humedad baja).

    Args:
        fila: DataFrame de una fila con las columnas de entrada.
        meta: Metadata con la envolvente (media y covarianza).

    Returns:
        Distancia de Mahalanobis (escalar no negativo).
    """
    env = meta["envolvente"]
    x = fila[env["features"]].to_numpy(dtype=float).ravel()
    mu = np.asarray(env["media"], dtype=float)
    inv = np.linalg.pinv(np.asarray(env["cov"], dtype=float))
    delta = x - mu
    return float(np.sqrt(delta @ inv @ delta))


def features_mas_atipicas(fila: pd.DataFrame, meta: dict[str, Any],
                          n: int = 2) -> list[str]:
    """Devuelve las features numéricas más alejadas de su mediana (en z).

    Sirve para orientar al usuario sobre *qué* está forzando una combinación
    fuera de la envolvente.

    Args:
        fila: DataFrame de una fila con las columnas de entrada.
        meta: Metadata del modelo.
        n: Cantidad de features a devolver.

    Returns:
        Lista de etiquetas legibles, de la más atípica a la menos.
    """
    env = meta["envolvente"]
    x = fila[env["features"]].to_numpy(dtype=float).ravel()
    mu = np.asarray(env["media"], dtype=float)
    sd = np.sqrt(np.diag(np.asarray(env["cov"], dtype=float)))
    z = np.abs((x - mu) / np.where(sd == 0, 1.0, sd))
    orden = np.argsort(z)[::-1][:n]
    return [meta["rangos"][env["features"][i]]["etiqueta"] for i in orden]


# ---------------------------------------------------------------------------
# Gráficos de contexto
# ---------------------------------------------------------------------------
def grafico_histograma_focos(meta: dict[str, Any], es_alto: bool) -> plt.Figure:
    """Histograma de ``n_focos`` con la zona estimada resaltada.

    El modelo no predice el número exacto de focos, sino el **lado** del umbral
    (150 focos) en que caería el hexágono. Se marca ese umbral y se sombrea la
    región correspondiente a la clase estimada.

    Args:
        meta: Metadata con ``distribucion_n_focos`` y ``umbral_riesgo_focos``.
        es_alto: Si la clasificación estimada es riesgo alto.

    Returns:
        La figura de matplotlib.
    """
    focos = np.asarray(meta["distribucion_n_focos"], dtype=float)
    focos = focos[focos > 0]
    umbral = meta["umbral_riesgo_focos"]

    fig, ax = plt.subplots(figsize=(6.2, 3.2))
    bins = np.logspace(0, np.log10(focos.max() + 1), 40)
    ax.hist(focos, bins=bins, color="#a9cce3", edgecolor="white")
    ax.set_xscale("log")

    if es_alto:
        ax.axvspan(umbral, bins[-1], color=COLOR_ALTO, alpha=0.15)
        texto, color = "zona estimada:\nRIESGO ALTO", COLOR_ALTO
    else:
        ax.axvspan(bins[0], umbral, color=COLOR_BAJO, alpha=0.15)
        texto, color = "zona estimada:\nRIESGO BAJO", COLOR_BAJO
    ax.axvline(umbral, color=COLOR_ALTO, ls="--", lw=2)
    ax.annotate(f"umbral = {umbral} focos", xy=(umbral, 0),
                xytext=(umbral, ax.get_ylim()[1] * 0.78), color=COLOR_ALTO,
                fontsize=8, ha="center")
    ax.text(0.02, 0.95, texto, transform=ax.transAxes, va="top", fontsize=9,
            color=color, fontweight="bold")

    ax.set_xlabel("N.º de focos por hexágono, 2012–2023 (escala log)")
    ax.set_ylabel("Cantidad de hexágonos")
    ax.set_title("Distribución histórica y zona estimada")
    fig.tight_layout()
    return fig


def grafico_importancias(meta: dict[str, Any]) -> plt.Figure:
    """Gráfico de barras de la importancia de cada predictora en el modelo.

    Args:
        meta: Metadata con el diccionario ``importancias``.

    Returns:
        La figura de matplotlib.
    """
    imp = meta["importancias"]
    etiquetas = meta["etiquetas"].copy()
    etiquetas[meta["feature_categorica"]] = "Cobertura vegetal"

    items = sorted(imp.items(), key=lambda kv: kv[1])
    nombres = [etiquetas.get(k, k) for k, _ in items]
    valores = [v for _, v in items]

    fig, ax = plt.subplots(figsize=(6.2, 3.4))
    ax.barh(nombres, valores, color="#5499c7")
    ax.set_xlabel("Importancia relativa (XGBoost)")
    ax.set_title("Qué pesa más en la estimación")
    ax.grid(axis="x", alpha=0.3)
    fig.tight_layout()
    return fig


# ---------------------------------------------------------------------------
# Aplicación
# ---------------------------------------------------------------------------
def main() -> None:
    """Punto de entrada de la app Streamlit."""
    st.set_page_config(page_title="PatagonIA — Riesgo de incendio",
                       page_icon="🔥", layout="wide")

    if not RUTA_MODELO.exists() or not RUTA_METADATA.exists():
        st.error(
            "No se encontraron los artefactos del modelo en `app/models/`. "
            "Ejecutá primero el notebook `notebooks/09_serializacion_tp4.ipynb`."
        )
        st.stop()

    modelo = cargar_modelo(RUTA_MODELO)
    meta = cargar_metadata(RUTA_METADATA)

    # --- Título y descripción ---
    st.title("PatagonIA — Estimación de riesgo de incendio")
    st.markdown(
        "Estimá si un hexágono de la Patagonia es de **riesgo alto de "
        "incendio** a partir de sus condiciones ambientales. "
        f"El modelo es un **{meta['modelo']}** (gradient boosting) entrenado "
        f"sobre **{meta['n_muestras']} hexágonos H3** de la región "
        "(datos 2012–2023, agregados). Define *riesgo alto* como haber "
        f"registrado más de **{meta['umbral_riesgo_focos']} focos** históricos. "
        "Usa **sólo variables ambientales** (clima, relieve, vegetación, "
        "distancias) — ninguna variable del propio fuego."
    )
    st.divider()

    # --- Entrada (barra lateral) ---
    entrada = recoger_entradas(meta)

    # --- Predicción ---
    umbral = float(meta["umbral_decision"])
    prob = float(modelo.predict_proba(entrada)[:, 1][0])
    es_alto = prob >= umbral

    # === Salida ===
    col_res, col_ctx = st.columns([1, 1])

    with col_res:
        st.subheader("Resultado")
        if es_alto:
            st.error("### 🔴 Riesgo ALTO")
        else:
            st.success("### 🟢 Riesgo BAJO")

        st.metric("Probabilidad estimada de riesgo alto", f"{prob * 100:.1f} %")
        st.progress(min(max(prob, 0.0), 1.0))
        st.caption(
            f"Se clasifica como *riesgo alto* cuando la probabilidad supera el "
            f"umbral de decisión **{umbral:.3f}**, elegido para **priorizar la "
            f"detección** (recall alto) sobre la precisión."
        )

        # --- Texto interpretativo, sin jerga ---
        if es_alto:
            st.markdown(
                "**Qué significa.** Bajo estas condiciones, el hexágono se "
                "parece a las zonas que **históricamente concentraron muchos "
                "focos** de incendio. No quiere decir que se vaya a incendiar: "
                "indica que el **ambiente es propicio** y que sería una zona a "
                "**priorizar para vigilancia y prevención**."
            )
        else:
            st.markdown(
                "**Qué significa.** Estas condiciones se parecen a las de zonas "
                "que **históricamente registraron pocos focos**. El ambiente "
                "resulta **menos propicio** al fuego, aunque el riesgo nunca es "
                "cero: conviene igual mantener las precauciones habituales."
            )

        # --- Validación: envolvente del dataset ---
        dist = distancia_mahalanobis(entrada, meta)
        if dist > float(meta["envolvente"]["cutoff_mahalanobis"]):
            culpables = features_mas_atipicas(entrada, meta)
            st.warning(
                "⚠️ **Combinación poco habitual.** Los valores elegidos "
                "(sobre todo *" + "* y *".join(culpables) + "*) forman una "
                "combinación que **casi no aparece** en los datos reales de la "
                "Patagonia (p. ej. mucha lluvia con poca humedad, que rara vez "
                "coexisten). La estimación es **menos confiable** en esta zona: "
                "el modelo está extrapolando."
            )

    with col_ctx:
        st.subheader("Contexto")
        st.pyplot(grafico_histograma_focos(meta, es_alto))

    st.divider()

    # === Contexto informativo ===
    c1, c2 = st.columns([1, 1])
    with c1:
        st.pyplot(grafico_importancias(meta))
    with c2:
        st.markdown("#### Métricas de referencia del modelo")
        met = meta["metricas"]
        st.markdown(
            f"- **F1 (clase riesgo alto):** {met['f1_pos_media']:.2f} ± "
            f"{met['f1_pos_desvio']:.2f} (multi-semilla, TP3)\n"
            f"- **AUC-PR:** {met['auc_pr']:.2f} "
            f"(línea base por azar: {meta['prevalencia']:.2f})\n"
            f"- En el umbral **{umbral:.3f}** → "
            f"**recall {met['recall_en_umbral']:.2f}** / "
            f"**precisión {met['precision_en_umbral']:.2f}**"
        )
        st.caption(
            "El modelo prioriza el *recall*: prefiere marcar de más (algunas "
            "falsas alarmas) antes que dejar pasar un hexágono realmente "
            "peligroso. Las métricas provienen de la evaluación con validación "
            "cruzada del TP3, no de estos mismos datos de entrenamiento."
        )

    with st.expander("ℹ️ Limitaciones — leer antes de usar"):
        st.markdown(
            "- **Sólo hexágonos con actividad previa.** El dataset se construyó "
            "sobre celdas que registraron **al menos un foco** entre 2012 y "
            "2023; la herramienta no dice nada sobre zonas sin historial.\n"
            "- **Datos agregados 2012–2023.** Cada hexágono resume más de una "
            "década: es un perfil de **propensión estructural**, no una "
            "predicción para un día u hora concretos.\n"
            "- **Zona de *flaring* excluida.** Se removió el cuadrante de Vaca "
            "Muerta, donde la señal térmica corresponde a **quema de gas**, no "
            "a incendios.\n"
            "- **No es un sistema operativo de alerta.** Es un trabajo "
            "académico (TP4): no incorpora meteorología en tiempo real, "
            "combustible actual ni ignición humana inmediata, y **no debe "
            "usarse para decisiones operativas** de emergencia.\n"
            "- **Estima propensión, no certeza.** Un 'riesgo alto' señala un "
            "ambiente propicio; un 'riesgo bajo' no garantiza que no pueda "
            "haber fuego."
        )


if __name__ == "__main__":
    main()
