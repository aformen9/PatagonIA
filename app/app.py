# -*- coding: utf-8 -*-
"""App Streamlit del TP4 — estimación de riesgo de incendio (PatagonIA).

Dashboard oscuro que carga el `Pipeline` serializado en el Paso 1
(`app/models/`) y su metadata, y estima —a partir de 8 predictoras
ambientales— si un hexágono es de **riesgo alto** de incendio. No re-entrena ni
necesita el dataset crudo: sólo consume `clasificador_riesgo.joblib` y
`metadata.json`.

Ejecutar:  ``streamlit run app/app.py``
"""
from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

# --- Rutas de los artefactos (relativas a este archivo) ---------------------
DIR_APP = Path(__file__).resolve().parent
DIR_MODELOS = DIR_APP / "models"
RUTA_MODELO = DIR_MODELOS / "clasificador_riesgo.joblib"
RUTA_METADATA = DIR_MODELOS / "metadata.json"

# --- Paleta del tema oscuro (coherente con .streamlit/config.toml) ----------
COLOR_FONDO = "#0B1120"
COLOR_PANEL = "#151E32"
COLOR_TEXTO = "#E5E9F0"
COLOR_TENUE = "#8A94A6"
COLOR_ALTO = "#F87171"       # riesgo alto (rojo suave)
COLOR_BAJO = "#00D97E"       # riesgo bajo (verde primario)
COLOR_BARRA = "#3B82F6"      # azul para el histograma

# --- Agrupación de los sliders en la barra lateral --------------------------
GRUPOS_SLIDERS: dict[str, list[str]] = {
    "Clima": ["temp_media", "precip_anual", "viento_medio", "humedad_relativa"],
    "Relieve y vegetación": ["elevacion"],   # + cobertura_veg (categórica)
    "Accesibilidad": ["dist_asentamiento_km", "dist_ruta_km"],
}


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
# Lógica de dominio (idéntica al modelo original; sólo cambia la presentación)
# ---------------------------------------------------------------------------
def parametros_slider(
    rango: dict[str, float],
) -> tuple[float, float, float, float, str]:
    """Calcula ``(min, max, valor_defecto, paso, formato)`` para un slider.

    El **valor por defecto es siempre la mediana** leída de ``metadata.json``.
    Para que Streamlit ubique bien el control, redondea extremos y mediana a una
    misma cantidad de decimales y usa un paso de ``10**-decimales``: así la
    mediana cae **exactamente sobre la grilla** del slider (si no, el frontend
    de Streamlit desplaza el control cerca del máximo). Los extremos se
    redondean hacia afuera para no recortar el rango observado.

    Args:
        rango: Entrada de ``metadata["rangos"]`` con ``min``, ``max`` y
            ``mediana``.

    Returns:
        Tupla ``(lo, hi, valor_defecto, paso, formato)`` para ``st.slider``.
    """
    lo_obs, hi_obs = float(rango["min"]), float(rango["max"])
    mediana = float(rango["mediana"])
    ancho = hi_obs - lo_obs
    objetivo = ancho / 200.0 if ancho > 0 else 0.01
    decimales = min(max(0, math.ceil(-math.log10(objetivo))), 3)

    factor = 10 ** decimales
    lo = math.floor(lo_obs * factor) / factor
    hi = math.ceil(hi_obs * factor) / factor
    valor = min(max(round(mediana, decimales), lo), hi)  # mediana sobre grilla
    paso = 1.0 / factor
    return lo, hi, valor, paso, f"%.{decimales}f"


def recoger_entradas(meta: dict[str, Any]) -> pd.DataFrame:
    """Dibuja los controles agrupados y devuelve un DataFrame de una fila.

    Los sliders se organizan en tres expanders temáticos (Clima, Relieve y
    vegetación, Accesibilidad). Cada slider está acotado al rango observado
    (imposible ingresar valores absurdos) y muestra su min/max debajo. La
    cobertura vegetal es un ``selectbox``.

    Args:
        meta: Metadata del modelo.

    Returns:
        DataFrame de una fila con las 8 columnas que espera el Pipeline, en el
        orden correcto.
    """
    valores: dict[str, Any] = {}
    st.sidebar.header("Condiciones del hexágono")
    st.sidebar.caption(
        "Ajustá las condiciones ambientales. Los límites son los observados en "
        "la Patagonia (2012–2023): no se pueden ingresar valores imposibles."
    )
    cat = meta["feature_categorica"]

    for grupo, columnas in GRUPOS_SLIDERS.items():
        with st.sidebar.expander(grupo, expanded=(grupo == "Clima")):
            for col in columnas:
                r = meta["rangos"][col]
                lo, hi, valor, paso, fmt = parametros_slider(r)
                valores[col] = st.slider(
                    r["etiqueta"], min_value=lo, max_value=hi, value=valor,
                    step=paso, format=fmt,
                    help=f"Por defecto: la mediana ({fmt % r['mediana']} "
                         f"{r['unidad']}).",
                )
                st.caption(
                    f"mín {fmt % r['min']}  ·  máx {fmt % r['max']} "
                    f"{r['unidad']}"
                )
            # La cobertura vegetal acompaña al relieve.
            if grupo == "Relieve y vegetación":
                valores[cat] = st.selectbox(
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
# Gráficos Plotly (tema oscuro)
# ---------------------------------------------------------------------------
def _layout_oscuro(fig: go.Figure, altura: int = 300) -> go.Figure:
    """Aplica el tema oscuro común a una figura Plotly.

    Args:
        fig: Figura a estilizar.
        altura: Altura en píxeles.

    Returns:
        La misma figura, con fondo transparente y tipografía del tema.
    """
    fig.update_layout(
        template="plotly_dark",
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        font=dict(color=COLOR_TEXTO, family="Inter, Segoe UI, sans-serif"),
        margin=dict(l=10, r=10, t=40, b=10),
        height=altura,
    )
    return fig


def grafico_gauge(prob: float, umbral: float, es_alto: bool) -> go.Figure:
    """Gauge circular de probabilidad con el umbral de decisión marcado.

    Args:
        prob: Probabilidad estimada de riesgo alto (0–1).
        umbral: Umbral de decisión (0–1).
        es_alto: Si la clasificación es riesgo alto.

    Returns:
        Figura Plotly con un indicador ``gauge+number``.
    """
    color = COLOR_ALTO if es_alto else COLOR_BAJO
    fig = go.Figure(go.Indicator(
        mode="gauge+number",
        value=prob * 100.0,
        number={"suffix": " %", "font": {"size": 44, "color": color}},
        gauge={
            "axis": {"range": [0, 100], "tickcolor": COLOR_TENUE,
                     "tickfont": {"color": COLOR_TENUE}},
            "bar": {"color": color, "thickness": 0.28},
            "bgcolor": "rgba(0,0,0,0)",
            "borderwidth": 0,
            "steps": [
                {"range": [0, umbral * 100], "color": "rgba(0,217,126,0.12)"},
                {"range": [umbral * 100, 100], "color": "rgba(248,113,113,0.12)"},
            ],
            "threshold": {
                "line": {"color": COLOR_TEXTO, "width": 3},
                "thickness": 0.85, "value": umbral * 100,
            },
        },
        domain={"x": [0, 1], "y": [0, 1]},
    ))
    fig.add_annotation(
        x=0.5, y=0.02, showarrow=False, xref="paper", yref="paper",
        text=f"umbral de decisión: {umbral * 100:.1f} %",
        font=dict(color=COLOR_TENUE, size=12),
    )
    return _layout_oscuro(fig, altura=300)


def grafico_histograma_focos(meta: dict[str, Any], es_alto: bool) -> go.Figure:
    """Histograma de ``n_focos`` (escala log) con la zona estimada resaltada.

    El modelo no predice el número exacto de focos, sino el **lado** del umbral
    (150 focos) en que caería el hexágono. Se marca ese umbral y se sombrea la
    región de la clase estimada.

    Args:
        meta: Metadata con ``distribucion_n_focos`` y ``umbral_riesgo_focos``.
        es_alto: Si la clasificación estimada es riesgo alto.

    Returns:
        Figura Plotly.
    """
    focos = np.asarray(meta["distribucion_n_focos"], dtype=float)
    focos = focos[focos > 0]
    umbral = float(meta["umbral_riesgo_focos"])
    log_focos = np.log10(focos)
    log_umbral = math.log10(umbral)

    fig = go.Figure()
    fig.add_trace(go.Histogram(
        x=log_focos, nbinsx=40, marker_color=COLOR_BARRA,
        marker_line_color=COLOR_FONDO, marker_line_width=1,
        hovertemplate="~%{x} (log10 focos)<extra></extra>", showlegend=False,
    ))

    zona_color = "rgba(248,113,113,0.14)" if es_alto else "rgba(0,217,126,0.14)"
    if es_alto:
        fig.add_vrect(x0=log_umbral, x1=log_focos.max(), fillcolor=zona_color,
                      line_width=0)
    else:
        fig.add_vrect(x0=log_focos.min(), x1=log_umbral, fillcolor=zona_color,
                      line_width=0)
    fig.add_vline(x=log_umbral, line_dash="dash", line_color=COLOR_ALTO,
                  line_width=2)
    fig.add_annotation(x=log_umbral, y=1, yref="paper", yanchor="bottom",
                       showarrow=False, text=f" umbral {int(umbral)} focos",
                       font=dict(color=COLOR_ALTO, size=12), xanchor="left")

    etiqueta = "RIESGO ALTO" if es_alto else "RIESGO BAJO"
    color_txt = COLOR_ALTO if es_alto else COLOR_BAJO
    fig.add_annotation(x=0.02, y=0.95, xref="paper", yref="paper",
                       showarrow=False, xanchor="left", yanchor="top",
                       text=f"zona estimada: <b>{etiqueta}</b>",
                       font=dict(color=color_txt, size=13))

    ticks = [1, 10, 100, 1000]
    fig.update_xaxes(
        title_text="N.º de focos por hexágono, 2012–2023 (escala log)",
        tickvals=[math.log10(t) for t in ticks], ticktext=[str(t) for t in ticks],
        gridcolor="rgba(255,255,255,0.06)",
    )
    fig.update_yaxes(title_text="Hexágonos", gridcolor="rgba(255,255,255,0.06)")
    return _layout_oscuro(fig, altura=300)


def grafico_importancias(meta: dict[str, Any]) -> go.Figure:
    """Barras horizontales de la importancia de cada predictora en el modelo.

    Args:
        meta: Metadata con el diccionario ``importancias``.

    Returns:
        Figura Plotly.
    """
    etiquetas = dict(meta["etiquetas"])
    etiquetas[meta["feature_categorica"]] = "Cobertura vegetal"
    items = sorted(meta["importancias"].items(), key=lambda kv: kv[1])
    nombres = [etiquetas.get(k, k) for k, _ in items]
    valores = [v for _, v in items]

    fig = go.Figure(go.Bar(
        x=valores, y=nombres, orientation="h",
        marker_color=COLOR_BAJO, marker_line_width=0,
        hovertemplate="%{y}: %{x:.3f}<extra></extra>",
    ))
    fig.update_xaxes(title_text="Importancia relativa (XGBoost)",
                     gridcolor="rgba(255,255,255,0.06)")
    fig.update_yaxes(gridcolor="rgba(255,255,255,0.0)")
    return _layout_oscuro(fig, altura=320)


# ---------------------------------------------------------------------------
# Componentes de UI
# ---------------------------------------------------------------------------
def inyectar_estilos() -> None:
    """Inyecta CSS para tipografía, espaciado y tarjetas KPI."""
    st.markdown(
        f"""
        <style>
        .block-container {{ padding-top: 2.4rem; padding-bottom: 2.5rem;
            max-width: 1320px; }}
        h1, h2, h3 {{ font-family: 'Inter','Segoe UI',sans-serif;
            letter-spacing: -0.02em; }}
        .subtitulo {{ color: {COLOR_TENUE}; font-size: 1.02rem;
            margin-top: -0.4rem; margin-bottom: 0.4rem; }}
        .kpi {{ background: {COLOR_PANEL}; border: 1px solid #24304a;
            border-radius: 14px; padding: 18px 20px; height: 118px;
            display: flex; flex-direction: column; justify-content: center; }}
        .kpi-label {{ font-size: 0.72rem; letter-spacing: 0.09em;
            text-transform: uppercase; color: {COLOR_TENUE};
            margin-bottom: 8px; }}
        .kpi-value {{ font-size: 1.85rem; font-weight: 700; line-height: 1.05; }}
        .kpi-sub {{ font-size: 0.8rem; color: {COLOR_TENUE}; margin-top: 6px; }}
        .badge {{ display: inline-block; padding: 2px 10px; border-radius: 999px;
            font-size: 0.74rem; font-weight: 600; }}
        .badge-ok {{ background: rgba(0,217,126,0.16); color: {COLOR_BAJO}; }}
        .badge-warn {{ background: rgba(248,113,113,0.16); color: {COLOR_ALTO}; }}
        .seccion {{ font-size: 0.78rem; letter-spacing: 0.09em;
            text-transform: uppercase; color: {COLOR_TENUE};
            margin: 0.5rem 0 0.2rem 0; }}
        </style>
        """,
        unsafe_allow_html=True,
    )


def tarjeta_kpi(rotulo: str, valor: str, sub: str = "", *,
                color: str = COLOR_TEXTO, badge: tuple[str, str] | None = None
                ) -> str:
    """Construye el HTML de una tarjeta KPI.

    Args:
        rotulo: Etiqueta superior (en mayúsculas por CSS).
        valor: Valor grande, ya formateado.
        sub: Texto auxiliar debajo del valor.
        color: Color del valor grande.
        badge: Par ``(texto, clase_css)`` para un pill opcional, o ``None``.

    Returns:
        Cadena HTML de la tarjeta.
    """
    pill = f'<span class="badge {badge[1]}">{badge[0]}</span>' if badge else ""
    return (
        f'<div class="kpi"><div class="kpi-label">{rotulo}</div>'
        f'<div class="kpi-value" style="color:{color}">{valor}</div>'
        f'<div class="kpi-sub">{sub} {pill}</div></div>'
    )


# ---------------------------------------------------------------------------
# Aplicación
# ---------------------------------------------------------------------------
def main() -> None:
    """Punto de entrada de la app Streamlit."""
    st.set_page_config(page_title="PatagonIA — Riesgo de incendio",
                       layout="wide")
    inyectar_estilos()

    if not RUTA_MODELO.exists() or not RUTA_METADATA.exists():
        st.error(
            "No se encontraron los artefactos del modelo en `app/models/`. "
            "Ejecutá primero el notebook `notebooks/09_serializacion_tp4.ipynb`."
        )
        st.stop()

    modelo = cargar_modelo(RUTA_MODELO)
    meta = cargar_metadata(RUTA_METADATA)

    # --- Entrada (barra lateral) ---
    entrada = recoger_entradas(meta)

    # --- Predicción (lógica intacta respecto del diseño anterior) ---
    umbral = float(meta["umbral_decision"])
    prob = float(modelo.predict_proba(entrada)[:, 1][0])
    es_alto = prob >= umbral
    color_res = COLOR_ALTO if es_alto else COLOR_BAJO

    dist = distancia_mahalanobis(entrada, meta)
    cutoff = float(meta["envolvente"]["cutoff_mahalanobis"])
    atipico = dist > cutoff

    # === Header: título + subtítulo en una línea ===
    st.title("PatagonIA — Estimación de riesgo de incendio")
    st.markdown(
        '<div class="subtitulo">Propensión estructural al fuego por hexágono '
        "a partir de condiciones ambientales · promedio 2012–2023, "
        "no el riesgo de hoy.</div>",
        unsafe_allow_html=True,
    )
    with st.expander("Sobre este modelo"):
        st.markdown(
            f"Estima si un hexágono de la Patagonia es de **riesgo alto de "
            f"incendio** a partir de sus condiciones ambientales. El modelo es "
            f"un **{meta['modelo']}** (gradient boosting) entrenado sobre "
            f"**{meta['n_muestras']} hexágonos H3** de la región (datos "
            f"2012–2023, agregados). Define *riesgo alto* como haber registrado "
            f"más de **{meta['umbral_riesgo_focos']} focos** históricos. Usa "
            f"**sólo variables ambientales** (clima, relieve, vegetación, "
            f"distancias) — ninguna variable del propio fuego. La app estima la "
            f"**propensión estructural** de una zona con estas características, "
            f"**no** el riesgo de incendio de un día concreto."
        )

    # === Fila de KPIs ===
    k1, k2, k3, k4 = st.columns(4)
    k1.markdown(
        tarjeta_kpi("Clasificación", "RIESGO ALTO" if es_alto else "RIESGO BAJO",
                    "según el umbral de decisión", color=color_res),
        unsafe_allow_html=True)
    k2.markdown(
        tarjeta_kpi("Probabilidad estimada", f"{prob * 100:.1f}%",
                    "de riesgo alto", color=COLOR_TEXTO),
        unsafe_allow_html=True)
    k3.markdown(
        tarjeta_kpi("Umbral de decisión", f"{umbral:.3f}",
                    "recall ≈ 0.80 (prioriza detección)", color=COLOR_TEXTO),
        unsafe_allow_html=True)
    badge = (("ATÍPICO", "badge-warn") if atipico else ("OK", "badge-ok"))
    k4.markdown(
        tarjeta_kpi("Distancia a la envolvente", f"{dist:.2f}",
                    f"corte {cutoff:.2f} ·", color=COLOR_TEXTO, badge=badge),
        unsafe_allow_html=True)

    st.write("")

    # === Fila 2: gauge + histograma de contexto ===
    c1, c2 = st.columns(2)
    with c1:
        st.markdown('<div class="seccion">Probabilidad</div>',
                    unsafe_allow_html=True)
        st.plotly_chart(grafico_gauge(prob, umbral, es_alto),
                        use_container_width=True)
    with c2:
        st.markdown('<div class="seccion">Contexto histórico</div>',
                    unsafe_allow_html=True)
        st.plotly_chart(grafico_histograma_focos(meta, es_alto),
                        use_container_width=True)

    # === Fila 3: importancias + interpretación ===
    c3, c4 = st.columns(2)
    with c3:
        st.markdown('<div class="seccion">Qué pesa en la estimación</div>',
                    unsafe_allow_html=True)
        st.plotly_chart(grafico_importancias(meta), use_container_width=True)
    with c4:
        st.markdown('<div class="seccion">Interpretación</div>',
                    unsafe_allow_html=True)
        if es_alto:
            st.markdown(
                "Bajo estas condiciones, el hexágono se parece a las zonas que "
                "**históricamente concentraron muchos focos** de incendio. No "
                "significa que se vaya a incendiar: indica que el **ambiente es "
                "propicio** y que sería una zona a **priorizar para vigilancia "
                "y prevención**."
            )
        else:
            st.markdown(
                "Estas condiciones se parecen a las de zonas que "
                "**históricamente registraron pocos focos**. El ambiente "
                "resulta **menos propicio** al fuego, aunque el riesgo nunca es "
                "cero: conviene mantener las precauciones habituales."
            )
        if atipico:
            culpables = features_mas_atipicas(entrada, meta)
            st.markdown(
                f"<span class='kpi-sub'>Nota: la combinación elegida (sobre "
                f"todo <b>{'</b> y <b>'.join(culpables)}</b>) casi no aparece "
                f"en los datos reales; el modelo está <b>extrapolando</b> y la "
                f"estimación es menos confiable.</span>",
                unsafe_allow_html=True,
            )

        met = meta["metricas"]
        st.markdown("<div class='seccion' style='margin-top:1rem'>Métricas de "
                    "referencia</div>", unsafe_allow_html=True)
        st.markdown(
            f"- **F1 (riesgo alto):** {met['f1_pos_media']:.2f} ± "
            f"{met['f1_pos_desvio']:.2f} (multi-semilla, TP3)\n"
            f"- **AUC-PR:** {met['auc_pr']:.2f} "
            f"(azar: {meta['prevalencia']:.2f})\n"
            f"- En el umbral **{umbral:.3f}** → recall "
            f"**{met['recall_en_umbral']:.2f}** / precisión "
            f"**{met['precision_en_umbral']:.2f}**"
        )

    # === Limitaciones ===
    with st.expander("Limitaciones — leer antes de usar"):
        st.markdown(
            "- **Sólo hexágonos con actividad previa.** El dataset se construyó "
            "sobre celdas que registraron **al menos un foco** entre 2012 y "
            "2023; no dice nada sobre zonas sin historial.\n"
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
