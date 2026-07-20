# -*- coding: utf-8 -*-
"""App Streamlit del TP4 — estimación de riesgo de incendio (PatagonIA).

Dashboard oscuro que carga el `Pipeline` serializado en el Paso 1
(`app/models/`) y su metadata, y estima —a partir de 8 predictoras
ambientales— si un hexágono es de **riesgo alto** de incendio. No re-entrena ni
necesita el dataset crudo: sólo consume `clasificador_riesgo.joblib`,
`metadata.json` y `hexagonos.csv` (el tablero para el mapa interactivo).

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
import pydeck as pdk
import streamlit as st
import xgboost as xgb

# --- Rutas de los artefactos (relativas a este archivo) ---------------------
DIR_APP = Path(__file__).resolve().parent
DIR_MODELOS = DIR_APP / "models"
RUTA_MODELO = DIR_MODELOS / "clasificador_riesgo.joblib"
RUTA_METADATA = DIR_MODELOS / "metadata.json"
RUTA_HEXAGONOS = DIR_MODELOS / "hexagonos.csv"

# --- Paleta del tema oscuro (coherente con .streamlit/config.toml) ----------
COLOR_FONDO = "#0B1120"
COLOR_PANEL = "#151E32"
COLOR_TEXTO = "#E5E9F0"
COLOR_TENUE = "#8A94A6"
COLOR_ALTO = "#F87171"       # riesgo alto (rojo suave)
COLOR_BAJO = "#00D97E"       # riesgo bajo (verde primario)
COLOR_BARRA = "#3B82F6"      # azul para el histograma

# Paleta del mapa de "focos observados": misma escala crema→naranja→marrón
# que knime/presentacion/generar_mapa_h3.py, para que ambas figuras del
# trabajo (presentación y app) se lean como el mismo sistema visual.
COLOR_FOCOS_CLARO = "#F4EFE6"
COLOR_FOCOS_MEDIO = "#E8843C"
COLOR_FOCOS_OSCURO = "#9E3B16"

# Paleta de la vista "Aciertos y errores" (matriz de confusión sobre el mapa).
COLOR_FN = "#A78BFA"   # riesgo alto no detectado (falso negativo, el error más costoso)
COLOR_FP = "#FBBF24"   # falsa alarma (falso positivo)
COLOR_VN = "#3B4759"   # acierto en riesgo bajo (neutro, no compite visualmente)

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


@st.cache_data(show_spinner=False)
def cargar_hexagonos(ruta: Path) -> pd.DataFrame:
    """Carga el tablero de hexágonos para el mapa interactivo.

    Args:
        ruta: Ruta a ``hexagonos.csv`` (coordenadas, predictoras, target
            observado y probabilidad *out-of-fold*), generado en el Paso 1.

    Returns:
        DataFrame con una fila por hexágono H3.
    """
    return pd.read_csv(ruta)


# ---------------------------------------------------------------------------
# Lógica de dominio (idéntica al modelo original; sólo cambia la presentación)
# ---------------------------------------------------------------------------
def parametros_slider(
    rango: dict[str, float],
) -> tuple[float, float, float, float, str, int]:
    """Calcula ``(min, max, valor_defecto, paso, formato, decimales)`` para un slider.

    El **valor por defecto es siempre la mediana** leída de ``metadata.json``.
    Para que Streamlit ubique bien el control, redondea extremos y mediana a una
    misma cantidad de decimales y usa un paso de ``10**-decimales``: así la
    mediana cae **exactamente sobre la grilla** del slider (si no, el frontend
    de Streamlit desplaza el control cerca del máximo). Los extremos se
    redondean hacia afuera para no recortar el rango observado. Se devuelven
    también los ``decimales`` porque se reusan para redondear valores reales
    de un hexágono (mapa) a la misma grilla.

    Args:
        rango: Entrada de ``metadata["rangos"]`` con ``min``, ``max`` y
            ``mediana``.

    Returns:
        Tupla ``(lo, hi, valor_defecto, paso, formato, decimales)`` para
        ``st.slider``.
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
    return lo, hi, valor, paso, f"%.{decimales}f", decimales


def hexagono_seleccionado() -> dict[str, Any] | None:
    """Devuelve la fila del hexágono clickeado en el mapa, si hay alguno.

    Lee ``st.session_state["mapa_hex"]``, el estado de selección de
    ``st.pydeck_chart``. Streamlit ya lo puebla con el resultado del click
    *antes* de que el script llegue a instanciar de nuevo el widget del mapa,
    así que puede leerse al principio de ``main()`` para sincronizar los
    sliders antes de dibujar la barra lateral.

    Returns:
        Diccionario con todas las columnas de ``hexagonos.csv`` para el
        hexágono seleccionado, o ``None`` si no hay selección.
    """
    estado = st.session_state.get("mapa_hex")
    if not estado:
        return None
    objetos = estado.get("selection", {}).get("objects", {}).get("hexagonos")
    return objetos[0] if objetos else None


def sincronizar_seleccion(meta: dict[str, Any], fila_hex: dict[str, Any]) -> None:
    """Escribe los valores reales de un hexágono en el estado de los sliders.

    Debe llamarse **antes** de instanciar los widgets (``recoger_entradas``):
    Streamlit no permite reasignar el valor de un widget con ``key`` después
    de haberlo dibujado en el mismo run.

    Args:
        meta: Metadata del modelo (rangos, categorías).
        fila_hex: Fila de ``hexagonos.csv`` del hexágono clickeado.
    """
    for col in meta["features_numericas"]:
        r = meta["rangos"][col]
        lo, hi, _, _, _, decimales = parametros_slider(r)
        valor = min(max(round(float(fila_hex[col]), decimales), lo), hi)
        st.session_state[f"in_{col}"] = valor

    cat = meta["feature_categorica"]
    if fila_hex[cat] in meta["categorias_cobertura"]:
        st.session_state[f"in_{cat}"] = fila_hex[cat]


def recoger_entradas(meta: dict[str, Any], hexagonos: pd.DataFrame) -> pd.DataFrame:
    """Dibuja los controles agrupados y devuelve un DataFrame de una fila.

    Los sliders se organizan en tres expanders temáticos (Clima, Relieve y
    vegetación, Accesibilidad). Cada slider está acotado al rango observado
    (imposible ingresar valores absurdos) y muestra su min/max debajo. La
    cobertura vegetal es un ``selectbox``. Todos los controles tienen
    ``key=f"in_{columna}"``: es lo que permite que ``sincronizar_seleccion``
    los sobrescriba con los valores reales de un hexágono clickeado en el mapa.

    Args:
        meta: Metadata del modelo.
        hexagonos: Tablero de hexágonos (para mostrar la ubicación del
            hexágono actualmente cargado, si hay uno).

    Returns:
        DataFrame de una fila con las 8 columnas que espera el Pipeline, en el
        orden correcto.
    """
    valores: dict[str, Any] = {}
    cat = meta["feature_categorica"]
    st.sidebar.header("Condiciones del hexágono")

    hex_actual = st.session_state.get("_hex_procesado")
    if st.session_state.get("_mostrando_hex") and hex_actual is not None:
        fila = hexagonos.loc[hexagonos["hex"] == hex_actual]
        ubic = (f" · lat {fila.lat.iloc[0]:.2f}, lon {fila.lon.iloc[0]:.2f}"
                if not fila.empty else "")
        st.sidebar.success(f"📍 Hexágono **{hex_actual}**{ubic}")
        if st.sidebar.button("↺ Volver a la mediana", use_container_width=True):
            for col in meta["features_numericas"] + [cat]:
                st.session_state.pop(f"in_{col}", None)
            st.session_state["_mostrando_hex"] = False
            # OJO: `_hex_procesado` se deja intacto a propósito. El mapa no
            # se puede "deseleccionar" por código (Streamlit no permite
            # modificar el estado de selección de pydeck), así que si lo
            # limpiáramos acá, el próximo rerun volvería a detectar "selección
            # nueva" y recargaría el mismo hexágono, anulando este botón.
            st.rerun()  # evita mostrar el banner viejo un frame de más
    else:
        st.sidebar.caption(
            "Ajustá las condiciones ambientales o hacé clic en un hexágono del "
            "mapa. Los límites son los observados en la Patagonia (2012–2023): "
            "no se pueden ingresar valores imposibles."
        )

    for grupo, columnas in GRUPOS_SLIDERS.items():
        with st.sidebar.expander(grupo, expanded=(grupo == "Clima")):
            for col in columnas:
                r = meta["rangos"][col]
                lo, hi, valor, paso, fmt, _ = parametros_slider(r)
                key = f"in_{col}"
                # `value=` sólo se pasa si la clave todavía no existe en
                # session_state: si ya existe (slider tocado a mano, o cargado
                # desde un click en el mapa), pasar ambos dispara un warning
                # de política de Streamlit y `value` quedaría ignorado igual.
                kwargs = dict(
                    min_value=lo, max_value=hi, step=paso, format=fmt, key=key,
                    help=f"Por defecto: la mediana ({fmt % r['mediana']} "
                         f"{r['unidad']}).",
                )
                if key not in st.session_state:
                    kwargs["value"] = valor
                valores[col] = st.slider(r["etiqueta"], **kwargs)
                st.caption(
                    f"mín {fmt % r['min']}  ·  máx {fmt % r['max']} "
                    f"{r['unidad']}"
                )
            # La cobertura vegetal acompaña al relieve.
            if grupo == "Relieve y vegetación":
                key_cat = f"in_{cat}"
                kwargs_cat = dict(
                    options=meta["categorias_cobertura"], key=key_cat,
                    help="Tipo de cobertura dominante del hexágono.",
                )
                if key_cat not in st.session_state:
                    # la más frecuente: notebook 09 ordena por value_counts()
                    kwargs_cat["index"] = 0
                valores[cat] = st.selectbox("Cobertura vegetal", **kwargs_cat)

    columnas = meta["features_numericas"] + [cat]
    return pd.DataFrame([valores])[columnas]


@st.cache_data(show_spinner=False)
def _inversa_covarianza(cov: list[list[float]]) -> np.ndarray:
    """Inversa (Moore-Penrose) de la covarianza de la envolvente, cacheada.

    Es la misma matriz para toda la sesión (viene de ``metadata.json``, no del
    input); recalcularla en cada movimiento de slider es trabajo repetido de
    balde.

    Args:
        cov: Matriz de covarianza de ``metadata["envolvente"]["cov"]``.

    Returns:
        Pseudo-inversa de ``cov``.
    """
    return np.linalg.pinv(np.asarray(cov, dtype=float))


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
    inv = _inversa_covarianza(env["cov"])
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


def contribuciones_locales(
    fila: pd.DataFrame, modelo: Any, meta: dict[str, Any]
) -> tuple[dict[str, float], float]:
    """Descomposición TreeSHAP exacta de la predicción para ESTA fila.

    XGBoost calcula TreeSHAP de forma nativa (``pred_contribs=True``): no hace
    falta instalar la librería ``shap`` aparte. La salida está en **log-odds**
    y es exacta, no una aproximación: ``base_value + Σ contribuciones``
    reproduce el logit de la predicción, y su sigmoide reproduce la
    probabilidad que muestra el gauge (se verifica en la app, no sólo se
    afirma).

    Las columnas one-hot de ``cobertura_veg`` se agregan a una sola
    contribución (mismo patrón que ``importancias`` en el notebook 09): sólo
    una de ellas es distinta de cero por fila, así que sumarlas no mezcla
    nada.

    Args:
        fila: DataFrame de una fila con las 8 columnas de entrada.
        modelo: Pipeline completo (``prep`` + ``model``).
        meta: Metadata del modelo.

    Returns:
        Tupla ``(contribuciones, base_value)``: diccionario de las 8 features
        originales a su contribución en log-odds, y el valor base del modelo.
    """
    prep = modelo.named_steps["prep"]
    booster = modelo.named_steps["model"].get_booster()
    nombres = list(prep.get_feature_names_out())
    Xt = prep.transform(fila)
    dmat = xgb.DMatrix(Xt, feature_names=nombres)
    fila_contribs = booster.predict(dmat, pred_contribs=True)[0]

    cat = meta["feature_categorica"]
    contribs: dict[str, float] = {}
    for c in meta["features_numericas"]:
        contribs[c] = float(fila_contribs[nombres.index(f"num__{c}")])
    contribs[cat] = float(sum(
        v for n, v in zip(nombres, fila_contribs[:-1]) if n.startswith("cat__")))
    base_value = float(fila_contribs[-1])
    return contribs, base_value


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

    El umbral se marca con una línea sobre el arco (``threshold``) y con el
    sombreado verde/rojo de los tramos (``steps``); a propósito **no** lleva
    un texto superpuesto dentro de la figura. Plotly posiciona el número
    grande (``mode="gauge+number"``) en la zona baja-central del semicírculo,
    que es el mismo lugar donde iría cualquier anotación de texto agregada
    ahí — con probabilidades altas (número de 3 dígitos) los dos se pisan.
    Por eso el texto del umbral se agrega afuera, como ``st.caption`` en
    ``main()``.

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
    return _layout_oscuro(fig, altura=300)


def grafico_histograma_focos(
    meta: dict[str, Any], es_alto: bool, n_focos_real: float | None = None
) -> go.Figure:
    """Histograma de ``n_focos`` (escala log) con la zona estimada resaltada.

    Es la distribución histórica **completa y fija** de los 1981 hexágonos: no
    cambia con los sliders. Lo único que sí cambia con la clasificación actual
    es qué mitad se sombrea, porque el modelo no predice un número de focos,
    sólo el **lado** del umbral (150) en que caería el hexágono.

    Si se pasa ``n_focos_real`` —el hexágono viene de un click en el mapa, no
    de una combinación manual de sliders— se marca además con una línea sólida
    el valor **observado** de ese hexágono puntual. Es la única referencia
    concreta que puede dibujarse: para una combinación hipotética de sliders
    no existe un ``n_focos`` real que marcar.

    A propósito, la figura **no lleva texto superpuesto**: sobre un histograma
    con barras de distinta altura, cualquier anotación cae sobre alguna barra
    en algún punto y queda ilegible. El significado de cada línea (umbral,
    zona sombreada, hexágono actual) se explica en el texto que ``main()``
    dibuja debajo del gráfico, fuera de la figura.

    Args:
        meta: Metadata con ``distribucion_n_focos`` y ``umbral_riesgo_focos``.
        es_alto: Si la clasificación estimada es riesgo alto.
        n_focos_real: ``n_focos`` observado del hexágono actual, si viene del
            mapa. ``None`` si los sliders son una combinación manual.

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

    if n_focos_real is not None and n_focos_real > 0:
        log_real = math.log10(n_focos_real)
        fig.add_vline(x=log_real, line_dash="solid", line_color=COLOR_TEXTO,
                      line_width=2)

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


def grafico_contribuciones(contribs: dict[str, float], meta: dict[str, Any]) -> go.Figure:
    """Barras divergentes con la contribución de cada feature a ESTA predicción.

    A diferencia de ``grafico_importancias`` (global, igual para cualquier
    input), esto cambia con cada combinación de sliders: es la respuesta a
    "¿qué pesó en este caso puntual?".

    Args:
        contribs: Salida de ``contribuciones_locales`` (log-odds por feature).
        meta: Metadata del modelo (para las etiquetas legibles).

    Returns:
        Figura Plotly.
    """
    etiquetas = dict(meta["etiquetas"])
    etiquetas[meta["feature_categorica"]] = "Cobertura vegetal"
    items = sorted(contribs.items(), key=lambda kv: abs(kv[1]))
    nombres = [etiquetas.get(k, k) for k, _ in items]
    valores = [v for _, v in items]
    colores = [COLOR_ALTO if v >= 0 else COLOR_BAJO for v in valores]

    fig = go.Figure(go.Bar(
        x=valores, y=nombres, orientation="h",
        marker_color=colores, marker_line_width=0,
        hovertemplate="%{y}: %{x:+.3f} (log-odds)<extra></extra>",
    ))
    fig.add_vline(x=0, line_color=COLOR_TENUE, line_width=1)
    fig.update_xaxes(title_text="Contribución a esta predicción (log-odds)",
                     gridcolor="rgba(255,255,255,0.06)")
    fig.update_yaxes(gridcolor="rgba(255,255,255,0.0)")
    return _layout_oscuro(fig, altura=320)


# ---------------------------------------------------------------------------
# Mapa H3 interactivo (pydeck)
# ---------------------------------------------------------------------------
def _hex_a_rgb(color_hex: str) -> tuple[int, int, int]:
    """Convierte ``"#RRGGBB"`` a una tupla ``(r, g, b)``."""
    color_hex = color_hex.lstrip("#")
    return tuple(int(color_hex[i:i + 2], 16) for i in (0, 2, 4))


def _interpolar_rgb(
    c0: tuple[int, int, int], c1: tuple[int, int, int], t: float
) -> list[int]:
    """Interpola linealmente entre dos colores RGB en ``t ∈ [0, 1]``."""
    return [int(c0[i] + (c1[i] - c0[i]) * t) for i in range(3)]


def colores_por_probabilidad(prob: pd.Series) -> list[list[int]]:
    """Gradiente verde→rojo por probabilidad estimada (out-of-fold), 0 a 1.

    Misma paleta que el resto de la app (``COLOR_BAJO`` → ``COLOR_ALTO``), para
    que "riesgo alto" signifique el mismo color en el mapa, el gauge y los KPIs.
    """
    c0, c1 = _hex_a_rgb(COLOR_BAJO), _hex_a_rgb(COLOR_ALTO)
    t = prob.clip(0, 1)
    return [[*_interpolar_rgb(c0, c1, v), 190] for v in t]


def colores_por_focos(n_focos: pd.Series) -> list[list[int]]:
    """Gradiente crema→naranja→marrón por ``n_focos`` en escala logarítmica.

    Misma paleta y misma razón que en
    ``knime/presentacion/generar_mapa_h3.py``: la distribución de focos está
    muy sesgada, así que el log resalta el gradiente real de actividad.
    """
    c0 = _hex_a_rgb(COLOR_FOCOS_CLARO)
    c1 = _hex_a_rgb(COLOR_FOCOS_MEDIO)
    c2 = _hex_a_rgb(COLOR_FOCOS_OSCURO)
    log_f = np.log10(n_focos.clip(lower=1))
    rango = log_f.max() - log_f.min()
    t = ((log_f - log_f.min()) / rango) if rango > 0 else log_f * 0
    colores = []
    for v in t.fillna(0.0):
        if v < 0.5:
            colores.append([*_interpolar_rgb(c0, c1, v / 0.5), 190])
        else:
            colores.append([*_interpolar_rgb(c1, c2, (v - 0.5) / 0.5), 190])
    return colores


def colores_por_confusion(
    df: pd.DataFrame, umbral: float
) -> tuple[list[list[int]], pd.Series]:
    """Colorea cada hexágono según el tipo de acierto/error del modelo.

    Compara ``proba_oof >= umbral`` (lo que el modelo habría dicho, honesto,
    sin haber visto ese hexágono en entrenamiento) contra ``riesgo_alto``
    observado. Es la vista más defendible del mapa: hace **geográfica** la
    precisión de 0.46 en vez de dejarla como un número suelto en una tabla.

    Args:
        df: Tablero de hexágonos con ``proba_oof`` y ``riesgo_alto``.
        umbral: Umbral de decisión.

    Returns:
        Tupla ``(colores, categoria)``: lista de colores RGBA y la serie de
        categorías (``"VP"``, ``"FN"``, ``"FP"``, ``"VN"``).
    """
    pred_alto = df["proba_oof"] >= umbral
    obs_alto = df["riesgo_alto"].astype(bool)
    categoria = pd.Series(
        np.select(
            [obs_alto & pred_alto, obs_alto & ~pred_alto, ~obs_alto & pred_alto],
            ["VP", "FN", "FP"], default="VN",
        ),
        index=df.index,
    )
    paleta = {
        "VP": [*_hex_a_rgb(COLOR_ALTO), 210],
        "FN": [*_hex_a_rgb(COLOR_FN), 210],
        "FP": [*_hex_a_rgb(COLOR_FP), 210],
        "VN": [*_hex_a_rgb(COLOR_VN), 140],
    }
    return [paleta[c] for c in categoria], categoria


VISTAS_MAPA = {
    "Probabilidad estimada": (
        "Verde = baja probabilidad estimada · rojo = alta. Probabilidad "
        "**out-of-fold** (honesta): no es la del modelo de producción, que ya "
        "vio estos hexágonos al entrenar."
    ),
    "Focos observados": (
        "Escala logarítmica: crema = pocos focos históricos (2012–2023) · "
        "marrón = muchos."
    ),
    "Aciertos y errores": (
        "Rojo = acierto en riesgo alto · violeta = riesgo alto no detectado "
        "(el error más costoso) · ámbar = falsa alarma · gris = acierto en "
        "riesgo bajo."
    ),
}


def grafico_mapa_hexagonos(
    df: pd.DataFrame, vista: str, umbral: float, hex_resaltado: str | None
) -> pdk.Deck:
    """Arma el ``pdk.Deck`` del mapa H3 según la vista elegida.

    Args:
        df: Tablero de hexágonos completo.
        vista: Una de las claves de ``VISTAS_MAPA``.
        umbral: Umbral de decisión (para la vista de aciertos/errores).
        hex_resaltado: Id del hexágono actualmente cargado en los sliders (se
            dibuja con un borde resaltado), o ``None``.

    Returns:
        ``pdk.Deck`` listo para ``st.pydeck_chart``.
    """
    d = df.copy()
    if vista == "Focos observados":
        d["color"] = colores_por_focos(d["n_focos"])
    elif vista == "Aciertos y errores":
        d["color"], _ = colores_por_confusion(d, umbral)
    else:
        d["color"] = colores_por_probabilidad(d["proba_oof"])

    d["prob_pct"] = (d["proba_oof"] * 100).round(1).astype(str) + " %"
    d["n_focos_str"] = d["n_focos"].round().astype(int).astype(str)
    d["clase_obs"] = np.where(d["riesgo_alto"] == 1,
                              "ALTO (observado)", "BAJO (observado)")

    capas = [pdk.Layer(
        "H3HexagonLayer", data=d, get_hexagon="hex", get_fill_color="color",
        get_line_color=[11, 17, 32], line_width_min_pixels=0.5,
        pickable=True, auto_highlight=True,
        highlight_color=[255, 255, 255, 90], id="hexagonos",
    )]
    if hex_resaltado:
        resaltado = d[d["hex"] == hex_resaltado]
        if not resaltado.empty:
            capas.append(pdk.Layer(
                "H3HexagonLayer", data=resaltado, get_hexagon="hex",
                get_fill_color=[0, 0, 0, 0], get_line_color=[255, 255, 255, 255],
                line_width_min_pixels=3, stroked=True, filled=False,
                pickable=False, id="seleccionado",
            ))

    vista_inicial = pdk.ViewState(
        latitude=float(df["lat"].mean()), longitude=float(df["lon"].mean()),
        zoom=4.0, pitch=0,
    )
    return pdk.Deck(
        layers=capas, initial_view_state=vista_inicial,
        map_provider="carto", map_style="dark",
        tooltip={
            "html": "<b>{hex}</b><br/>Focos históricos: {n_focos_str}<br/>"
                    "Prob. estimada (OOF): {prob_pct}<br/>Observado: {clase_obs}",
            "style": {"backgroundColor": COLOR_PANEL, "color": COLOR_TEXTO,
                      "fontSize": "12px"},
        },
    )


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

    if (not RUTA_MODELO.exists() or not RUTA_METADATA.exists()
            or not RUTA_HEXAGONOS.exists()):
        st.error(
            "No se encontraron los artefactos del modelo en `app/models/`. "
            "Ejecutá primero el notebook `notebooks/09_serializacion_tp4.ipynb`."
        )
        st.stop()

    modelo = cargar_modelo(RUTA_MODELO)
    meta = cargar_metadata(RUTA_METADATA)
    hexagonos = cargar_hexagonos(RUTA_HEXAGONOS)
    umbral = float(meta["umbral_decision"])
    met = meta["metricas"]

    # --- Sincronizar la selección del mapa con los sliders, ANTES de dibujar
    #     la barra lateral: Streamlit no permite reasignar el estado de un
    #     widget con `key` después de haberlo instanciado en el mismo run. ---
    fila_sel = hexagono_seleccionado()
    sel_id = fila_sel["hex"] if fila_sel is not None else None
    if sel_id is not None and sel_id != st.session_state.get("_hex_procesado"):
        sincronizar_seleccion(meta, fila_sel)
        st.session_state["_hex_procesado"] = sel_id
        st.session_state["_mostrando_hex"] = True

    # --- Entrada (barra lateral) ---
    entrada = recoger_entradas(meta, hexagonos)

    # --- Predicción (lógica intacta respecto del diseño anterior) ---
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

    # === Mapa de hexágonos ===
    st.markdown('<div class="seccion">Mapa de hexágonos</div>',
                unsafe_allow_html=True)
    vista = st.radio(
        "Colorear por", list(VISTAS_MAPA), horizontal=True,
        label_visibility="collapsed",
    )
    deck = grafico_mapa_hexagonos(
        hexagonos, vista, umbral, st.session_state.get("_hex_procesado"))
    st.pydeck_chart(deck, on_select="rerun", selection_mode="single-object",
                    key="mapa_hex", height=430)
    st.caption(
        VISTAS_MAPA[vista] + " Hacé clic en un hexágono para cargar sus "
        "condiciones reales en los sliders de la izquierda."
    )

    st.write("")

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
        tarjeta_kpi(
            "Umbral de decisión", f"{umbral:.3f}",
            f"recall {met['recall_en_umbral']:.2f} · precisión "
            f"{met['precision_en_umbral']:.2f}",
            color=COLOR_TEXTO),
        unsafe_allow_html=True)
    badge = (("ATÍPICO", "badge-warn") if atipico else ("OK", "badge-ok"))
    k4.markdown(
        tarjeta_kpi("Distancia a la envolvente", f"{dist:.2f}",
                    f"corte {cutoff:.2f} ·", color=COLOR_TEXTO, badge=badge),
        unsafe_allow_html=True)

    st.write("")

    # === Fila 2: gauge + histograma de contexto ===
    # Si el hexágono actual viene de un click en el mapa, tenemos su n_focos
    # observado y podemos marcarlo en el histograma. Si son sliders manuales
    # (combinación hipotética), no existe un n_focos real que marcar.
    n_focos_real = None
    if st.session_state.get("_mostrando_hex"):
        fila_hex_actual = hexagonos.loc[
            hexagonos["hex"] == st.session_state.get("_hex_procesado")]
        if not fila_hex_actual.empty:
            n_focos_real = float(fila_hex_actual["n_focos"].iloc[0])

    c1, c2 = st.columns(2)
    with c1:
        st.markdown('<div class="seccion">Probabilidad</div>',
                    unsafe_allow_html=True)
        st.plotly_chart(grafico_gauge(prob, umbral, es_alto),
                        use_container_width=True)
        st.caption(f"Umbral de decisión: {umbral * 100:.1f} %  ·  verde por "
                   f"debajo, rojo por encima.")
    with c2:
        st.markdown('<div class="seccion">Contexto histórico</div>',
                    unsafe_allow_html=True)
        st.plotly_chart(grafico_histograma_focos(meta, es_alto, n_focos_real),
                        use_container_width=True)
        etiqueta_zona = "RIESGO ALTO" if es_alto else "RIESGO BAJO"
        umbral_focos = int(meta["umbral_riesgo_focos"])
        leyenda = (
            f"Distribución histórica <b>fija</b> (no cambia con los sliders) "
            f"· línea punteada roja: umbral de {umbral_focos} focos · franja "
            f"sombreada: lado del umbral de la <b style='color:{color_res}'>"
            f"{etiqueta_zona}</b> estimada"
        )
        if n_focos_real is not None:
            leyenda += (
                f" · línea sólida blanca: focos <b>observados</b> de este "
                f"hexágono ({int(round(n_focos_real))})"
            )
        st.markdown(f"<span class='kpi-sub'>{leyenda}.</span>",
                    unsafe_allow_html=True)

    # === Fila 3: explicación local (SHAP) + interpretación ===
    c3, c4 = st.columns(2)
    with c3:
        st.markdown('<div class="seccion">Por qué esta estimación</div>',
                    unsafe_allow_html=True)
        contribs, base_value = contribuciones_locales(entrada, modelo, meta)
        st.plotly_chart(grafico_contribuciones(contribs, meta),
                        use_container_width=True)
        suma_contribs = sum(contribs.values())
        logit = base_value + suma_contribs
        prob_verificada = 1.0 / (1.0 + math.exp(-logit))
        st.caption(
            f"Contribuciones en log-odds (TreeSHAP exacto de XGBoost, no una "
            f"aproximación). Verificación: base ({base_value:+.3f}) + Σ "
            f"contribuciones ({suma_contribs:+.3f}) = {logit:+.3f} → sigmoide "
            f"= {prob_verificada * 100:.1f} %, igual a la probabilidad de arriba."
        )
        with st.expander("Ver importancia global del modelo"):
            st.caption(
                "A diferencia del gráfico de arriba, esto es fijo: no cambia "
                "al mover los sliders. Resume qué pesa el modelo **en "
                "promedio**, sobre todas las predicciones."
            )
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
            "- **Probabilidades del mapa, out-of-fold.** El color de cada "
            "hexágono en el mapa sale de validación cruzada (5 folds), no del "
            "modelo de producción (reentrenado con el 100 % de los datos). Al "
            "hacer clic en un hexágono, la probabilidad del gauge puede diferir "
            "levemente de la del mapa: es esperable, no un error.\n"
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
