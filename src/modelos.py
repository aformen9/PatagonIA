# -*- coding: utf-8 -*-
"""Utilidades de modelado para la regresión de `n_focos` (TP2).

Centraliza el preprocesamiento (imputación + escalado + one-hot) dentro de un
``Pipeline`` de scikit-learn, de modo que toda transformación se aprenda sólo
con el fold de entrenamiento y nunca haya fuga de información hacia validación
o test. Provee además el envoltorio para la formulación en escala logarítmica
(``log1p`` / ``expm1``) y el cálculo unificado de métricas.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer, TransformedTargetRegressor
from sklearn.impute import SimpleImputer
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    cohen_kappa_score,
    f1_score,
    mean_absolute_error,
    mean_squared_error,
    precision_score,
    r2_score,
    recall_score,
    roc_auc_score,
)
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

# Predictoras numéricas (7) y categórica (1) del TP.
NUMERICAS = [
    "elevacion",
    "temp_media",
    "precip_anual",
    "viento_medio",
    "humedad_relativa",
    "dist_asentamiento_km",
    "dist_ruta_km",
]
CATEGORICA = "cobertura_veg"


def construir_preprocesador() -> ColumnTransformer:
    """Crea el preprocesador de features.

    - Numéricas: imputación por **mediana** (cubre el único nulo de
      ``elevacion``) seguida de estandarización (``StandardScaler``).
    - Categórica: ``OneHotEncoder`` sobre ``cobertura_veg``.

    Al vivir dentro del ``Pipeline``, la mediana y las medias/desvíos del
    escalado se estiman sólo con datos de entrenamiento en cada fold.

    Returns:
        ``ColumnTransformer`` listo para anteponer a cualquier estimador.
    """
    num = Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", StandardScaler()),
    ])
    cat = Pipeline([
        ("onehot", OneHotEncoder(handle_unknown="ignore")),
    ])
    return ColumnTransformer([
        ("num", num, NUMERICAS),
        ("cat", cat, [CATEGORICA]),
    ])


def construir_pipeline(estimador, log1p: bool = False):
    """Ensambla preprocesador + estimador, opcionalmente en escala log.

    Args:
        estimador: Regresor de scikit-learn (o compatible).
        log1p: Si ``True``, envuelve el pipeline en un
            ``TransformedTargetRegressor`` que entrena sobre ``log1p(y)`` y
            devuelve predicciones ya en la escala original vía ``expm1``.

    Returns:
        Un ``Pipeline`` (crudo) o un ``TransformedTargetRegressor`` (log).
    """
    pipe = Pipeline([
        ("prep", construir_preprocesador()),
        ("model", estimador),
    ])
    if log1p:
        return TransformedTargetRegressor(
            regressor=pipe, func=np.log1p, inverse_func=np.expm1
        )
    return pipe


def prefijo_param(log1p: bool) -> str:
    """Devuelve el prefijo de parámetros del estimador para ``GridSearchCV``.

    Args:
        log1p: Si el pipeline está envuelto en ``TransformedTargetRegressor``.

    Returns:
        ``"regressor__model__"`` si es log, ``"model__"`` si es crudo.
    """
    return "regressor__model__" if log1p else "model__"


def calcular_metricas(y_true, y_pred) -> dict:
    """Calcula MSE, RMSE, MAE y R² en la escala provista.

    Args:
        y_true: Valores observados (escala original de ``n_focos``).
        y_pred: Predicciones (misma escala que ``y_true``).

    Returns:
        Diccionario con las claves ``MSE``, ``RMSE``, ``MAE`` y ``R2``.
    """
    mse = mean_squared_error(y_true, y_pred)
    return {
        "MSE": mse,
        "RMSE": float(np.sqrt(mse)),
        "MAE": mean_absolute_error(y_true, y_pred),
        "R2": r2_score(y_true, y_pred),
    }


def evaluar_modelo(modelo, X_train, y_train, X_test, y_test) -> dict:
    """Entrena el modelo y calcula métricas en train y test.

    Las métricas se computan **siempre en la escala original** de ``n_focos``.
    Para la formulación log, ``TransformedTargetRegressor.predict`` ya aplica
    ``expm1``, por lo que la comparación entre formulaciones es directa.

    Args:
        modelo: Pipeline/estimador (ya construido, sin entrenar).
        X_train, y_train: Partición de entrenamiento.
        X_test, y_test: Partición de prueba.

    Returns:
        Diccionario plano con métricas ``train_*`` y ``test_*`` y el modelo
        entrenado bajo la clave ``_modelo``.
    """
    modelo.fit(X_train, y_train)
    m_tr = calcular_metricas(y_train, modelo.predict(X_train))
    m_te = calcular_metricas(y_test, modelo.predict(X_test))
    fila = {f"train_{k}": v for k, v in m_tr.items()}
    fila.update({f"test_{k}": v for k, v in m_te.items()})
    fila["_modelo"] = modelo
    return fila


# ---------------------------------------------------------------------------
# Clasificación (TP3): riesgo_alto = (n_focos > 150)
# ---------------------------------------------------------------------------
# Se reutiliza EXACTAMENTE el mismo preprocesador que en regresión
# (``construir_preprocesador``): imputación por mediana + escalado de las 7
# numéricas y one-hot de ``cobertura_veg``, todo dentro del ``Pipeline`` para
# que se aprenda sólo con el fold de entrenamiento. `n_focos` y el resto de
# variables de fuego NO forman parte de las features: el target se deriva de
# `n_focos`, así que incluirlo sería fuga directa.


def construir_pipeline_clf(estimador) -> Pipeline:
    """Ensambla preprocesador + clasificador (sin transformar el target).

    Análogo a ``construir_pipeline`` pero para clasificación: no hay
    ``TransformedTargetRegressor`` porque el objetivo es binario.

    Args:
        estimador: Clasificador de scikit-learn (o compatible con su API).

    Returns:
        ``Pipeline`` con pasos ``prep`` (preprocesamiento) y ``model``.
    """
    return Pipeline([
        ("prep", construir_preprocesador()),
        ("model", estimador),
    ])


def calcular_metricas_clf(y_true, y_pred, y_score=None) -> dict:
    """Calcula el panel de métricas de clasificación binaria.

    Métricas obligatorias del TP: accuracy, precision/recall/F1 tanto de la
    **clase positiva** (``riesgo_alto=1``, la de interés) como en promedio
    **weighted** (pondera por soporte), y el Kappa de Cohen. Si se pasa un
    score continuo de la clase positiva, agrega ROC-AUC y AUC-PR (average
    precision), las métricas robustas al desbalance.

    Args:
        y_true: Etiquetas observadas (0/1).
        y_pred: Etiquetas predichas (0/1).
        y_score: Score/probabilidad de la clase positiva. Opcional.

    Returns:
        Diccionario de métricas escalares.
    """
    m = {
        "accuracy": accuracy_score(y_true, y_pred),
        "precision_pos": precision_score(y_true, y_pred, pos_label=1,
                                         zero_division=0),
        "recall_pos": recall_score(y_true, y_pred, pos_label=1,
                                   zero_division=0),
        "f1_pos": f1_score(y_true, y_pred, pos_label=1, zero_division=0),
        "precision_w": precision_score(y_true, y_pred, average="weighted",
                                       zero_division=0),
        "recall_w": recall_score(y_true, y_pred, average="weighted",
                                 zero_division=0),
        "f1_w": f1_score(y_true, y_pred, average="weighted", zero_division=0),
        "kappa": cohen_kappa_score(y_true, y_pred),
    }
    if y_score is not None:
        m["roc_auc"] = roc_auc_score(y_true, y_score)
        m["ap"] = average_precision_score(y_true, y_score)  # AUC-PR
    return m


def score_positivo(modelo, X):
    """Devuelve el score continuo de la clase positiva.

    Usa ``predict_proba[:, 1]`` si el estimador la expone; si no, recurre a
    ``decision_function`` (p. ej. un SVC sin ``probability``). Devuelve
    ``None`` si el modelo no ofrece ninguna (p. ej. ``DummyClassifier`` en
    modos sin probabilidad útil, aunque éste sí expone ``predict_proba``).

    Args:
        modelo: Clasificador entrenado.
        X: Matriz de features.

    Returns:
        Array de scores de la clase positiva, o ``None``.
    """
    if hasattr(modelo, "predict_proba"):
        return modelo.predict_proba(X)[:, 1]
    if hasattr(modelo, "decision_function"):
        return modelo.decision_function(X)
    return None


def evaluar_clasificador(modelo, X_train, y_train, X_test, y_test) -> dict:
    """Entrena el clasificador y calcula el panel de métricas en train y test.

    Reportar train y test permite leer el sobreajuste de forma explícita
    (p. ej. un árbol o KNN con recall_pos perfecto en train y pobre en test).

    Args:
        modelo: Pipeline/estimador de clasificación (sin entrenar).
        X_train, y_train: Partición de entrenamiento.
        X_test, y_test: Partición de prueba.

    Returns:
        Diccionario plano con métricas ``train_*`` y ``test_*`` y el modelo
        entrenado bajo la clave ``_modelo``.
    """
    modelo.fit(X_train, y_train)
    s_tr = score_positivo(modelo, X_train)
    s_te = score_positivo(modelo, X_test)
    m_tr = calcular_metricas_clf(y_train, modelo.predict(X_train), s_tr)
    m_te = calcular_metricas_clf(y_test, modelo.predict(X_test), s_te)
    fila = {f"train_{k}": v for k, v in m_tr.items()}
    fila.update({f"test_{k}": v for k, v in m_te.items()})
    fila["_modelo"] = modelo
    return fila
