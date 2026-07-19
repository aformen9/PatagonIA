# PatagonIA — App de estimación de riesgo de incendio (TP4)

Aplicación **Streamlit** que estima si un hexágono de la Patagonia es de
**riesgo alto de incendio** a partir de 8 predictoras ambientales.

## Qué hace

- Recibe las condiciones de un hexágono (7 variables numéricas + cobertura
  vegetal) mediante sliders y un menú, todos acotados al rango real observado.
- Devuelve la **clasificación** (riesgo ALTO / BAJO), la **probabilidad**
  estimada con barra de progreso y una **explicación en lenguaje llano**.
- **Valida** la entrada: los sliders impiden valores absurdos y un aviso
  detecta combinaciones fuera de la envolvente del dataset (distancia de
  Mahalanobis), como precipitación alta con humedad baja.
- Muestra **contexto**: histograma histórico de focos con la zona estimada,
  importancia de cada variable y una sección de **limitaciones**.

## Qué modelo usa

Un `XGBClassifier` (gradient boosting) — el mejor clasificador del TP3, con sus
hiperparámetros tuneados — **reentrenado sobre el dataset completo** (1981
hexágonos H3, datos 2012–2023 ya sin el cuadrante de *flaring* de Vaca Muerta).
El objetivo es `riesgo_alto = (n_focos > 150)`. Todo el preprocesamiento
(imputación + escalado + one-hot) viaja **dentro** del mismo `Pipeline`.

El modelo se sirve desde dos artefactos en `app/models/`, generados por
`notebooks/09_serializacion_tp4.ipynb`:

| Archivo | Contenido |
|---|---|
| `clasificador_riesgo.joblib` | El `Pipeline` completo serializado con joblib. |
| `metadata.json` | Rangos de las predictoras, categorías, umbral de decisión (0.386), métricas de referencia, importancias, envolvente de validación y la distribución de `n_focos`. |

La app **no re-entrena** ni necesita el CSV crudo: sólo carga esos dos archivos.

## Cómo correrla en local

Desde la **raíz del proyecto** (`PatagonIA/`):

```bash
# 1. (Opcional) entorno virtual
python -m venv .venv
# Windows:  .venv\Scripts\activate
# Linux/Mac: source .venv/bin/activate

# 2. Dependencias de la app (versiones pineadas)
pip install -r app/requirements.txt

# 3. Verificar que existan los artefactos del modelo
#    (si faltan, ejecutar primero el notebook del Paso 1)
#    -> app/models/clasificador_riesgo.joblib
#    -> app/models/metadata.json

# 4. Lanzar la app
streamlit run app/app.py
```

Se abre en el navegador (por defecto `http://localhost:8501`).

> **Importante:** las versiones de `scikit-learn` y `xgboost` deben coincidir
> con las usadas al serializar el modelo (ver `app/requirements.txt`); versiones
> distintas pueden fallar al deserializar el `Pipeline`.

## Limitaciones

Es un trabajo académico, **no un sistema operativo de alerta**. Estima
propensión estructural sobre datos agregados 2012–2023 y sólo cubre hexágonos
con actividad de fuego previa. Ver la sección *Limitaciones* dentro de la app.
