# PatagonIA — App de estimación de riesgo de incendio (TP4)

Aplicación **Streamlit** que estima si un hexágono de la Patagonia es de
**riesgo alto de incendio** a partir de 8 predictoras ambientales.

## Qué hace

- **Mapa H3 interactivo** (`pydeck`) con los 1981 hexágonos, en tres vistas:
  probabilidad estimada (out-of-fold), focos observados 2012–2023 (escala log)
  y aciertos/errores del modelo (VP / FN / FP / VN) contra el umbral de
  decisión. Hacer clic en un hexágono carga sus condiciones **reales** en los
  sliders; un botón permite volver a los valores por defecto (la mediana).
- Recibe las condiciones de un hexágono (7 variables numéricas + cobertura
  vegetal) mediante sliders y un menú, todos acotados al rango real observado.
- Devuelve la **clasificación** (riesgo ALTO / BAJO), la **probabilidad**
  estimada y una **explicación en lenguaje llano**.
- **Explica cada predicción**: descomposición TreeSHAP (nativa de XGBoost, sin
  dependencias extra) de las 8 features para el hexágono actual — no sólo la
  importancia global, sino qué pesó *en este caso puntual*. Incluye la
  verificación explícita de que `base + Σ contribuciones` reproduce la
  probabilidad mostrada.
- **Valida** la entrada: los sliders impiden valores absurdos y un aviso
  detecta combinaciones fuera de la envolvente del dataset (distancia de
  Mahalanobis), como precipitación alta con humedad baja.
- Muestra **contexto**: histograma histórico de focos con la zona estimada,
  importancia global de cada variable y una sección de **limitaciones**.

## Qué modelo usa

Un `XGBClassifier` (gradient boosting) — el mejor clasificador del TP3, con sus
hiperparámetros tuneados — **reentrenado sobre el dataset completo** (1981
hexágonos H3, datos 2012–2023 ya sin el cuadrante de *flaring* de Vaca Muerta).
El objetivo es `riesgo_alto = (n_focos > 150)`. Todo el preprocesamiento
(imputación + escalado + one-hot) viaja **dentro** del mismo `Pipeline`.

El modelo se sirve desde tres artefactos en `app/models/`, generados por
`notebooks/09_serializacion_tp4.ipynb`:

| Archivo | Contenido |
|---|---|
| `clasificador_riesgo.joblib` | El `Pipeline` completo serializado con joblib. |
| `metadata.json` | Rangos de las predictoras, categorías, umbral de decisión (0.386), métricas de referencia, importancias, envolvente de validación y la distribución de `n_focos`. |
| `hexagonos.csv` | Una fila por hexágono: coordenadas, las 8 predictoras, `n_focos`/`riesgo_alto` observados y la probabilidad **out-of-fold** (`proba_oof`, de validación cruzada). Alimenta el mapa. |

La app **no re-entrena** ni necesita el dataset crudo: sólo carga esos tres
archivos.

> **Por qué `proba_oof` y no la probabilidad del modelo final:** el `Pipeline`
> de producción se entrenó sobre las 1981 filas completas, así que evaluarlo
> sobre esas mismas filas sería resustitución (el mapa se vería
> artificialmente preciso). `proba_oof` es la probabilidad de cada hexágono
> cuando **no** formaba parte del fold de entrenamiento — la única lectura
> honesta para colorear el mapa, sobre todo en la vista de aciertos/errores.
> Por eso, al hacer clic en un hexágono, la probabilidad del gauge puede
> diferir levemente de la que muestra el mapa: es esperable, no un bug.

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
#    -> app/models/hexagonos.csv

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
