# PatagonIA — IA y Aprendizaje Automático 1

**Trabajos Prácticos TP1–TP4**  
IA y Aprendizaje Automático 1 · Licenciatura en Ciencias de Datos · UCA Rosario · 2026

---

## Separación con la materia de Minería

Este trabajo usa el mismo dataset que *Minería de Datos y Big Data*, pero con enfoque supervisado:

- **En Minería:** descubrimiento de patrones no supervisado (K-Means + Apriori en KNIME).
  La variable `foco_presente` **no entra como input**.

- **En IA:** `foco_presente` es el label de entrenamiento para clasificación binaria.
  Los clusters generados en Minería pueden usarse como feature adicional
  (*unsupervised pre-training como feature engineering*) — esto no constituye leakage
  porque el clustering se hizo sin ver el target.

---

## Pregunta analítica

> ¿Es posible predecir con 7 días de anticipación si una celda de 25km×25km
> en la Patagonia Argentina tendrá al menos un foco de incendio,
> usando exclusivamente variables climáticas, topográficas y de cobertura vegetal?

---

## Estructura de TPs

| TP | Período | Contenido | Notebook |
|----|---------|-----------|----------|
| TP1 | 2–15 jun | EDA y preparación de datos | `05_eda_tp1.ipynb` |
| TP2 | 16–22 jun | Regresión (Ridge, RF, XGBoost) | `06_regresion_tp2.ipynb` |
| TP3 | 23–29 jun | Clasificación + Clustering + Ensamble | `07_clasificacion_tp3.ipynb` |
| TP4 | 30 jun–6 jul | Deploy Streamlit + Paper IEEE | `08_deploy_tp4.ipynb` |

---

## Decisiones metodológicas

- **Split temporal estricto:** train < 2022, test ≥ 2022. Nunca aleatorio.
- **Anti-leakage en rolling windows:** todas las ventanas temporales aplican `shift(1)`
  sobre la celda antes del cálculo.
- **Desbalance de clases (~90/10):** tratado con `class_weight='balanced'`.
  Métricas: F1, AUC-ROC, AUC-PR. Accuracy descartada.
- **Pipeline sklearn obligatorio:** scaler + encoder + modelo en un único objeto
  serializado con joblib.

---

## App

Disponible en: `https://TU_APP.streamlit.app` *(julio 2026)*

Permite ingresar condiciones climáticas y topográficas de una celda patagónica
y obtener una predicción de riesgo con los factores más influyentes.

---

## Dataset de entrada

**Archivo:** `../data/processed/patagonia_dataset.csv`  
Generado por los notebooks en `../pipeline/` — ver README principal.
