# PatagonIA — Minería de Datos y Big Data

**Trabajo Integrador Final**  
Minería de Datos y Big Data · Licenciatura en Ciencias de Datos · UCA Rosario · 2026

---

## Separación con la materia de IA

Este trabajo usa el mismo dataset que la materia *IA y Aprendizaje Automático 1*,
pero con un enfoque completamente distinto:

- **En IA:** el dataset se usa para entrenar modelos supervisados que *predicen* focos a 7 días.
  La variable `foco_presente` es el label de entrenamiento.

- **En Minería:** el dataset entra como input congelado. La variable `foco_presente`
  **no entra como input de ningún modelo** — solo se usa a posteriori para validar
  la interpretación de los clusters. El objetivo es *descubrir patrones*, no predecir.

Esta distinción fue aprobada explícitamente por la cátedra.

---

## Pregunta analítica

> ¿Existen perfiles ambientales recurrentes en la Patagonia que concentren
> desproporcionadamente la ocurrencia de focos de incendio, y qué combinaciones
> de condiciones climáticas y de cobertura vegetal los caracterizan?

---

## Técnicas aplicadas

### K-Means (clustering)
- **Qué responde:** ¿qué celdas son ambientalmente similares?
- **Input:** variables continuas normalizadas (temperatura, humedad, viento, precipitación, elevación)
- **`tipo_vegetacion`:** no entra como input — se usa para interpretar los clusters resultantes
- **Selección de k:** Elbow Method + Silhouette Score
- **Herramienta:** KNIME → nodo *K-Means*

### Apriori (reglas de asociación)
- **Qué responde:** ¿qué combinaciones de condiciones aparecen juntas en celdas con focos?
- **Input:** variables discretizadas con umbrales físicamente justificados
- **Métricas de filtrado:** soporte, confianza, lift (se descartan reglas con lift < 2)
- **Herramienta:** KNIME → nodo *Association Rule Learner*

---

## Pipeline KNIME

```
[CSV Reader]
    → [Column Filter]          elimina lat/lon/celda_id del análisis
    → [Missing Value]          imputación median (continuas) / mode (categóricas)
    → [Normalizer Z-Score]     obligatorio antes de K-Means
    → [Category to Number]     tipo_vegetacion → numérico
          ↓
    ┌─────────────────────────────────────┐
    │ rama A                              │
    │ [K-Means]                           │
    │ [Color Manager]                     │
    │ [Scatter Plot / Geographic Map]     │
    └─────────────────────────────────────┘
          ↓
    ┌─────────────────────────────────────┐
    │ rama B                              │
    │ [Numeric Binner]                    │
    │ [One to Many]                       │
    │ [Association Rule Learner (Apriori)]│
    │ [Rule Filter por lift ≥ 2]          │
    └─────────────────────────────────────┘
```

El workflow completo exportado está en `knime/pipeline_patagonia.knwf`.

---

## Dataset de entrada

**Archivo:** `../data/processed/patagonia_dataset.csv`  
Generado por los notebooks en `../pipeline/` — ver README principal.

| Variable | Tipo | Descripción |
|----------|------|-------------|
| `celda_id` | string | ID único de celda en la grilla |
| `lat`, `lon` | float | Coordenadas del centroide |
| `temp_max_c` | float | Temperatura máxima (°C) |
| `humedad_rel` | float | Humedad relativa media (%) |
| `vel_viento` | float | Velocidad media del viento (m/s) |
| `precip_7d_mm` | float | Precipitación acumulada 7 días (mm) |
| `tipo_vegetacion` | string | Cobertura vegetal dominante (ESA WorldCover) |
| `elevacion_m` | float | Elevación media (m s.n.m.) |
| `foco_presente` | int (0/1) | ¿Hubo ≥1 foco de incendio en el período? |
| `n_focos` | int | Cantidad de focos detectados |
| `frp_mean` | float | Fire Radiative Power promedio (MW) |

---

## Entregables

| Archivo | Descripción |
|---------|-------------|
| `knime/pipeline_patagonia.knwf` | Workflow KNIME completo, documentado y funcional |
| `report/paper_final.pdf` | Informe en plantilla cátedra (6–8 páginas, formato IEEE) |

---

## Rúbrica de evaluación

| Criterio | Peso | Objetivo |
|----------|------|---------|
| Selección y justificación del dataset | 10% | Pregunta analítica bien definida |
| Preprocesamiento y pipeline KNIME | 20% | Pipeline robusto, documentado, reutilizable |
| Análisis exploratorio | 15% | Análisis profundo con interpretación crítica |
| Aplicación de técnicas KDD | 20% | Parámetros justificados (k por Elbow+Silhouette, lift > 2) |
| Descubrimiento e interpretación de patrones | 20% | Patrones significativos, no triviales |
| Formato del artículo académico | 5% | Plantilla cátedra + IEEE correcto |
| Redacción y estructura del informe | 5% | Conclusiones sólidas |
| Presentación oral | 5% | Dominio total del tema |

---

## Limitaciones explícitas (para el informe y la oral)

- Google Sheets no aplica acá, pero sí aplica la limitación análoga: **Google Sheets como
  persistence layer no escala** — en este proyecto, el equivalente es que la grilla de
  25km×25km es un simplificación. La resolución de ERA5 (~28km) introduce suavizado
  espacial que puede ocultar patrones locales.
- El balance de clases (~90% sin foco) afecta directamente el soporte mínimo en Apriori.
  Un soporte de 0.3 no encuentra ninguna regla que involucre focos. Justificar soporte
  bajo (0.01–0.05) es obligatorio en el informe.
- K-Means asume clusters esféricos y es sensible a outliers. La Patagonia tiene
  heterogeneidad geográfica extrema (estepa, bosque andino, cordillera) — los clusters
  no van a ser perfectamente esféricos en el espacio de features.
