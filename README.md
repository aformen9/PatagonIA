# PatagonIA

> Análisis y clasificación de riesgo de incendio en la Patagonia argentina
> a partir de datos satelitales y ambientales públicos.

**Licenciatura en Ciencia de Datos · UCA Rosario · 2026**
Materia A: *Minería de Datos y Big Data* · Materia B: *IA y Aprendizaje Automático I*

🔗 **Aplicación desplegada:** https://patagonia.streamlit.app/

---

## Descripción general

PatagonIA construye un dataset geoespacial sobre una grilla hexagonal H3 (resolución 5, ~253 km² por celda) sobre la Patagonia argentina, integrando fuentes de datos públicas y gratuitas.

Ese dataset es el punto de partida compartido para **dos trabajos académicos con enfoques distintos**:

| | Minería de Datos y Big Data | IA y Aprendizaje Automático I |
|---|---|---|
| **Pregunta** | ¿Qué perfiles ambientales caracterizan zonas con focos? | ¿Se puede clasificar el riesgo de un hexágono con variables ambientales? |
| **Enfoque** | No supervisado — descubrimiento de patrones | Supervisado + no supervisado |
| **Técnicas** | K-Means + Apriori (KNIME) | RF, XGBoost, K-Means (sklearn + xgboost) |
| **Rol del dataset** | Entra como caja negra ya construida | Se reconstruye desde la fuente y se modela |
| **Entregable** | `.knwf` + paper académico PDF | Notebooks + app Streamlit + paper IEEE |
| **Carpeta** | `knime/` + `reports/` | `notebooks/` + `app/` |

> **Nota sobre la relación entre ambos trabajos.** La reconstrucción del dataset realizada para la materia de IA reveló dos problemas en la fuente original (ver *Hallazgos de ingeniería de datos*) que **invierten la conclusión de dominio** obtenida en el trabajo de Minería. No se trata de un error metodológico de aquel trabajo, sino de incompletitud de la descarga original de datos. Ambos trabajos se conservan en el repositorio, y la discrepancia está documentada y cuantificada en el paper de IA.

---

## Fuentes de datos

| Fuente | Contenido | Acceso |
|--------|-----------|--------|
| [NASA FIRMS VIIRS](https://firms.modaps.eosdis.nasa.gov/) | Focos de calor históricos 2012–2023 (375 m) | API key gratuita |
| [NASA POWER](https://power.larc.nasa.gov/) | Normales climáticas (temp, precip, viento, humedad) | API REST sin cuenta |
| [OpenTopoData / SRTM](https://www.opentopodata.org/) | Elevación del terreno | API REST |
| [ESA WorldCover 2021](https://esa-worldcover.org/) | Cobertura vegetal dominante | GeoTIFF descarga directa |
| [IGN Argentina](https://datos.gob.ar/dataset/ign) | Rutas y asentamientos | Shapefile descarga directa |

---

## Hallazgos de ingeniería de datos

Al reconstruir el dataset desde la fuente original aparecieron dos problemas que condicionaron todo el trabajo posterior:

1. **Contaminación por *flaring*.** El 56,7 % de las detecciones correspondían a antorchas de gas de la cuenca de Vaca Muerta, no a incendios: 561 hexágonos con actividad persistente los doce años, ~78–80 % nocturna y confianza del sensor inferior al 1,5 %. Se excluyeron mediante filtro geográfico explícito (`lat > -39.5 & lon > -70`), tras verificar que un filtro por firma espectral no discriminaba la población (16,69 % vs. 16,57 % de positivos).

2. **Omisión del 28 % del dataset.** La descarga original de FIRMS (ID 767267) contenía 1.980 hexágonos; al re-descargar el archivo completo (ID 775078) aparecieron 2.542. Los 562 faltantes correspondían sistemáticamente al corredor andino húmedo —precipitación 1.038 mm vs. 430 mm, humedad 83 % vs. 62 %—, que es justamente la zona de mayor actividad acumulada.

**Consecuencia:** el perfil de mayor riesgo no es cálido/seco/bajo sino húmedo/boscoso/remoto. Bajo agregación decenal domina la disponibilidad de combustible, no la meteorología puntual: la estepa se prende con facilidad pero cada evento es pequeño; el bosque andino arde con más dificultad pero genera eventos extensos y persistentes.

---

## Resultados (IA y Aprendizaje Automático I)

Dataset final: **1.981 hexágonos H3**, 8 predictoras ambientales, sin variables derivadas del fuego.

| Tarea | Modelo seleccionado | Métrica |
|---|---|---|
| Regresión (`n_focos`) | RandomForest sobre `log1p` | MAE = 59,4 ± 5,6 (CV 5-fold) |
| Clasificación (`riesgo_alto`) | XGBoost | F1 = 0,60 ± 0,07 · AUC-PR = 0,65 |
| Clustering | K-Means (k=3) | Cluster andino: 24,2 % riesgo vs. 6,5 % estepa |

**Umbral de decisión en producción:** 0,386 (recall 0,72 · precisión 0,46), elegido por criterio de dominio —un falso negativo cuesta más que un falso positivo— y no por maximización de F1.

El patrón contraintuitivo se confirma por **tres métodos independientes**: correlación de Pearson, clustering no supervisado (que nunca ve el target) y clasificación supervisada.

---

## Estructura del repositorio

```
PatagonIA/
├── notebooks/                        ← construcción del dataset + TPs de IA
│   ├── 01_firms_a_h3.py              ← focos VIIRS 2012-2023 → agregación H3 res 5
│   ├── 02_elevacion_clima.py         ← elevación + clima NASA POWER
│   ├── 03_vegetacion_esa.py          ← cobertura vegetal ESA WorldCover 2021
│   ├── 04_distancias_ign.py          ← distancias a asentamientos y rutas (IGN)
│   ├── 05_join_final.py              ← integración del dataset final
│   ├── 06_eda_tp1.ipynb              ← TP1: análisis exploratorio
│   ├── 07_regresion_tp2.ipynb        ← TP2: modelos de regresión
│   ├── 08_clasificacion_tp3.ipynb    ← TP3: clasificación + clustering
│   └── 09_*.ipynb                    ← TP4: entrenamiento final y serialización
│
├── knime/                            ← Minería de Datos y Big Data
│   ├── PatagonIA_Mineria.knwf        ← workflow K-Means + Apriori
│   ├── results_knime/                ← figuras del workflow
│   └── informe_mdbd/                 ← artículo académico (PDF)
│
├── app/                              ← TP4: app Streamlit
│   ├── app.py
│   └── models/                       ← Pipeline serializado (.joblib)
│
├── data/
│   ├── raw/                          ← datos crudos por fuente (no versionados)
│   ├── processed/                    ← patagonia_dataset.csv (entregable compartido)
│   └── static/                       ← grilla H3, variables estáticas
│
├── src/                              ← funciones reutilizables
│   ├── data.py                       ← carga y preprocesamiento
│   ├── viz.py                        ← visualización
│   └── modelos.py                    ← definición y evaluación de modelos
│
├── model_context/                    ← documentación del proyecto y consignas
├── reports/                          ← informes PDF de ambas materias
├── requirements.txt
├── .gitignore
└── README.md
```

El pipeline de construcción del dataset (`notebooks/01` a `05`) es neutral — no pertenece a ninguna materia en particular.

---

## Instalación

```bash
git clone https://github.com/aformen9/PatagonIA.git
cd PatagonIA
pip install -r requirements.txt
```

### Credenciales necesarias

**NASA Earthdata** (FIRMS) — crear cuenta en https://urs.earthdata.nasa.gov
Guardar en `~/.netrc`:
```
machine urs.earthdata.nasa.gov login TU_USUARIO password TU_PASSWORD
```

El resto de las fuentes no requieren cuenta.

---

## Ejecución

**Reconstruir el dataset desde las fuentes:**
```bash
# Ejecutar en orden — cada script depende del anterior
python notebooks/01_firms_a_h3.py
python notebooks/02_elevacion_clima.py
python notebooks/03_vegetacion_esa.py
python notebooks/04_distancias_ign.py
python notebooks/05_join_final.py
```
Output: `data/processed/patagonia_dataset.csv`

**Reproducir el análisis de IA:** abrir los notebooks `06` a `09` en orden.

**Ejecutar la aplicación localmente:**
```bash
streamlit run app/app.py
```

---

## Variables del modelo

Ocho predictoras ambientales, todas previas al fenómeno:

| Variable | Descripción | Fuente |
|---|---|---|
| `elevacion` | Elevación media del hexágono (m) | OpenTopoData |
| `temp_media` | Temperatura media anual (°C) | NASA POWER |
| `precip_anual` | Precipitación anual (mm) | NASA POWER |
| `viento_medio` | Velocidad media del viento (m/s) | NASA POWER |
| `humedad_relativa` | Humedad relativa media (%) | NASA POWER |
| `dist_asentamiento_km` | Distancia al asentamiento más cercano | IGN |
| `dist_ruta_km` | Distancia a la ruta más cercana | IGN |
| `cobertura_veg` | Cobertura vegetal dominante (categórica) | ESA WorldCover |

**Excluidas por *data leakage*:** `brillo_*`, `frp_*`, `pct_noche`, `pct_verano`, `pct_conf_alta`, `n_anios_activo`, `mes_pico` — todas derivan de las mismas detecciones VIIRS que definen el target. Incluirlas produciría métricas artificialmente altas sin capacidad predictiva real.

---

## Decisiones metodológicas

| Decisión | Elegido | Descartado | Por qué |
|---|---|---|---|
| Grilla espacial | H3 resolución 5 (~253 km²) | Grilla cuadrada lat/lon | Vecindades equidistantes; una grilla lat/lon se deforma con la latitud |
| Exclusión de *flaring* | Filtro geográfico | Filtro por firma espectral | El espectral no discriminaba la población (16,69 % vs. 16,57 %) |
| Target de clasificación | `n_focos > 150` | Percentil arbitrario | Coincide con el límite superior IQR (119,5); deja 13,68 % de positivos |
| Transformación del target | `log1p(n_focos)` | Eliminar outliers | Reduce asimetría de 5,81 a 0,74 preservando los eventos extremos reales |
| Criterio de selección (regresión) | MAE | RMSE | La varianza del RMSE entre particiones (~38) supera la diferencia entre modelos (~6) |
| Métrica de optimización (clasificación) | AUC-PR | Accuracy | Con 13,68 % de prevalencia, el Dummy alcanza 86,3 % de accuracy sin detectar un solo positivo |
| Balanceo de clases | `scale_pos_weight ≈ 6,3` | Submuestreo | Derivado del cociente negativos/positivos (1.710 / 271) |
| Selección de *k* | k = 3 | Solo método del codo | Convergencia de codo, silueta e índice de Davies-Bouldin |
| Serialización | Pipeline completo | Solo el modelo | Evita *training-serving skew* entre entrenamiento e inferencia |

---

## Limitaciones conocidas

- **Variables estáticas.** El modelo usa promedios de doce años, sin meteorología del día: estima propensión estructural, no riesgo actual.
- **Filtro de *flaring* geográfico.** Un filtro rectangular no distingue una antorcha de gas de un incendio forestal genuino dentro de esa zona. Se declara como área de cobertura no confiable.
- **Precisión del 46 %** en el umbral operativo: más de la mitad de las alertas de riesgo alto serían falsas si se usaran como disparador automático. La herramienta es de **priorización territorial con supervisión humana**, no de alerta automática.
- **Sin extrapolación garantizada.** El dataset solo contiene hexágonos con actividad previa. La aplicación valida las combinaciones de entrada mediante distancia de Mahalanobis respecto de la envolvente de entrenamiento, pero eso mitiga el riesgo, no lo elimina.
- **No escala directamente a otras regiones.** El umbral de 150 focos y el filtro de *flaring* están calibrados al régimen de fuego patagónico.

---

## Trabajo futuro

- Incorporar series temporales meteorológicas diarias como capa complementaria a la propensión estructural aquí modelada.
- Validación de campo con el Servicio Nacional de Manejo del Fuego.
- Monitoreo de deriva del modelo (*data drift*) y reentrenamiento periódico.
- Extensión a otras regiones de Argentina con recalibración de umbrales.

---

## Stack

Python 3.11 · pandas · numpy · scikit-learn · xgboost · matplotlib · seaborn · h3 · streamlit · joblib · KNIME (Minería)

---

## Autores

| Nombre | Participación |
|--------|---------------|
| Juan Cruz Chocobares — juancruzchocobares@uca.edu.ar | IA y ML · Minería de Datos |
| Agustín Formenti | IA y ML · Minería de Datos |
| Andrés Morenico | IA y ML · Minería de Datos |
| Lorenzo Mendes | Minería de Datos |

Facultad de Química e Ingeniería, Pontificia Universidad Católica Argentina (UCA), Rosario.

---

## Licencia

MIT License — libre uso con atribución.
