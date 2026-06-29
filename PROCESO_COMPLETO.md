# PatagonIA — Informe de proceso completo (Minería de Datos) y proyección a IA

> Documento de memoria del trabajo: **qué se hizo, cómo se razonó, qué se quiso
> obtener y para qué**, paso a paso y decisión por decisión. Sirve para entender
> el proyecto de punta a punta, preparar la defensa oral y dejar registrado el
> vínculo con el Trabajo Final de IA y Aprendizaje Automático.
>
> Complementa —no reemplaza— a:
> - `knime/informe_mdbd/Informe_PatagonIA_MDBD.pdf` — el artículo académico formal (entregable).
> - `knime/README.md` — guía operativa del workflow KNIME, nodo por nodo.
> - `.claude/contexto/PROCESO_DATASET.md` — bitácora técnica de la construcción del dataset.
> - `.claude/contexto/DEFENSA_ORAL.md` — preguntas y respuestas para la oral.

---

## 0. Idea central en una frase

Construimos **un único dataset geoespacial de la Patagonia** (una fila por zona,
22 variables) y lo usamos para **descubrir, sin supervisión, qué perfiles
ambientales caracterizan a las zonas que más se incendian** — usando K-Means
(agrupamiento) y Apriori (reglas de asociación) dentro de KNIME. Ese mismo
dataset, reformulado, es la base del trabajo predictivo de IA.

La pregunta que guía todo el trabajo de Minería es:

> *¿Qué perfiles ambientales caracterizan las zonas con mayor actividad histórica
> de incendios en la Patagonia argentina, y qué combinaciones de factores
> climáticos, de vegetación y de presión humana se asocian con alta ocurrencia de
> focos?*

**Por qué con minería de datos y no con un modelo supervisado:** no tenemos una
etiqueta "verdad" que predecir; queremos que los **patrones emerjan de los datos**
(enfoque inductivo, sin hipótesis a priori). Eso es exactamente lo que hacen el
clustering y las reglas de asociación.

---

## 1. Contexto: dos materias, un solo dataset

El proyecto sirve a **dos materias** de la Licenciatura en Ciencia de Datos (UCA
Rosario, 2026) sobre el **mismo dominio** (incendios en la Patagonia), pero con
enfoques distintos:

| | Minería de Datos y Big Data | IA y Aprendizaje Automático I |
|---|---|---|
| **Pregunta** | ¿Qué perfiles ambientales tienen las zonas con focos? | ¿Cuánto riesgo de incendio habrá en los próximos 7 días? |
| **Enfoque** | No supervisado (descubrir patrones) | Supervisado (predecir) |
| **Técnicas** | K-Means + Apriori (KNIME) | Regresión y clasificación (Python) |
| **Unidad de análisis** | 1 fila = 1 zona (perfil **estático**) | 1 fila = 1 zona × 1 fecha (**temporal**) |
| **Entregable** | `.knwf` + paper PDF | Notebooks + app Streamlit + paper |

Por eso el **pipeline de construcción del dataset** (`notebooks/01..05`) es
**neutral**: no pertenece a ninguna materia, se corre una sola vez y alimenta a
las dos. Esta separación —el dataset entra a Minería como "caja negra" ya
construida— fue aprobada por la cátedra.

---

## 2. Construcción del dataset (pipeline neutral, pasos 01–05)

El dataset se construye en **5 pasos encadenados**, cada uno produce un parquet
intermedio que alimenta al siguiente. La salida final es
`data/processed/patagonia_dataset.csv`.

**Por qué en pasos separados y no un solo script:** cada paso puede fallar de
forma independiente (un tile no disponible, un timeout de API). Guardar parquets
intermedios permite **retomar desde el paso fallido sin rehacer todo** y hacer
control de calidad por etapa (cada script imprime estadísticos de su salida).
Esto también es lo que da **reproducibilidad**: correr los 5 en orden regenera
exactamente el mismo dataset.

```
FIRMS (focos crudos)
  └─ 01_firms_a_h3.py        →  1.980 hexágonos con régimen de fuego
  └─ 02_elevacion_clima.py   →  + elevación (SRTM) + clima (NASA POWER)
  └─ 03_vegetacion_esa.py    →  + cobertura vegetal (ESA WorldCover)
  └─ 04_distancias_ign.py    →  + distancias a asentamientos y rutas (IGN)
  └─ 05_join_final.py        →  patagonia_dataset.csv  (1.980 × 22 variables)
```

### Decisión transversal: ¿por qué la unidad de análisis es una *zona* (hexágono H3) y no un *foco*?

Minería busca patrones **estructurales** ("¿qué tipos de ambiente se incendian?").
La unidad natural es la **zona con su perfil ambiental acumulado**, no el evento
puntual. Si usáramos una fila por foco tendríamos ~400.000 filas con coordenadas
casi idénticas y las variables ambientales (clima, elevación, vegetación)
repetidas, lo que **violaría la independencia entre observaciones** y no aportaría
información nueva. Por eso agregamos los focos por hexágono.

### Decisión transversal: ¿por qué H3 resolución 5?

- **H3 vs grilla cuadrada:** en una grilla cuadrada los 4 vecinos diagonales están
  √2 ≈ 1,41 veces más lejos que los cardinales (anisotropía). En H3 los **6 vecinos
  están siempre a la misma distancia** del centroide — asignación no ambigua,
  estándar emergente en estudios geoespaciales de incendios.
- **Resolución 5 (~253 km²):** captura variabilidad ambiental sin fragmentar de
  más. Res 4 (~1.700 km²) es demasiado gruesa; res 6 (~36 km²) es más fina que la
  resolución de los datos meteorológicos (~28 km), lo que generaría datos
  repetidos sin información real.

---

### Paso 01 — `01_firms_a_h3.py`: régimen de fuego por zona

**Qué hace:** toma los focos crudos de NASA FIRMS (satélite VIIRS), los recorta a
la Patagonia, le asigna a cada foco su hexágono H3, y agrega todos los focos de
cada hexágono en una fila que describe el **régimen de incendio histórico
(2012–2023)** de esa zona.

**Qué queremos obtener y para qué:** 11 variables que resuman "cómo se incendia"
cada zona — no solo cuánto, sino con qué intensidad, cuándo y con qué recurrencia:

| Variable | Derivación | Para qué sirve |
|---|---|---|
| `n_focos` | conteo | Cuánta actividad histórica total |
| `brillo_medio` / `brillo_max` | media / máx de brightness (K) | Intensidad térmica de los focos |
| `frp_medio` / `frp_max` | media / máx de FRP (MW) | Energía radiativa liberada |
| `brillo_t31_medio` | media canal 31 | Temperatura de fondo |
| `pct_noche` | fracción nocturna | Focos nocturnos = incendios más establecidos |
| `pct_verano` | fracción dic-ene-feb | Estacionalidad (verano austral) |
| `pct_conf_alta` | fracción confianza 'h' | Calidad/confianza de las detecciones |
| `n_anios_activo` | años distintos con foco | **Recurrencia** del fuego |
| `mes_pico` | mes más frecuente | Cuándo se concentra la actividad |

**Decisiones justificadas:**
- **VIIRS y no MODIS:** VIIRS tiene píxeles de ~375 m vs ~1 km de MODIS → mejor
  localización del foco → asignación más precisa al hexágono.
- **Período 2012–2023 y no 2010–2023:** VIIRS (Suomi NPP) se lanzó en octubre de
  2011; el primer año calendario completo y confiable del producto FIRMS VIIRS
  Collection 2 es **2012**. Usar 2010-2011 obligaría a mezclar MODIS, con umbrales
  de detección distintos → salto sistemático en la serie. Preferimos **homogeneidad
  de fuente** y sacrificamos dos años.
- **Bounding box lat [-56, -38], lon [-76, -62]:** Patagonia argentina al sur de
  -38°. (El valor -36 que aparecía en planificación previa quedó descartado; el
  código usa -38 y el dato real llega a -37,94°.)

**Salida:** ~1.980 hexágonos con su régimen de fuego + centroide (lat, lon).

---

### Paso 02 — `02_elevacion_clima.py`: terreno y clima

**Qué hace:** agrega a cada hexágono variables de **terreno y clima**, que son los
*drivers* ambientales del riesgo de incendio (Kitzberger et al.): elevación,
temperatura media, precipitación anual, viento medio y humedad relativa.

**Qué queremos obtener y para qué:** caracterizar el "clima típico" de cada zona,
porque eso es lo que define su pirofilia estructural (no el tiempo de un día
puntual).

**Decisiones justificadas:**
- **OpenTopoData SRTM 30m para elevación (no Open-Meteo):** Open-Meteo tiene cuota
  horaria que se agotaba en desarrollo (HTTP 429). OpenTopoData sirve los mismos
  datos base (NASA SRTM v3) sin cuota estricta. El terreno se pide **por hexágono**
  porque cambia rápido (montañas).
- **NASA POWER Climatology (no ERA5/Open-Meteo Archive):** devuelve las **normales
  climáticas 1981–2020 en una sola llamada por punto**, sin cuota horaria, y es
  citable (Stackhouse et al. 2019). No necesitamos pedir 10 años de datos diarios
  y promediar: para caracterizar el clima de una zona, la normal climática es
  exactamente lo que se necesita. Variables: T2M, PRECTOTCORR (→ mm/año), WS10M, RH2M.
- **Clima muestreado a resolución 4 (celda "madre"), no a res 5:** los datos meteo
  tienen resolución nativa ~28 km. Pedir clima para cada hexágono de ~9 km de lado
  sería **pseudo-replicación** (copiar el mismo dato en ~7 vecinos con ruido), lo
  que **inflaría artificialmente la separación entre clusters**. Solución: agrupar
  los hexágonos por su celda madre H3 de res 4 (~28 km), pedir el clima **una vez
  por celda madre** y asignarlo a los hijos. Reduce las llamadas ~4× y es
  metodológicamente correcto.

> Nota: `evapotranspiración` (que requería ERA5) se reemplazó por **humedad
> relativa** (RH2M), que NASA POWER sí provee y es igualmente válida como proxy
> de aridez ambiental.

---

### Paso 03 — `03_vegetacion_esa.py`: cobertura vegetal (el combustible)

**Qué hace:** asigna a cada hexágono su **tipo de cobertura vegetal dominante**
(matorral, pastizal, bosque, etc.) a partir de ESA WorldCover 2021 (10 m).

**Qué queremos obtener y para qué:** la vegetación es el **combustible** del
incendio — no es lo mismo un bosque andino húmedo que una estepa de matorral seco.

**Decisiones justificadas:**
- **Streaming COG y no descarga:** los tiles que cubren la Patagonia pesan ~1,5 GB,
  pero son Cloud-Optimized GeoTIFF: vía `/vsicurl/` GDAL lee **solo los píxeles que
  nos interesan** (HTTP range requests). Leemos unos MB en vez de 1,5 GB.
  Reproducible desde el bucket público de ESA.
- **Moda de 7 puntos por hexágono:** un hexágono res 5 tiene ~2,5 millones de
  píxeles de 10 m; calcular la clase exacta sería costosísimo. Muestreamos 7 puntos
  (centroide + los 7 sub-hexágonos res 6) y tomamos la **moda**. Aproximación
  robusta y computacionalmente trivial.

**Distribución resultante:** matorral (~41,5%) y pastizal (~40,3%) dominan, seguidos
de suelo desnudo, agua, bosque y cultivo — coherente con la estepa patagónica.

---

### Paso 04 — `04_distancias_ign.py`: presión humana

**Qué hace:** agrega la distancia (km) al **asentamiento humano** más cercano y a
la **ruta nacional** más cercana, usando datos del IGN (servicio WFS público).

**Qué queremos obtener y para qué:** el **95% de los incendios patagónicos son de
origen humano** (SNMF); Kitzberger et al. identifican la distancia a asentamientos
como el **predictor estático más robusto** de ocurrencia de incendios. Sin esta
variable, el dataset ignoraría el principal factor de ignición.

**Decisión justificada — proyección métrica AEQD:** las coordenadas en grados no
son distancias lineales (un grado de longitud mide distinto a -38° que a -54°).
Reproyectamos todo a una **Azimutal Equidistante** centrada en la Patagonia
(-45, -69), donde las distancias al centro están en metros reales, y usamos
`sjoin_nearest` de geopandas para hallar el vecino más cercano.

---

### Paso 05 — `05_join_final.py`: integración y limpieza

**Qué hace:** une todo, ordena las columnas en grupos lógicos, **elimina filas con
faltantes** en variables ambientales (muy pocas: zonas costeras sin dato meteo),
redondea para legibilidad y guarda el entregable: `patagonia_dataset.csv` (+ parquet).

**Resultado final:**
- **1.980 filas** (hexágonos con ≥1 foco 2012–2023) × **22 columnas**.
- **0 valores faltantes** (ninguna fila eliminada en la limpieza final relevante).
- **22 variables = 19 analíticas + 3 identificadores** (`hex`, `lat`, `lon`).

Los 5 grupos de variables: identificación (3) · régimen de fuego (11) · terreno y
clima (5) · vegetación (1 categórica) · presión humana (2).

> **Aclaración importante (22 vs 19):** el resumen y las conclusiones del paper
> hablan de **22 variables totales**; el nodo Statistics analiza **19**, porque
> Column Filter excluye los 3 identificadores geográficos. No es contradicción.

---

## 3. El workflow KNIME (Minería de Datos), nodo por nodo

El workflow (`knime/PatagonIA_Mineria.knwf`) implementa el **proceso KDD completo**
(Fayyad et al.: selección → preprocesamiento → transformación → minería →
interpretación) en una sola plataforma reproducible. Tiene dos ramas que parten
del mismo dato:

```
CSV Reader → Column Filter ─┬─ Statistics            (EDA)
                            ├─ Scatter Matrix        (EDA)
                            └─ Normalizer ─ k-Means ─ Color Manager ─┬─ Box Plot
                                              │                      └─ Scatter Plot
                                              └─ Denormalizer ─ Numeric Binner ─
                                                 Columns to Collection ─
                                                 Association Rule Learner ─ Row Filter
```

### 3.1. Ingesta y selección
- **CSV Reader:** carga `patagonia_dataset.csv`.
- **Column Filter:** excluye `hex`, `lat`, `lon`. **Por qué:** son identificadores
  geográficos, no variables analíticas — no aportan al clustering ni a las reglas.
  Quedan las **19 variables analíticas**.

### 3.2. EDA (análisis exploratorio)
- **Statistics:** media, desvío, mín, máx, asimetría, curtosis, nulos e histograma
  por variable. **Para qué:** verificar que los datos tienen sentido físico y
  detectar la forma de cada distribución.
- **Scatter Matrix:** correlaciones visuales entre variables de fuego/brillo.

> **Aclaración sobre el Scatter Matrix:** está conectado al **Column Filter**
> (igual que Statistics) → es parte del **EDA**, no de Apriori. En el canvas quedó
> visualmente "entre" las dos ramas, pero la rama de Apriori arranca en el
> Denormalizer. Si preguntan en la oral, aclarar esto.

### 3.3. Rama K-Means (agrupamiento)
- **Normalizer (Z-Score):** lleva las 18 variables numéricas a media 0, desvío 1.
  **Por qué es obligatorio:** K-Means usa distancia euclidiana; sin normalizar,
  `n_focos` (rango 1–5.406) dominaría sobre `pct_noche` (rango 0–1) y los clusters
  se formarían casi solo por n_focos.
- **k-Means (k=3, semilla 0, máx 99 iteraciones):** agrupa los hexágonos por perfil
  ambiental. **Por qué k=3:** criterio de dominio — la Patagonia tiene al menos
  tres perfiles cualitativamente distintos (andino húmedo / transicional / estepa
  árida pirofílica). La semilla fija garantiza reproducibilidad.
- **Color Manager → Scatter Plot / Box Plot:** colorea por cluster y visualiza la
  separación (precip vs n_focos) y la distribución de n_focos por cluster.

### 3.4. Rama Apriori (reglas de asociación)
- **Denormalizer:** revierte el Z-Score (usando el modelo del Normalizer) para
  volver a la escala original antes de discretizar.
- **Numeric Binner:** discretiza **7 variables continuas** en categorías con
  etiquetas interpretables (ver tabla). **Por qué:** Apriori trabaja con ítems
  categóricos, no con números continuos.

  | Variable | Categorías | Umbrales | Justificación |
  |---|---|---|---|
  | `n_focos` | foco_bajo / foco_medio / foco_alto | <20 / 20–150 / >150 | Percentiles de la distribución |
  | `temp_media` | fría / templada / cálida | <5 / 5–13 / ≥13 °C | Umbrales ecológicos patagónicos |
  | `precip_anual` | seca / semi / húmeda | <300 / 300–600 / ≥600 mm | Árida / semiárida / húmeda |
  | `viento_medio` | moderado / fuerte | <5 / ≥5 m/s | Umbral operacional SNMF |
  | `humedad_relativa` | seca / húmeda | <60 / ≥60 % | Déficit hídrico atmosférico |
  | `dist_asentamiento_km` | cercano / lejano | <50 / ≥50 km | Influencia antrópica directa |
  | `elevacion` | baja / media / alta | <200 / 200–800 / ≥800 m | Pisos altitudinales |

  > Las etiquetas de `n_focos` llevan el prefijo `foco_` (foco_bajo/medio/alto) para
  > **evitar ambigüedad** con las de `elevacion` (baja/media/alta), que solo
  > diferían en una letra.

- **Columns to Collection:** arma el formato "Collection Set" que pide Apriori, con
  las **7 variables discretizadas + la cobertura vegetal** categórica (8 ítems
  interpretables). **Corrección clave de la revisión final:** al inicio se incluían
  variables numéricas crudas (brillo, FRP, porcentajes) que contaminaban el espacio
  de ítems con valores sin sentido (0.0, 367.0); se las quitó.
- **Association Rule Learner (soporte ≥ 0,10; confianza ≥ 0,60):**
  - **Soporte 0,10:** un patrón debe aparecer en al menos **198 de 1.980** zonas
    para ser estructural y no anecdótico. Más bajo → miles de reglas triviales;
    más alto → se perderían patrones reales de zonas de alto riesgo (minoría).
  - **Confianza 0,60:** la regla debe cumplirse en al menos el 60% de los casos.
- **Row Filter (lift ≥ 1,2):** retiene solo reglas donde el antecedente aporta al
  menos **20% de probabilidad adicional** sobre el azar. **Resultado: 184 reglas.**

---

## 4. Resultados e interpretación

### 4.1. EDA
- **Fuerte desbalance en `n_focos`:** media 88,1 vs máx 5.406; asimetría 10,8;
  curtosis 184,1. → **Pocas zonas concentran actividad extrema** de fuego.
- **`precip_anual` muy variable** (CV ≈ 52%): coherente con el gradiente este-oeste
  patagónico (cordillera húmeda → estepa árida).
- **Correlación casi perfecta `brillo_medio` ↔ `brillo_max` (r ≈ 0,98):** miden lo
  mismo (canal VIIRS con distinta agregación temporal) → una podría eliminarse.

### 4.2. K-Means: tres perfiles ambientales
- **cluster_1 (rojo) — Estepa seca pirofílica:** baja precipitación, los valores más
  altos de n_focos (atípicos hasta Z=22). **Zona de mayor riesgo histórico**
  (mesetas áridas de Neuquén y Río Negro).
- **cluster_2 (naranja) — Estepa transicional:** precipitación baja-media, actividad
  moderada. Riesgo intermedio.
- **cluster_0 (verde) — Zona húmeda/andina:** amplio rango de precipitación, n_focos
  consistentemente bajos. Corredor andino.

**Por qué esto valida el trabajo:** el clustering **reproduce inductivamente la
tripartición ecológica clásica de la Patagonia** sin usar ninguna etiqueta — eso le
da validez externa.

### 4.3. Apriori: dos familias de reglas (184 en total)

**Familia 1 — co-ocurrencia con alta actividad de fuego** (las que responden al
abstract):

| Regla | Sop. | Conf. | Lift |
|---|---|---|---|
| {foco_alto, baja} ⇒ calida | 0,100 | 0,975 | **2,052** |
| {calida, foco_alto} ⇒ baja | 0,100 | 0,731 | 2,052 |
| {foco_alto, seca} ⇒ calida | 0,127 | 0,965 | 2,031 |

(`baja` = elevación <200 m; `seca` = precipitación <300 mm.)

**Familia 2 — co-ocurrencia ambiental general:**

| Regla | Sop. | Conf. | Lift |
|---|---|---|---|
| {calida, lejano} ⇒ matorral | 0,104 | 0,827 | 1,995 |
| {calida, seca} ⇒ matorral | 0,317 | 0,814 | 1,964 |

**Interpretación correcta y honesta:** las zonas de alta actividad de fuego
**co-ocurren** con condiciones cálidas, secas y de baja elevación. Apriori mide
**co-ocurrencia, no causalidad ni dirección predictiva**: las reglas caracterizan
el *perfil ambiental* de las zonas pirofílicas, no predicen fuego futuro.

**Matiz que conviene anticipar:** las reglas estrella tienen `foco_alto` en el
**antecedente**, no en el consecuente. Con soporte 0,10, `foco_alto` (328/1.980 =
16,6% de las zonas) aparece sobre todo en antecedentes; a soporte 0,05 aparece una
única regla con foco_alto como consecuente (mencionable como análisis de
sensibilidad). La dirección "condiciones → fuego" se modela en el trabajo
supervisado de IA, no acá.

### 4.4. Triangulación metodológica
Tres métodos independientes convergen en el mismo perfil de riesgo (cálido / seco /
baja elevación / matorral): el **clustering K-Means**, las **reglas Apriori**, y la
**literatura** (Kitzberger et al.). Esa convergencia es el principal argumento de
robustez del trabajo.

---

## 5. Limitaciones (reconocerlas suma puntos)

1. **k=3 por criterio de dominio**, no cuantitativo. El coeficiente de silueta
   permitiría validarlo numéricamente (trabajo futuro).
2. **Apriori asume observaciones independientes:** ignora la **autocorrelación
   espacial** entre hexágonos vecinos. Se podría abordar con DBSCAN geoespacial o
   indicadores LISA.
3. **Solo zonas con ≥1 foco (1.980 de ~3.200 burnables):** correcto para Minería
   (la pregunta es sobre zonas con actividad), pero introduce **sesgo de selección**
   — limitación que el trabajo de IA debe corregir (ver §6).
4. **Clima estático (normales 1981–2020):** describe el perfil típico, no la
   dinámica diaria. Para predecir hace falta meteo diaria (ver §6).

---

## 6. Vínculo y proyección con el Trabajo Final de IA

Esta es la parte que conecta lo descriptivo (Minería) con lo predictivo (IA).

### 6.1. El cambio conceptual clave

| | Minería (este trabajo) | IA y ML (TP1–TP4) |
|---|---|---|
| Naturaleza | Descriptivo, **estático** | Predictivo, **temporal** |
| Unidad | 1 fila = 1 zona | 1 fila = 1 zona × 1 **fecha** |
| Meteo | NASA POWER (**normales** 1981-2020) | **GFS hindcast diario** |
| Zonas | Solo con foco (1.980) | **Todos** los ~3.200 burnables |
| Objetivo | Descubrir perfiles | Predecir focos D+1…D+7 |
| Balance | — | Desbalance ~90/10 (sin/con foco) |

El dataset de IA **no es el mismo archivo**: comparte el dominio, las fuentes y
buena parte de la lógica de construcción, pero se reformula a **hexágono × fecha**,
agrega **todos** los hexágonos burnables (con `focos = 0` los días sin detección) y
reemplaza las normales climáticas por **GFS diario**, porque ahí sí importa "qué
tiempo hizo (o hará) ese día".

### 6.2. Cómo Minería alimenta concretamente a IA

**A. Validación empírica de features (el puente más fuerte).**
Las reglas Apriori + el clustering **confirman con evidencia** —no por intuición—
que `temp_media`, `precip_anual` y `elevacion` discriminan zonas de alto fuego.
Eso **justifica incluir esas variables como predictores** en los modelos
supervisados de TP2/TP3. El perfil pirofílico descubierto (cálido / seco / baja
elevación / matorral) es la **hipótesis física** que el modelo supervisado debe
capturar.

**B. Reúso directo del clustering en TP3.**
La consigna de TP3 pide, como actividad complementaria de alto valor, **aplicar
clustering y contrastarlo con las clases supervisadas**. El K-Means de Minería se
reutiliza tal cual: ¿los clusters coinciden con la etiqueta `riesgo_alto`? ¿revelan
subestructura? Es trabajo ya hecho que se capitaliza.

**C. Anticipación del desbalance.**
El EDA de Minería ya mostró que **pocas zonas concentran casi todo el fuego**
(curtosis 184, atípicos a Z=22). Eso anticipa el **desbalance de clases ~90/10** que
TP3 debe tratar (estratificación, `class_weight`, SMOTE) y explica por qué Accuracy
sola no alcanza (hay que mirar Precision/Recall/F1).

**D. Corrección de la limitación de selección.**
La limitación de Minería (solo zonas con foco) se **resuelve** en IA incluyendo
todos los hexágonos burnables: así el modelo aprende también de zonas que **no** se
incendian, que es justo lo que un sistema de alerta necesita distinguir.

### 6.3. Recorrido de los TPs de IA y dónde encaja cada pieza

- **TP1 — EDA y preparación.** Mismo dominio, dataset reformulado (hexágono × fecha,
  ≥1.500 filas, ≥12 variables, mezcla numéricas/categóricas, nulos y outliers). Se
  formulan ≥10 preguntas (analíticas + de **predicción** + de **clasificación**).
  El EDA de Minería es el punto de partida natural. Define los dos *targets*:
  `focos_D+k` (regresión) y `riesgo_alto_D+k` (clasificación).
- **TP2 — Regresión (predecir focos a D+1…D+7).** ≥3 regresores (baseline lineal +
  Random Forest / XGBoost), métricas MSE/RMSE/MAE/R², partición temporal con semilla
  fija. Anti-leakage: las features meteo van con `shift` para no usar información del
  futuro. Las variables validadas en Minería entran con fundamento.
- **TP3 — Clasificación (¿riesgo alto sí/no?).** ≥3 clasificadores + tratamiento del
  desbalance + matrices de confusión + ROC/AUC. **Acá conecta el clustering de
  Minería** (actividad complementaria) y los ensambles (Random Forest, boosting).
- **TP4 — Despliegue + paper integrador.** App **Streamlit** que sirve el mejor
  modelo (entrada de variables → riesgo a 7 días), modelo serializado (`joblib`),
  repo público, y **artículo IEEE de 10–15 páginas** (plantilla de cátedra, Arial,
  español con términos técnicos en cursiva) que integra TP1–TP4.

### 6.4. Frase de cierre defendible

> *El trabajo de Minería y el sistema de forecasting de IA son **dos entregables
> distintos sobre el mismo dominio**. Minería caracteriza, de forma descriptiva y
> no supervisada, qué perfil ambiental tienen las zonas que se incendian; ese
> hallazgo —validado por triangulación entre clustering, reglas de asociación y
> literatura— **fundamenta la selección de variables** del modelo predictivo
> supervisado. El dataset construido y los patrones descubiertos constituyen la
> base directa para el modelado de IA y Aprendizaje Automático.*

---

## 7. Mapa de archivos del proyecto

| Ruta | Qué es |
|---|---|
| `notebooks/01..05_*.py` | Pipeline neutral de construcción del dataset |
| `notebooks/README.md` | Qué hace cada paso y por qué |
| `data/processed/patagonia_dataset.csv` | **Dataset final** (entregable compartido) |
| `knime/PatagonIA_Mineria.knwf` | **Workflow KNIME** (entregable Minería) |
| `knime/README.md` | Guía del workflow nodo por nodo |
| `knime/informe_mdbd/Informe_PatagonIA_MDBD.pdf` | **Artículo académico** (entregable Minería) |
| `knime/results_knime/*` | Figuras del workflow (Statistics, clusters, reglas…) |
| `.claude/contexto/PROCESO_DATASET.md` | Bitácora técnica del dataset |
| `.claude/contexto/DEFENSA_ORAL.md` | Q&A para la defensa oral |
| `model_context/consignas_*` | Consignas de ambas materias |
| `PROCESO_COMPLETO.md` | **Este documento** |
