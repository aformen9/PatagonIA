# Guía KNIME — PatagonIA (Minería de Datos)

Esta guía explica cómo construir el workflow en KNIME Analytics Platform
usando el dataset ya generado. El workflow completo se llama `PatagonIA_Mineria.knwf`
y vive en esta carpeta.

---

## Dataset de entrada

**Archivo:** `data/processed/patagonia_dataset.csv`  
**Filas:** 1.980 (una por zona geográfica — hexágono H3 ~253 km²)  
**Columnas:** 22

| Grupo | Columnas |
|---|---|
| Identificación (no usar en análisis) | `hex`, `lat`, `lon` |
| Régimen de fuego (FIRMS) | `n_focos`, `brillo_medio`, `brillo_max`, `frp_medio`, `frp_max`, `brillo_t31_medio`, `pct_noche`, `pct_verano`, `pct_conf_alta`, `n_anios_activo`, `mes_pico` |
| Terreno y clima | `elevacion`, `temp_media`, `precip_anual`, `viento_medio`, `humedad_relativa` |
| Vegetación | `cobertura_veg` (categórica: matorral / pastizal / bosque / suelo_desnudo / agua / cultivo / urbano / nieve_hielo) |
| Presión humana | `dist_asentamiento_km`, `dist_ruta_km` |

---

## Arquitectura del workflow

```
[CSV Reader]
     ↓
[Column Filter]        ← saca hex/lat/lon del análisis
     ↓
[Statistics]           ← EDA: estadísticos descriptivos
     ↓
[Scatter Matrix]       ← EDA: correlaciones visuales
     ↓
[Normalizer]           ← escala variables numéricas (Z-score)
     ↓
[K-Means]              ← clustering de zonas
     ↓
[Color Manager]        ← colorea por cluster
     ↓
[Cluster Assigner]     ← asigna cluster a cada fila
     ↓         ↓
[Box Plot]    [Scatter Plot]   ← caracterización de clusters
     ↓
[Denormalizer]         ← volver a escala original para Apriori
     ↓
[Auto-Binner × N]      ← discretiza variables numéricas en rangos
     ↓
[Apriori]              ← reglas de asociación
     ↓
[Rule Filter]          ← filtra por lift > 1 y confianza > 0.6
```

---

## Nodo por nodo

### 1. CSV Reader
- **Nodo:** `CSV Reader`
- **Settings:**
  - File: ruta al archivo `data/processed/patagonia_dataset.csv`
  - Has column header: ✓
  - Has row ID: ✗
  - Column delimiter: `,`
- **Verificar:** que reconozca `cobertura_veg` y `mes_pico` como String, y el resto como Double/Long.

---

### 2. Column Filter
- **Nodo:** `Column Filter`
- **Excluir del análisis:** `hex`, `lat`, `lon`
  - Son identificadores geográficos, no variables analíticas. No aportan información
    al clustering ni a las reglas.
- **Dejar todo el resto** (19 variables).

---

### 3. Statistics (EDA)
- **Nodo:** `Statistics`
- Conectar a la salida del Column Filter.
- Genera media, desvío, mín, máx, missing values por columna.
- **Para el informe:** hacer captura de la tabla de estadísticos. Sirve para la
  sección "Análisis Exploratorio" y permite verificar que los datos tienen sentido
  (ej. temperatura media entre -0.3 y 15.9°C es coherente con Patagonia).

---

### 4. Scatter Matrix (EDA)
- **Nodo:** `Scatter Matrix`
- Seleccionar las variables numéricas más relevantes para visualizar correlaciones:
  `n_focos`, `temp_media`, `precip_anual`, `viento_medio`, `humedad_relativa`,
  `elevacion`, `dist_asentamiento_km`
- **Para el informe:** capturar la matriz. Si se ve correlación negativa
  precip–n_focos y positiva viento–n_focos, eso valida las hipótesis del proyecto.

---

### 5. Normalizer
- **Nodo:** `Normalizer`
- **Método:** Z-Score (media=0, desvío=1)
- **Por qué normalizar:** K-Means usa distancia euclidiana. Sin normalizar,
  variables con escalas grandes (ej. `n_focos` va hasta 5.406, `precip_anual`
  hasta 1.708 mm) dominarían la distancia sobre variables en escala 0-1
  (`pct_noche`, `pct_verano`).
- **Incluir:** solo columnas numéricas (excluir `cobertura_veg` y `mes_pico`).
- **Guardar el nodo Normalizer** para poder conectarlo al Denormalizer después.

---

### 6. K-Means — elegir k

**Cómo elegir el número de clusters (k):**

Correr K-Means varias veces con distintos k (3, 4, 5, 6, 7) y registrar la
**Within-Cluster Sum of Squares (WCSS)**. El k óptimo es donde la curva "coda"
(método del codo — *elbow method*).

En KNIME:
1. Usar un loop (`Chunk Loop Start` → `K-Means` → `Chunk Loop End`) con k variando, O
2. Más simple: correr K-Means con k=3, 4, 5 por separado y comparar el score.

**k recomendado para este dataset: 4 o 5.**
Justificación: hay al menos 4 perfiles ambientales reconocibles en la Patagonia:
- Zona andina húmeda (bosque, alta precip, alta elevación)
- Estepa seca (matorral/pastizal, baja precip, viento alto)
- Zona costera / litoral
- Zona norpatagónica (más cálida, más cerca de asentamientos)

**Nodo:** `K-Means`
- Número de clusters: empezar con **4**, verificar con 5
- Distance: Euclidean
- Max iterations: 200
- Columnas: **solo las normalizadas numéricas** (excluir `cobertura_veg`, `mes_pico`)

---

### 7. Color Manager + visualización de clusters

- **Nodo:** `Color Manager`
  - Colorear por la columna `Cluster` que genera K-Means.
- **Nodo:** `Scatter Plot`
  - X: `precip_anual`, Y: `n_focos` (o `temp_media` vs `n_focos`)
  - Color: cluster → debería verse separación clara.
- **Nodo:** `Box Plot`
  - Variable: `n_focos`, agrupado por `Cluster`
  - Permite comparar el régimen de fuego entre clusters.

**Para el informe:** capturar el scatter plot y el box plot. Describir qué
caracteriza a cada cluster (ej. "Cluster 0: alta precipitación, bajo n_focos →
zona andina húmeda").

---

### 8. Denormalizer (antes de Apriori)

- **Nodo:** `Denormalizer`
- Conectar el mismo modelo del Normalizer (puerto izquierdo = model).
- Devuelve los valores a escala original antes de discretizar para Apriori.

---

### 9. Auto-Binner (discretización para Apriori)

Apriori trabaja con variables **categóricas** (ítems). Las variables continuas
hay que discretizarlas en rangos con etiquetas descriptivas.

Variables a discretizar y rangos sugeridos:

| Variable | Nro. bins | Etiquetas sugeridas |
|---|---|---|
| `n_focos` | 3 | bajo (<20), medio (20-150), alto (>150) |
| `temp_media` | 3 | fría (<5°C), templada (5-13°C), cálida (>13°C) |
| `precip_anual` | 3 | seca (<300mm), semi (<600mm), húmeda (>600mm) |
| `viento_medio` | 2 | moderado (<5 m/s), fuerte (≥5 m/s) |
| `humedad_relativa` | 2 | seca (<60%), húmeda (≥60%) |
| `dist_asentamiento_km` | 2 | cercano (<50 km), lejano (≥50 km) |
| `elevacion` | 3 | baja (<200m), media (200-800m), alta (>800m) |

**Nodo:** `Auto-Binner` (uno por variable, o usar `Equal Frequency Binner` si se
prefiere que cada bin tenga la misma cantidad de filas).

`cobertura_veg` y `mes_pico` ya son categóricas, se usan directo.

---

### 10. Apriori

- **Nodo:** `Apriori Association Rule Learner`
- **Parámetros:**
  - Min support: **0.10** (un patrón debe aparecer en al menos 10% de las zonas = 198 zonas)
  - Min confidence: **0.60** (la regla debe cumplirse en el 60% de los casos)
  - Max antecedent length: 3 (hasta 3 condiciones en el antecedente)
- **Consecuente de interés:** `n_focos=alto` (el ítem que queremos predecir)

**Por qué estos valores de soporte y confianza:**
- Soporte 0.10: en 1.980 zonas, un patrón con soporte 5% = 99 zonas, que es
  estadísticamente relevante pero demasiado específico. 10% asegura que el patrón
  es estructural, no anecdótico.
- Confianza 0.60: buscar reglas con >60% de precisión para que sean accionables.

---

### 11. Rule Filter

- **Nodo:** `Rule Filter` o filtrar la tabla de reglas con `Row Filter`
- Condiciones:
  - `lift > 1.2` (la regla mejora al menos un 20% sobre el azar)
  - consecuente = `n_focos=alto`
- **Para el informe:** reportar las 5 reglas con mayor lift. Ejemplo esperado:
  `{matorral, viento=fuerte, humedad=seca} → n_focos=alto` con lift ~2.x

---

## Checklist de entrega

- [ ] Workflow guardado como `PatagonIA_Mineria.knwf` en la carpeta `knime/`
- [ ] Todos los nodos ejecutados (sin nodos en rojo)
- [ ] Screenshots guardados: Statistics, Scatter Matrix, Scatter Plot clusters,
      Box Plot clusters, tabla de reglas Apriori
- [ ] Documentación interna del workflow: agregar `Workflow Annotations`
      (clic derecho → Add annotation) en cada sección explicando qué hace
- [ ] K elegido justificado (codo o silhouette)

---

## Outputs para el informe

Del workflow deben salir estos elementos para incluir en el PDF:

1. **Tabla de estadísticos** (Statistics) — sección EDA del informe
2. **Scatter matrix** o gráfico de correlaciones — sección EDA
3. **Gráfico del codo** (WCSS vs k) — sección K-Means
4. **Scatter plot coloreado por cluster** — sección K-Means
5. **Box plots de n_focos por cluster** — sección K-Means
6. **Tabla de las 5 mejores reglas Apriori** (soporte, confianza, lift) — sección Apriori

---

## Preguntas frecuentes para la defensa oral

**¿Por qué K-Means y no otro clustering?**
K-Means es apropiado cuando se espera que los grupos sean relativamente compactos y
de tamaño similar en el espacio de features. Para zonas geográficas con perfiles
ambientales continuos (temperatura, precipitación, etc.) es la elección estándar.
Alternativa válida: clustering jerárquico (Ward), pero K-Means es más escalable.

**¿Por qué normalizar antes de K-Means?**
K-Means minimiza la suma de distancias euclidianas. Sin normalizar, una variable
como `n_focos` (rango 1-5.406) dominaría sobre `pct_noche` (rango 0-1), asignando
clusters según n_focos casi exclusivamente.

**¿Por qué support=0.10 en Apriori?**
Con 1.980 zonas, support 10% = 198 zonas mínimas para que el patrón sea considerado.
Un valor más bajo (ej. 2%) generaría miles de reglas triviales o específicas a
situaciones anecdóticas. Un valor más alto (ej. 25%) podría eliminar patrones
reales de zonas de alto riesgo (que son minoría).

**¿Qué es el lift y por qué filtramos lift > 1?**
Lift = confianza(regla) / soporte(consecuente). Si lift = 1, la regla no aporta
información (el consecuente ocurre con la misma frecuencia con o sin el antecedente).
Lift > 1 significa que el antecedente aumenta la probabilidad del consecuente.
Filtramos lift > 1.2 para quedarnos solo con reglas que tengan impacto práctico.
