# Presentación — Defensa oral (Minería de Datos)

Diapositivas para la defensa oral del Trabajo Final de Minería de Datos.

## Archivo

- **`defensa_patagonia.html`** — presentación completa (10 diapositivas), autocontenida.
  Se abre directamente en cualquier navegador, **no requiere servidor ni dependencias**
  (sólo conexión a internet la primera vez, para las fuentes de Google Fonts).

## Origen

Implementación standalone del diseño hecho en Claude Design
(`Defensa PatagonIA.dc.html`). Se reescribió como HTML + CSS plano, sin el runtime
propietario del editor, para que el entregable viva versionado en el repo y sea
reproducible/abrible sin herramientas externas (ver regla de defensibilidad en
`CLAUDE.local.md`).

El contenido de cada diapositiva está alineado con el guión de
[`.claude/contexto/DEFENSA_ORAL.md`](../../.claude/contexto/DEFENSA_ORAL.md).

## Imágenes

Tres diapositivas embeben resultados reales de KNIME desde
[`../results_knime/`](../results_knime):

| Diapositiva | Imagen |
|---|---|
| 04 · Pipeline | `flujo_knime_comentado.jpeg` |
| 05 · EDA | `stats1.jpeg` |
| 07 · Apriori | `apriori_reglas.jpeg` |

Las diapositivas 01 (portada) y 03 (dataset) usan **`mapa_h3.png`**, el mapa de la
grilla H3 de la Patagonia coloreada por actividad de fuego. Se genera de forma
reproducible con [`generar_mapa_h3.py`](generar_mapa_h3.py) desde el dataset final:

```powershell
# desde la raíz del repo, con la venv
.\.venv\Scripts\python.exe knime\presentacion\generar_mapa_h3.py
```

Ya no quedan marcadores pendientes: las 10 diapositivas tienen su contenido final.

## Controles

| Tecla | Acción |
|---|---|
| `→` `↓` `Espacio` · click | Siguiente |
| `←` `↑` | Anterior |
| `Inicio` / `Fin` | Primera / última |
| `N` | Mostrar/ocultar notas del orador |
| `F` | Pantalla completa |
| click en una imagen | Ampliar a pantalla completa (lightbox) |
| click / `Esc` (con imagen ampliada) | Cerrar la imagen |

**Ampliar imágenes (lightbox):** las figuras de resultados (workflow KNIME, Statistics,
reglas Apriori y el mapa H3) se pueden tocar para verlas a pantalla completa con el
fondo de la presentación difuminado — útil para mostrar en detalle durante la defensa.
El click sobre la imagen no avanza la diapositiva.

Las diapositivas están diseñadas a 1920×1080 y se escalan automáticamente a la
pantalla manteniendo la proporción 16:9. Se puede abrir en una diapositiva
concreta agregando `#N` a la URL (p. ej. `defensa_patagonia.html#6`).
