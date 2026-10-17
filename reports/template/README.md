# Plantilla de entregas (informe + presentación)

Extraída de la plantilla Beamer del trabajo "Deep Double Descent" (tema Madrid,
16:9, español, UPM). Ya está **compilada y probada** (pdfLaTeX, Overleaf compatible).
Va en `reports/template/`; cada entrega nueva es una copia de esta carpeta.

## Qué se conservó de la plantilla original

- `beamer` 10pt 16:9, tema `Madrid`, `\usecolortheme{default}`, sin símbolos de navegación, captions numerados.
- Paquetes: `inputenc` (utf8), `fontenc` (T1), `babel` (spanish), `amsmath`, `amssymb`, `graphicx`, `booktabs`.
- Portada: título + subtítulo, autor, institución (UPM, Máster en IA, Visión por Computador).
- Macro `\imgplaceholder[alto]{texto}` para huecos de imagen.
- Patrones de frame: figura + viñetas en columnas (`columns[c,onlytextwidth]`), `block`/`alertblock`, captions `\footnotesize`.
- Logo UPM: sigue **comentado** (no hay archivo).
- Portada repetida al final y frame de referencias (opcionales).

## Qué cambia respecto a la original

| Original | Plantilla |
|----------|-----------|
| Todo en un solo `.tex` | Ensamblador (`slides.tex`, `report.tex`) + `common/` + `sections/` |
| Datos de portada escritos a mano | `common/static_meta.tex` (fijos) + `generated/meta.tex` (por entrega) |
| Imágenes añadidas a mano (`.png`) | Figuras curadas en `figures/<id>.png`, con `\safegraphics` (estilo de los experimentos, DEC-007) |
| Cifras escritas en el texto | `\res{clave}` y `\safetable{...}` desde `generated/` |
| Sin notas | `\speakernote{min}{texto}` en cada frame |

## Estructura

Todas las entregas comparten **un único proyecto de Overleaf**, `reports/overleaf/` (DEC-011):

```
reports/overleaf/
  common/                  shared_packages, slides_style, report_style, static_meta,
                           placeholders, results_interface   (estables; una sola copia para todas las entregas)
  delivery_NN_report.tex   ensambla el informe de la entrega NN   (documentos principales: en la raíz,
  delivery_NN_slides.tex   ensambla la presentación               Overleaf no los admite en subcarpetas)
  delivery_NN/
    generated/             meta.tex, results_macros.tex, figure_captions.tex, tables/   (SE REGENERA siempre)
    figures/               figuras curadas en PNG                 (SE REGENERA siempre)
    sections/              slides_01..07_*.tex y report_01..06_*.tex   (texto: editable)
```

Cada documento principal se crea a partir de `template/overleaf/report.tex` o `slides.tex`, con dos líneas tras
`\documentclass` que añaden `delivery_NN/` a las rutas de búsqueda de LaTeX (`\input@path` para `\input` e
`\IfFileExists`, `\graphicspath` para las imágenes). Así las secciones siguen usando rutas relativas
(`figures/...`, `generated/...`) y las entregas no se pisan: cada documento solo carga las macros de la suya. En la
raíz del proyecto no puede haber carpetas `sections/`, `generated/` ni `figures/` (taparían las de las entregas;
`check_bundle.py` lo comprueba).

Fuera del paquete de Overleaf, cada entrega tiene `delivery.yaml`, `figure_specs.yaml`, `drafts/` (borradores de
las secciones que escribe Claude; `make sections D=NN` los instala en `sections/` sin pisar los editados a mano),
`figure_log.md`, `results_index.json` y `review_notes.md`. Ejemplos de los dos `.yaml`: `*.example.yaml` (entrega 1).

## Convenciones (Claude Code debe respetarlas)

1. **Cifras**: siempre `\res{clave}`. Claves con guiones (`best-mean-recall`), nunca guiones bajos.
   `build_results.py` genera `generated/results_macros.tex` con
   `\expandafter\def\csname res@CLAVE\endcsname{valor}` y coma decimal.
   Una clave sin definir aparece en **rojo** (`[??clave]`) en el PDF.
2. **Tablas**: `build_results.py` escribe en `generated/tables/NOMBRE.tex` **solo el entorno `tabular`**
   (con `booktabs`, sin `table`, sin `caption`). Se incluyen con `\safetable{generated/tables/NOMBRE}`
   dentro de un `table` con su caption, como en `report_03_baseline.tex`.
3. **Figuras** (cambiado por DEC-007): `figures/<id>.png`, un solo archivo para informe y slides, con el mismo
   estilo que las gráficas de los experimentos (`plot_theme.py`) y textos en inglés. Se incluyen con
   `\safegraphics[opciones]{figures/<id>.png}{descripción}` (la ruta lleva extensión). Si falta el archivo, sale un
   recuadro gris (no falla la compilación). Los pies (en español, con el número de experimentos mostrados y el
   criterio de selección) se generan desde `figure_specs.yaml` y se usan con `\figcaption{<id>}`
   (`generated/figure_captions.tex`, que carga `results_macros.tex`).
4. **Texto pendiente**: `\todoslot{descripción}` (se ve en naranja). **No debe quedar ninguno** al terminar;
   `check_bundle.py` los lista y bloquea el empaquetado si quedan.
5. **Frames de decisión**: duplicar el patrón de `slides_03_decisions.tex` (hipótesis, evidencia, resultado,
   decisión). Objetivo: 8 a 10 frames de contenido en total; cada frame con `\speakernote{1}{...}`
   (los tiempos deben sumar ~10 min).
6. **Informe**: usar `\decisionpoint{hip}{evid}{res}{dec}` para cada decisión. 4 a 6 páginas.
7. **Pies de figura**: indicar cuántos experimentos se muestran y el criterio de selección. En las slides, pie corto
   `\figcaptionshort{<id>}` (`caption_slides_es`, máx. ~2 líneas: `check_bundle.py` da error si no cabe) y figura con
   `height=0.62\textheight,keepaspectratio`. **Las viñetas de un frame solo afirman lo que muestra su figura**; lo demás,
   en `\speakernote`.
8. `generated/` y `figures/` se sobrescriben; `sections/` solo según las reglas de regeneración del plan
   (hash en `.build_state.json`; si el usuario editó un archivo, escribir `*.proposed.tex`).
9. **Misma historia en informe y slides** (DEC-010): las mismas figuras y en el mismo orden (el informe es el guion de
   la presentación). En el informe, como mucho 2 figuras por fila y solo si están relacionadas, cada una con su
   `\caption` y `\label` (dos `minipage` en un `figure`); toda figura se cita con `\ref`. `check_bundle.py` lo comprueba.
10. Nombres de archivo en inglés; texto en español (acentos con utf8, igual que la original).
11. **Matriz de confusión en su propia diapositiva, justo antes de las conclusiones** (DEC-012, regla del usuario):
    figura `confusion` (`select: {variant: <final>}`: media de las matrices normalizadas por filas de sus semillas),
    sola en el frame que precede a "Conclusiones…". Por la regla 9 también va en el informe, al final del resultado
    final. `check_bundle.py` da error si falta.
12. **Una sola convención de métricas** (CHG-026): las variantes se comparan por *mean recall*, de validación y de
    train (la métrica con la que se decide; en las figuras de tipo `strength`, `options.metric: mean_recall`). La
    brecha train − validación y las curvas por época van en *accuracy*, la única métrica de train registrada por época.
    Se dice una vez en "Objetivo y datos" y en los pies de las curvas por época.

## Qué debe rellenar Claude Code al crear una entrega

- `generated/meta.tex` desde `delivery.yaml`: `\DeliveryNumber`, `\DeliveryTitle`,
  `\DeliveryShortTitle`, `\DeliverySubtitle`, `\DeliveryDate` (`\today` o fecha fija).
- Sustituir cada `\todoslot{...}` por contenido respaldado por la memoria y los logs.
- Quitar o añadir frames de decisión y secciones según la entrega (el esqueleto es una guía, no un molde rígido).
- Si no hay referencias: vaciar `slides_07_references.tex` y borrar las líneas de referencias en `delivery_NN_slides.tex`.

## Uso en Overleaf

`make bundle D=NN` genera `reports/overleaf_project.zip`, con todo el proyecto (todas las entregas).
- **La primera vez:** New Project → Upload Project con el zip.
- **Entregas siguientes, en el mismo proyecto:** con el botón Upload del árbol de archivos, subir la carpeta
  `delivery_NN/` (conserva su estructura) y los dos archivos `delivery_NN_report.tex` y `delivery_NN_slides.tex`.
  `common/` solo si cambió.
- **Compilar:** abrir `delivery_NN_report.tex` o `delivery_NN_slides.tex` y pulsar Recompile (Overleaf compila el
  archivo abierto si tiene `\documentclass`; también se puede fijar en Menú → Main document). Compilador pdfLaTeX.

## Estado de verificación

Compilada con pdfLaTeX: `slides.tex` (9 páginas con el esqueleto) y `report.tex` (2 páginas), sin errores,
incluyendo los casos de figura ausente, tabla ausente y clave `\res` sin definir (se muestran marcados).
Con datos reales: figura PDF y tabla `tabular` incluidas correctamente.
