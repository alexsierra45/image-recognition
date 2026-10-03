---
description: Crea una nueva entrega (informe + slides) y su paquete de Overleaf
argument-hint: <delivery_number> <baseline_experiment_id>
---

Crea la entrega número $1 con baseline $2. Todo es de solo lectura sobre `experiments/`.

1. Lee `reports/template/` (README incluido), la carpeta de la entrega anterior si existe y su `review_notes.md`.
2. Crea `reports/delivery_NN/` (NN con dos cifras): copia `reports/template/overleaf/common/`, `slides.tex` y
   `report.tex` a `reports/delivery_NN/overleaf/`, y los esqueletos `reports/template/overleaf/sections/*.tex` a
   `reports/delivery_NN/drafts/` (ahí se redactan; `make sections` los instala en `overleaf/sections/`).
   Escribe `delivery.yaml` (fases y variantes que representan cada resultado, sacadas de LOGBOOK.md) y
   `figure_specs.yaml`, partiendo de los `*.example.yaml`. Pregúntame si falta algún dato clave (título, fases que
   destacar). Si el split de datos cambió respecto a la entrega anterior, no mezcles resultados en tablas ni figuras.
3. Ejecuta `make results D=NN` y `make figures D=NN`. Las figuras siguen el estilo de los experimentos
   (`plot_theme.py`), con textos en inglés y en PNG; los pies van en español en `figure_specs.yaml`. Revisa cada PNG.
4. Lee LOGBOOK.md y redacta en `drafts/` las secciones en español, sustituyendo todos los `\todoslot{}` (frames de
   decisión: hipótesis, evidencia, resultado y `\decisionblock`; informe: `\decisionpoint`; cada frame con
   `\speakernote{min}{...}`, unos 10 min en total; figuras con `\safegraphics{figures/<id>.png}` y pie
   `\figcaption{<id>}`; tablas con `\safetable`). Toda
   cifra va con `\res{}` o con tablas de `generated/tables/`. No inventes nada: lo que no se pueda rastrear a
   LOGBOOK.md o a `experiments/` va a `review_notes.md`. Si hay entrega anterior, añade la comparación
   "entrega anterior → ahora".
5. `make bundle D=NN` (instala las secciones sin pisar las editadas a mano, comprueba y empaqueta). Corrige los
   errores de `check_bundle.py` hasta que no quede ninguno.
6. Registra el trabajo en LOGBOOK.md (CHG) con el skill `log-experiment`.
7. Termina con un resumen: figuras incluidas y omitidas (`figure_log.md`), afirmaciones sin respaldo y discrepancias
   (`review_notes.md`), y lo que necesite mi revisión antes de subir el zip a Overleaf (New Project → Upload Project;
   documento principal `report.tex` o `slides.tex`; compilador pdfLaTeX).
