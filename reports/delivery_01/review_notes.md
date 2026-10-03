# Review notes: delivery 1

Things to check before uploading to Overleaf. Text in Spanish is in `overleaf/sections/`; numbers come from
`generated/results_macros.tex` (origin of every key in `results_index.json`).

## Pending on the user

- **Compilation.** There is no LaTeX locally: `check_bundle.py` checks references, macros, `\safegraphics`/`\safetable`
  targets, environments, packages, pdfLaTeX-unsafe characters, pending `\todoslot` (none) and speaker-note minutes
  (10 frames, 10 min), but the first real compilation will be in Overleaf (pdfLaTeX). Look at the report length there
  (estimated ~5 pages; the template's `\parskip` may add some).
- **Template.** The bundle uses the user's template unchanged (`common/`, `slides.tex`, `report.tex`). Two additions,
  both outside the template's own files: `generated/results_macros.tex` loads `generated/figure_captions.tex`
  (`\figcaption{id}`), and the figures are `figures/<id>.png` (DEC-007) instead of `<id>_<context>.pdf`.
  With `babel` spanish (as in the template) tables are labelled "Cuadro", so the text says "Cuadro~\ref{...}". For
  "Tabla", add the option `es-tabla` to babel in `common/shared_packages.tex` and replace "Cuadro" in the sections.
- **References in the report:** the template has no bibliography section; a `thebibliography` was added at the end of
  `report_06_conclusions.tex`.
- **Git tag** `delivery-01`: to be created when the delivery is closed (not done yet).

## Discrepancies between LOGBOOK.md and the data (the documents use the data)

| Quantity | LOGBOOK.md | Data (macro) | Why |
|---|---|---|---|
| Gain of `Rescaling` (EXP-002 vs EXP-001) | +8,3 pp | +8,4 pp (`gain-normalisation`) | LOGBOOK subtracted rounded values (0,365 − 0,282); CSV: 0,3654 − 0,2817 |
| Speed-up of batch 256 | ×15 | ×16,4 (`speedup-batch256`) | LOGBOOK used one seed (10,6 / 0,7 s); the macro averages the 3 seeds of each configuration |

## Statements without a direct number in experiments/

- "La mayoría de las ReLU queda inactiva" a 224×224 (report, phase 2; slide "Decisión 1": "casi todas las ReLU
  mueren"). Measured in EXP-005 (93 % at 224×224) with the saved models, but that measurement is only in LOGBOOK.md,
  not in `history/*.json` (the field `inactive_units` exists only from EXP-006 on). The text does not quote the number.
- "Cada época volvía a leer los TIF y esa lectura dominaba el tiempo" (report, baseline; slide 3): from ISS-005 /
  CHG-003 / CHG-007 in LOGBOOK.md (EXP-000 took ~70 min on CPU). No timing of EXP-000 is in `experiments/`.
- "El notebook reproduce exactamente la semilla 1": run `EXP-018_20261003-142325` in `experiments/runs.csv` vs
  `EXP-018_final` in `experiments.csv` (same best epoch 47, mean recall 0,6443, val acc 0,6373). Verified, but the
  report does not cite the run id.
- "Se confunde sobre todo con los otros vehículos" (worst class, Truck): read from the confusion matrix of seed 1
  (Truck → Small car 0,15, Bus 0,14, Dump truck 0,17, Shipping container 0,11). Qualitative.
- Slide 3 says "desde la fase 5, 3 semillas por decisión". Strictly: the two finalists of 5.2 and every decision from
  5.4 on (DEC-006, EXP-009). Phase 4's row in the tables has 3 seeds because EXP-007 was re-run at the end of phase 5,
  but its decision was taken with one seed.

## Hand-written values that are definitions, not results

`[0, 255]`, `[0, 1]`, `Rescaling(1/255)`, `Dense(13)`, rotations of 90°, the 9 phases, "la mitad o el doble" of the lr
(EXP-016), and the target times of the speaker notes. Everything else goes through `\res{}`.

## Figures

- Same style as the experiment plots (`plot_theme.py`, shared with `plot_utils.py`), English labels and titles, PNG
  at 200 dpi (user's decision, DEC-007): the plan asked for Spanish labels and vector PDF. Captions are Spanish.
- Class names are the dataset's (English) in figures, tables and text, so that they match.
- No `\imgplaceholder` used: all figures exist. Selection criteria in `figure_log.md`.
- `class_distribution`, `capacity` are used only in the slides; `ablation` and `confusion` are generated but not
  used (the report shows the ablation as a table); add them if wanted.

## Other

- No previous delivery: the "entrega anterior → ahora" comparison starts with delivery 2.
- EXP-000 (original split) is excluded from every table and figure; the baseline is EXP-001.
- `experiments/logs/phase7.log` does not exist (phase 7 ran in the foreground); no impact, the CSV is complete.
