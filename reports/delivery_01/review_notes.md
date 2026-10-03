# Review notes: delivery 1

Scope (DEC-008): design and training of the ffNN -- architecture, optimisation algorithm and training parameters,
tuned on the validation set, which also decides the stopping point. **Phases 0-5** of the plan; final model
`EXP-010_bs256_sqrt_60ep` (3 seeds). Phases 6-9 (regularisation, class imbalance, augmentation, ablation; best model
EXP-018) stay in `experiments/` and LOGBOOK.md for a later delivery; the full version of this delivery is the git tag
`ffnn-full-plan`.

Text in Spanish is in `overleaf/sections/`; numbers come from `generated/results_macros.tex` (origin of every key in
`results_index.json`).

## Pending on the user

- **Compilation.** No LaTeX locally: `check_bundle.py` checks references, macros, `\safegraphics`/`\safetable`
  targets, environments, packages, pdfLaTeX-unsafe characters, pending `\todoslot` (none) and speaker-note minutes
  (10 frames, 10 min). First real compilation in Overleaf (pdfLaTeX); check the report length there (~5 pages
  estimated; the template's `\parskip` may add some).
- **Template additions** (outside the template's own files): `generated/results_macros.tex` loads
  `generated/figure_captions.tex` (`\figcaption{id}`); figures are `figures/<id>.png` (DEC-007). Tables are labelled
  "Cuadro" by babel spanish, so the text says "Cuadro~\ref{...}" (for "Tabla": option `es-tabla`).
- **References:** the report's bibliography is a `thebibliography` at the end of `report_06_conclusions.tex`. All
  references come from LOGBOOK.md ("Referencias externas"), written from memory and **not verified** against the
  sources (see the conversation of 2026-10-03; LOGBOOK cites [R31] and [R32] without listing them).
- `prediction.zip` in the project root (made by the user at 15:53) contains the predictions of EXP-018; the current
  `prediction.json` is from the delivery-1 model (EXP-010).
- **Git tag** `delivery-01`: to be created when the delivery is closed.

## Discrepancies between LOGBOOK.md and the data (the documents use the data)

| Quantity | LOGBOOK.md | Data (macro) | Why |
|---|---|---|---|
| Gain of `Rescaling` (EXP-002 vs EXP-001) | +8,3 pp | +8,4 pp (`gain-normalisation`) | LOGBOOK subtracted rounded values (0,365 − 0,282); CSV: 0,3654 − 0,2817 |
| Speed-up of batch 256 | ×15 | ×16,4 (`speedup-batch256`) | LOGBOOK used one seed (10,6 / 0,7 s); the macro averages the 3 seeds of each configuration |

## Statements without a direct number in experiments/

- "La mayoría de las ReLU queda inactiva" a 224×224 (report, phase 2; slide "Reducir la resolución"). Measured in
  EXP-005 (93 % at 224×224) with the saved models, but only recorded in LOGBOOK.md, not in `history/*.json`.
- "Cada época volvía a leer los TIF y esa lectura dominaba el tiempo" (report and slide of the baseline): from
  ISS-005 / CHG-003 / CHG-007 in LOGBOOK.md. No timing of EXP-000 is in `experiments/`.
- "El notebook reproduce exactamente la semilla 1": run `EXP-010_20261003-160510` in `experiments/runs.csv` vs
  `EXP-010_bs256_sqrt_60ep` in `experiments.csv` (best epoch 20 of 30, mean recall 0,5736, val acc 0,6091). Verified.
- "Truck se confunde con los otros vehículos": confusion matrices of the 3 seeds (Truck → Small car 0,18-0,24,
  Bus 0,11-0,13, Dump truck 0,10-0,11). Qualitative in the text.
- The seed noise (`seed-noise-pp`) is pooled over every variant with 3 seeds in `experiments.csv`, including
  phases 6-9 (outside this delivery's scope); it describes the training setup, not a result of those phases.

## Hand-written values that are definitions, not results

`[0, 255]`, `[0, 1]`, `Rescaling(1/255)`, the phase numbers and the target times of the speaker notes. Everything else
goes through `\res{}`.

## Figures

- Same style as the experiment plots (`plot_theme.py`), English labels, PNG at 200 dpi (DEC-007). Captions in
  Spanish. Class names as in the dataset. Selection criteria in `figure_log.md`.
- `class_distribution` and `normalisation` are used only in the slides. `normalisation` shows the training loss only:
  the validation loss of `EpochMetrics` is capped at ~16 (ISS-007).
- Slide captions are the short `caption_slides_es`; `check_bundle.py` estimates that each fits in 2 lines, but only the
  Overleaf compilation confirms it. Check every slide once there.

## Other

- No previous delivery: the "entrega anterior → ahora" comparison starts with delivery 2.
- EXP-000 (original split) is excluded from every table and figure; the baseline is EXP-001.
- `hyperparameter_sweep.ipynb` still shows all 9 phases (it is the full record of the sweep).
