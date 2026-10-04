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
  (11 frames, 11,5 min). First real compilation in Overleaf (pdfLaTeX); check the report length there.
- **Length (CHG-018, user's request for more detail on phases 0-5):** report ~9,5 pages estimated (template target
  4-6) and 11 slides with 11,5 min of notes (target 8-10 frames, ~10 min). `check_bundle.py` warns; trimming is the
  user's call (candidates: the phase-5 table, the phase-1 standardisation paragraph, smaller figures).
- **Template additions** (outside the template's own files): `generated/results_macros.tex` loads
  `generated/figure_captions.tex` (`\figcaption{id}`); figures are `figures/<id>.png` (DEC-007). Tables are labelled
  "Cuadro" by babel spanish, so the text says "Cuadro~\ref{...}" (for "Tabla": option `es-tabla`).
- **References:** the report's bibliography is a `thebibliography` at the end of `report_06_conclusions.tex` (19
  entries since CHG-018). All come from LOGBOOK.md ("Referencias externas"), written from memory and **not verified**
  against the sources. [R31] (SGDR) and [R32] (AdamW) are now listed there too.
- `prediction.zip` in the project root (re-made by the user on 2026-10-03 at 23:33) holds a `prediction.json` identical
  to the current one: the delivery-1 model (EXP-010, run `EXP-010_20261003-160510`).
- **Git tag** `delivery-01`: to be created when the delivery is closed.

## Discrepancies between LOGBOOK.md and the data (the documents use the data)

| Quantity | LOGBOOK.md | Data (macro) | Why |
|---|---|---|---|
| Gain of `Rescaling` (EXP-002 vs EXP-001) | +8,3 pp | +8,4 pp (`gain-normalisation`) | LOGBOOK subtracted rounded values (0,365 − 0,282); CSV: 0,3654 − 0,2817 |
| Speed-up of batch 256 | ×15 | ×16,4 (`speedup-batch256`) | LOGBOOK used one seed (10,6 / 0,7 s); the macro averages the 3 seeds of each configuration |

## Statements without a direct number in experiments/

- Inactive units of phase 2: now in `history/EXP-005_*.json` (`val.inactive_units`, CHG-017), measured on the saved
  models with the function used by every later run. No longer a LOGBOOK-only statement.
- "Un paso puede desplazar un logit miles de unidades" (baseline; slide note): estimate from EXP-000 in LOGBOOK.md,
  lr · number of inputs · mean pixel ≈ 1e-3 · 150 528 · 62 ≈ 9 300. The mean pixel is not in `experiments/`.
- "Las entradas estandarizadas, de mayor magnitud media" (phase 1): LOGBOOK EXP-004 (mean |x| ≈ 0,8 vs 0,24); the
  number is not in `experiments/`, so the text gives no value. The loss of the first epoch (60,2 vs 13,6) is a macro.
- "La learning rate óptima baja unas tres veces cada vez que se dobla el lado" (phase 2): the grid values themselves
  (1e-3, 3e-4, 1e-4 are macros); "tres veces" is their ratio.
- "Nesterov no da una ventaja consistente" (phase 5.2): EXP-009 table (0,559 with 0,9 vs 0,582 without; 0,572 with
  0,95 vs 0,508 without). Qualitative in the text; the 0,95 Nesterov row is in the phase-5 table.
- Counts written as words ("doce arquitecturas", "seis alternativas", "cinco pasos"): number of runs in
  `experiments.csv` (EXP-006: 12; EXP-007: 6 + base; phase 5: 5.1-5.5).
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
- Report and slides show the same 9 figures in the same order (DEC-010, updated by CHG-018): class_distribution,
  normalisation, resolution_sweep, capacity, activation, lr_sweep, progress, stopping, per_class_recall.
  `normalisation` shows the training loss only: the validation loss of `EpochMetrics` is capped at ~16 (ISS-007).
- `activation` (phase 4, two panels): left, inactive units of every phase-2 run by resolution and lr (CHG-017);
  right, the 7 activations of EXP-007 (LeakyReLU 0,3 with its 3 seeds) with the inactive units of each model.
- Slide captions are the short `caption_slides_es`; `check_bundle.py` estimates that each fits in 2 lines, but only the
  Overleaf compilation confirms it. Check every slide once there.

## Other

- No previous delivery: the "entrega anterior → ahora" comparison starts with delivery 2.
- EXP-000 (original split) is excluded from every table and figure; the baseline is EXP-001.
- `hyperparameter_sweep.ipynb` still shows all 9 phases (it is the full record of the sweep).
- **User edits (2026-10-04, CHG-018):** the six report sections edited by hand in `overleaf/sections/` became the
  drafts before being extended, so they are kept. Their `% TODO` comments are resolved with data (see CHG-018). One
  sentence was corrected: "no se separó el efecto de la activación del de la inicialización" (phase 4). The ReLU base
  (`EXP-006_h256-256-256-256`) already used `he_normal` (its config has `init: None`, which `build_model` maps to
  `he_normal` for ReLU), so between ReLU and LeakyReLU only the activation changes.
