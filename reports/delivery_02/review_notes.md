# Review notes: delivery 2

Scope (DEC-009): regularisation of the delivery-1 ffNN (`EXP-010_bs256_sqrt_60ep`) -- norm penalties, Dropout,
BatchNorm, data augmentation; no CNN, no transfer learning. Experiments: phase 6 of the plan (EXP-013, run on the
delivery-1 model) and EXP-019..EXP-021 (augmentation and combinations **without class weights**; class imbalance goes
to delivery 5). Final model: `EXP-021_final` (BatchNorm before the activation, lr 3.6e-3, 90° rotations, longer
training), 3 seeds.

## Pending on the user

- **Compilation in Overleaf** (pdfLaTeX); no LaTeX locally. `check_bundle.py`: 0 errors (9 frames, 9 min of notes,
  ~5 pages estimated).
- **Notebook.** `ffNN_example.ipynb` still trains the delivery-1 model (EXP-010). If delivery 2 also needs the notebook,
  it has to be aligned with EXP-021 (BatchNorm, lr 3.6e-3, rotations, patience 25/8) and re-run; not done, because
  delivery 1 has not been closed yet.
- **References:** from LOGBOOK.md, not verified against the sources.
- **Git tag** `delivery-02` when the delivery is closed.

## Statements to keep in mind

- The train-val gap goes from 14.0 pp (delivery 1) to 6.2 pp with rotations at the same epoch budget, and back to
  12.1 pp with the longer training of the final model: the text says that regularisation improves validation but does
  not remove the gap. Train accuracy is measured in inference mode on the whole training set (no augmentation).
- The "overfitting_curves" figure plots the training accuracy *during* each epoch, which for the regularised model is
  measured on augmented images (said in the caption).
- BatchNorm alone (EXP-013, +1.3 pp) is a marginal gain (t ~ 2 with 3 seeds); its large contribution (-4.8 pp
  without it) is measured together with augmentation (EXP-020).
- Rotations, flips and D4 tie within noise; rotations were chosen for the best mean of both metrics.
- The seed noise (`seed-noise-pp`) is pooled over every variant with 3 seeds in `experiments.csv`.

## Hand-written values that are definitions, not results

Rotations of 90°, `[0, 1]`, phase/step numbers and the target times of the speaker notes. Everything else goes through
`\res{}`.
