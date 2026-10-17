# Review notes: delivery 2

Scope (DEC-009): regularisation of the delivery-1 ffNN (`EXP-010_bs256_sqrt_60ep`) -- norm penalties, Dropout,
BatchNorm, data augmentation; no CNN, no transfer learning. Experiments: phase 6 of the plan (EXP-013, run on the
delivery-1 model) and EXP-019..EXP-021 (augmentation and combinations **without class weights**; class imbalance goes
to delivery 5). Final model: `EXP-021_final` (BatchNorm before the activation, lr 3.6e-3, 90° rotations, longer
training), 3 seeds. Expanded to the level of detail of delivery 1 in CHG-020.

## Pending on the user

- **Compilation in Overleaf** (pdfLaTeX). Compiled locally with Tectonic (XeTeX) in CHG-021: 9 pages, no missing figure/table/macro; tables resized (ISS-011). `check_bundle.py`: 0 errors, 2 warnings (length: ~8.5
  pages estimated against 4-6, and 11 frames / 11 min of notes against 8-10). Left as in delivery 1, where the user
  asked for more detail; trimming is the user's call.
- **Notebook:** `ffNN_example.ipynb` now trains EXP-021 and was run top to bottom (run `EXP-021_20261015-165309`,
  identical to `EXP-021_final`: best epoch 82 of 107, val acc 0.6421, mean recall 0.6600). `model.keras` and
  `prediction.json` (2 365 images, copied to this folder) come from it. Reload it in VS Code.
- **References:** from LOGBOOK.md "Referencias externas" (R5, R6, R7, R8, R9, R18, R32, R33, R34, R35), not
  verified against the PDFs.
- **Git tag** `delivery-02` when the delivery is closed (and `delivery-01` for commit `d557b09`, if wanted).

## Statements to keep in mind

- The train-val gap goes from 14.0 pp (delivery 1) to 6.2 pp with rotations at the same patience, and back to
  12.1 pp with the longer training of the final model: the text says that regularisation improves validation but does
  not remove the gap. Train accuracy in tables and gaps is measured at the end of each run, in inference mode, on the
  whole training set (no augmentation).
- The "overfitting_curves" and "stopping" figures plot the training accuracy *during* each epoch, which for the
  regularised model is measured on augmented images (said in the captions). That is why `stop-gap-*` (from the
  history) is not quoted as the gap; the text quotes `final-gap-acc`.
- BatchNorm alone (EXP-013, +1.3 pp) is a marginal gain (t ~ 2 with 3 seeds, in LOGBOOK, not in the text); "the three
  seeds match or beat the best seed of the start model" comes from EXP-013 in LOGBOOK, not from a macro.
- "No BatchNorm" in the combinations (EXP-020) also goes back to the delivery-1 lr (1.2e-3): two changes, said in the
  text.
- Rotations, flips and D4 tie within noise; rotations were chosen for the best mean of both metrics.
- Helipad (11 validation images) goes from 0.303 to 0.697: said to be very noisy.
- Two explanations are flagged in the text as not verified: why brightness/contrast hurts (colour is a large part of
  the information at 16x16) and why Truck does not improve (few pixels; rotating adds nothing). The Dropout + BN
  variance-shift explanation [li2019] is given as "a possible cause".
- The seed noise (`seed-noise-pp`) is pooled over every variant with 3 seeds in `experiments.csv`.

- EXP-022 (label smoothing) and EXP-023 (average of the 3 seeds) were added on 2026-10-15 (CHG-022). The average is a
  single measurement, +0.6 pp below the seed noise: the text calls it not conclusive. The final model stays EXP-021.
- The confusion matrix is the mean of the 3 seeds' row-normalised matrices (its own slide, DEC-012). Length after
  CHG-022: ~9.7 pages estimated (10 compiled with Tectonic), 13 frames, 13 min of notes.

- EXP-024 (capacity x regularisation, DEC-013), EXP-025/EXP-027 (shifts, half a pixel) and EXP-026 (TTA) added on
  2026-10-16 (CHG-024). None beats EXP-021; the final model, the notebook and prediction.json do not change. The
  half-pixel control is 1.5 pp below EXP-021 (seed-noise level): the text compares the half pixel with its control.
  Length after CHG-024: ~11.4 pages estimated (12 compiled), 15 frames, 15 min of notes.

- Slides cut to 10 content frames / 10 min of notes (CHG-025); the report keeps everything (~11.6 pages estimated).
  Label smoothing, shifts, the seed average and TTA are only one line of the conclusions slide and the speaker notes.

## Hand-written values that are definitions, not results

Label-smoothing intensities (0.05, 0.1, 0.2), widths 256/512/1024, shift sizes (1, 2 px, half a pixel), 224/16 = 14 = 7 x 2, rotations of 90°, the 8 symmetries of D4, `[0, 1]`, the architecture 1024-512-256 of the capacity example, step
numbers and the target times of the speaker notes. Everything else goes through `\res{}`.
