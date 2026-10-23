# CLAUDE.md: image_recognition

## Contexto
Proyecto de la asignatura de Visión por Computador (MUIA), trabajo "xview_recognition": clasificar 13 tipos de objeto en recortes 224×224×3 de imágenes aéreas (xView). El notebook [xview_recognition.ipynb](xview_recognition.ipynb) (hasta la entrega 2, `ffNN_example.ipynb`: DEC-014) irá ganando complejidad con el curso: ffNN → CNN → arquitecturas conocidas (VGG, ResNet, GoogLeNet…) entrenadas desde cero frente a transfer learning. Cada entrega deja su versión exacta en una etiqueta de git (`delivery-01`, `delivery-02`…).

**Calendario de entregas del curso** (DEC-009). Cada entrega parte del modelo final de la anterior y solo muestra lo que pide su enunciado. El tratamiento del desbalance de clases ya no está reservado a la entrega 5: se puede usar con normalidad (actualización de DEC-009, 2026-10-17):
1. **ffNN:** arquitectura, algoritmo de optimización y parámetros de entrenamiento; la validación ajusta los hiperparámetros y decide el punto de parada (DEC-008: fases 0-5, final `EXP-010_bs256_sqrt_60ep`).
2. **Regularización** de esa ffNN para reducir el sobreajuste: penalizaciones de norma, Dropout, BatchNorm, data augmentation. Sin CNN ni transfer learning.
3. **CNN propias desde cero**, comparadas con arquitecturas conocidas (GoogLeNet, VGG, ResNet…) también desde cero.
4. **Transfer learning** con esas arquitecturas preentrenadas, comparado con la entrega 3.
5. **Experimentos pendientes** de las fases anteriores y modelo final refinado.

Esta carpeta es la raíz del proyecto. **No usar material de la carpeta padre.** La documentación que haga falta la añade el usuario a [references/](references/README.md).

## Entorno
- **Ahora (CHG-029): supercomputador Magerit (CeSViMa, UPM)**, proyecto en `~/image-recognition`. Las reglas generales de Magerit están en `~/.claude/CLAUDE.md`. **Nada de entrenar en el nodo de acceso** (`login*`: sin CUDA, 10 min de CPU por proceso). Todo run va por SLURM y **cada `sbatch` necesita la autorización del usuario**.
- **`.venv`:** módulo `Python/3.13.5-GCCcore-14.3.0` (tiene RPATH, así que el venv arranca sin cargar el módulo y vale como kernel de VS Code) + `requirements.txt` (TF 2.21, Keras 3.12.4, mismas versiones que CHG-006).
  - Para recrearlo: `module load apps/2021 Python/3.13.5-GCCcore-14.3.0 && python -m venv .venv && TMPDIR=/home/y404/SCRATCH/y404034/tmp .venv/bin/pip install --no-compile -r requirements.txt`, y después añadir a `activate` el `export LD_LIBRARY_PATH` de CHG-006.
  - Activarlo siempre (`source .venv/bin/activate`) antes de ejecutar Python.
- **Lanzar runs:**
  - `sbatch slurm/run.sbatch <script> <args>`: standard-gpu, 1 L40S, 8 CPU, 32 GB, 2 h; se ajusta con opciones de `sbatch`.
  - Notebook: `sbatch --job-name=EXP-0NN slurm/run.sbatch tools/run_notebook.py xview_recognition.ipynb --python .venv/bin/python`.
  - Barrido: `sbatch --job-name=sweep slurm/run.sbatch sweep.py phaseN`.
  - Logs en `experiments/logs/<job>-<id>.out`. Seguimiento con `squeue -u $USER`, `sacct -j <id>` y `history.csv` del run.
  - Comprobar la GPU: `sbatch slurm/gpu_check.sbatch` (debug-gpu, 10 min).
- **Dataset:** `xview_recognition` es un enlace simbólico a `/home/y404/PROJECT/xview_recognition` (solo lectura, publicado por el profesor; no copiarlo, la cuota de 1 TB es del grupo). 18 746 imágenes de train, `uint8` en [0, 255], muy desbalanceado (Building 3 594, Helipad 111). Las rutas del notebook son relativas a esta carpeta.
- **Caché** (CHG-007): `cache/<split>_<res>.npy` (`uint8`, no versionado), creada por `xview_data.load_cache`. `cache` es un enlace simbólico a `/home/y404/SCRATCH/y404034/image-recognition-cache`: la caché se puede regenerar y en SCRATCH no ocupa la cuota del proyecto. En WSL una época a 224 tardaba ~19 s; en Magerit está *pendiente de medir*.
- **Barridos** (DEC-006): `sweep.py phaseN` → `experiment_utils.run_experiment` → `experiments/experiments.csv`, `experiments/history/`, `experiments/models/`, `experiments/plots/`.
- **Antes (CHG-006, hasta 2026-10-22):** WSL2 con una RTX 4050 de 6 GB y Python 3.10; los runs EXP-001..EXP-039 se hicieron ahí. El backend torch de CHG-003/DEC-004 (Windows) ya no se usa.

## Regla principal: la bitácora
[LOGBOOK.md](LOGBOOK.md) es la memoria del proyecto y la fuente de verdad. El usuario quiere que no se pierda nada.

1. **Todo cambio** al notebook o al código se registra en LOGBOOK.md **dentro de la misma tarea**, como EXP, CHG, ISS, IDEA o DEC (ver las convenciones y plantillas del propio archivo). Para hacerlo, usar el skill `log-experiment`.
2. **Toda idea o cambio lleva fundamento teórico citado**: primero los documentos de `references/` (clave `[C#]` y página o sección), si no, papers o libros (`[R#]`, añadiéndolos a "Referencias externas"). Si algo es heurística empírica sin base teórica sólida, se dice explícitamente. No inventar citas: si no hay certeza de una referencia, buscarla o decirlo.
3. **Los resultados salen de evidencias**: `experiments/runs.csv`, `experiments/runs/<run_id>/` o las salidas del notebook. Sin ejecución, el estado es *pendiente*.
4. Un experimento cambia un solo factor respecto a su base, y su base se indica siempre. Diferencias de ~1 pp entre runs no son concluyentes sin semilla fija o varias semillas.
5. Mantener al día "Estado actual" y el "Índice de experimentos". Las incidencias y las ideas no se borran: cambian de estado (resuelta → CHG/EXP, probada → EXP).
6. Antes de terminar una tarea que tocó código, comprobar que la bitácora lo refleja.
7. Las incidencias detectadas al leer el código (bugs, inconsistencias) se registran como ISS aunque el usuario no las haya pedido.

Para proponer los siguientes pasos, usar el skill `next-ideas`.

## Notebook
- El `EXP_ID` de la celda de tracking (antes de los callbacks) debe coincidir con la entrada de LOGBOOK.md. Recordar al usuario que lo cambie antes de cada run.
- El usuario suele tener el notebook abierto en VS Code: al editarlo desde fuera, avisarle de que lo recargue.
- No cambiar el modelo ni el entrenamiento sin que el usuario lo pida: proponer primero (como IDEA) y editar después.
- `git diff` muestra solo el código de las celdas (textconv): usarlo para ver qué cambió.
- **Ejecutar un experimento sin interfaz** (CHG-005, CHG-029): `tools/run_notebook.py` dentro de un trabajo de SLURM (ver "Entorno"), nunca en el nodo de acceso. Para seguir el progreso: el log por celdas, que queda en `experiments/logs/`, y `experiments/runs/<run_id>/history.csv` (una línea por época). El kernel de VS Code corre en el nodo de acceso: sirve para leer y editar, no para entrenar.

## Entregas (informe + slides para Overleaf)
- Sistema en `reports/` (CHG-011): `make bundle D=NN` regenera macros, tablas y figuras desde `experiments/` (solo lectura), instala `drafts/` en `reports/overleaf/delivery_NN/sections/` sin pisar ediciones y genera el zip. Nueva entrega: comando `/new-delivery <n> <baseline>`.
- **Un solo proyecto de Overleaf para todo el curso** (DEC-011): `reports/overleaf/` con `common/` compartido, una carpeta `delivery_NN/` por entrega y sus documentos principales `delivery_NN_report.tex` / `delivery_NN_slides.tex` en la raíz. Las fuentes de cada entrega (`delivery.yaml`, `drafts/`, notas) siguen en `reports/delivery_NN/`. Zip: `reports/overleaf_project.zip`.
- En los `.tex`, ninguna cifra a mano: `\res{clave}` y tablas de `generated/`. Lo que no se pueda rastrear va a `review_notes.md`.
- Figuras: mismo estilo que los experimentos (`plot_theme.py`), textos en inglés, PNG (DEC-007). Texto y pies, en español.
- No hay LaTeX local: se compila en Overleaf (pdfLaTeX).

## Git
- Tras clonar o inicializar el repositorio: `bash tools/setup.sh` (activa `.githooks/pre-commit` y el diff de notebooks).
- El hook pre-commit bloquea commits con `.ipynb`/`.py` si `LOGBOOK.md` no va en el commit. No usar `--no-verify` salvo petición expresa del usuario.
- Antes de cualquier commit o push: revisar `git diff --cached` frente a la bitácora y completar lo que falte.
- Los mensajes de commit citan los IDs de la bitácora, p. ej. `EXP-003: capa oculta de 512 con He init`.
- No versionar el dataset, los pesos (`*.keras`, `*.h5`), `prediction.json` ni los PDF de `references/`.
