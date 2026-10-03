# CLAUDE.md: image_recognition

## Contexto
Proyecto de la asignatura de Visión por Computador (MUIA), trabajo "xview_recognition": clasificar 13 tipos de objeto en recortes 224×224×3 de imágenes aéreas (xView). El notebook [ffNN_example.ipynb](ffNN_example.ipynb) irá ganando complejidad con el curso: ffNN → CNN → arquitecturas conocidas (VGG, ResNet, GoogLeNet…) entrenadas desde cero frente a transfer learning.

Esta carpeta es la raíz del proyecto. **No usar material de la carpeta padre.** La documentación que haga falta la añade el usuario a [references/](references/README.md).

## Entorno
- **Ahora (CHG-004)**: la celda 1 es la original (`import tensorflow as tf`), así que el notebook corre con el **Python 3.12 global** (TF 2.20, Keras 3.13 con backend TensorFlow) y entrena **en CPU**: TF ≥ 2.11 no soporta GPU en Windows nativo. Los imports son `keras.*`, equivalentes a `tensorflow.keras.*` en TF 2.20.
- **GPU disponible pero desactivada** (CHG-003, DEC-004, ISS-003): kernel **"Python 3.12 (image_recognition, GPU)"** = `.venv/` de esta carpeta (no versionado), con Keras 3.13 sobre `torch 2.6.0+cu124` y una RTX 4050 Laptop (6 GB). Para activarla, la primera celda debe fijar `os.environ['KERAS_BACKEND'] = 'torch'` antes de importar Keras, y hay que usar ese kernel; en el `.venv` no hay TensorFlow. En el Python global el backend torch falla: su torch 2.3.1 no funciona con su numpy 2.2.
- Para recrear el `.venv`: `py -3.12 -m venv .venv`, luego `pip install torch==2.6.0 --index-url https://download.pytorch.org/whl/cu124` y `pip install keras==3.13.0 numpy rasterio scikit-learn matplotlib ipykernel nbclient nbformat`.
- **Cuello de botella**: cargar las imágenes (~63 ms por batch de 16) cuesta más que el propio paso de entrenamiento (~26 ms en CPU, ~14 ms en GPU) (ISS-005). Tenerlo en cuenta al estimar el coste de cada experimento.
- Dataset en `xview_recognition/` (no versionado): 18 746 imágenes de train, `uint8` en [0, 255], muy desbalanceado (Building 3 594, Helipad 111). Las rutas del notebook son relativas a esta carpeta.

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
- **Ejecutar un experimento sin interfaz** (CHG-005): `.venv/Scripts/python.exe tools/run_notebook.py ffNN_example.ipynb --python C:/Users/alexs/AppData/Local/Programs/Python/Python312/python.exe`, en segundo plano. El ejecutor necesita `nbclient`, que está en el `.venv`; el kernel usa el intérprete de `--python`, hoy el global, por la celda 1. Para seguir el progreso: el log por celdas que imprime y `experiments/runs/<run_id>/history.csv` (una línea por época). Un run de EXP-000 tardó ~75 min en CPU (ISS-005).

## Git
- Tras clonar o inicializar el repositorio: `bash tools/setup.sh` (activa `.githooks/pre-commit` y el diff de notebooks).
- El hook pre-commit bloquea commits con `.ipynb`/`.py` si `LOGBOOK.md` no va en el commit. No usar `--no-verify` salvo petición expresa del usuario.
- Antes de cualquier commit o push: revisar `git diff --cached` frente a la bitácora y completar lo que falte.
- Los mensajes de commit citan los IDs de la bitácora, p. ej. `EXP-003: capa oculta de 512 con He init`.
- No versionar el dataset, los pesos (`*.keras`, `*.h5`), `prediction.json` ni los PDF de `references/`.
