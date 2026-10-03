# LOGBOOK: xview_recognition

Clasificación de 13 tipos de objeto en recortes 224×224 de imágenes aéreas de xView [R7].

Bitácora viva del notebook [ffNN_example.ipynb](ffNN_example.ipynb): todo cambio, experimento, idea, incidencia y decisión, con su justificación teórica. Es la memoria del proyecto: si no está aquí, no pasó.

**Convenciones**
- IDs: `EXP-###` experimento · `CHG-###` cambio de código sin efecto en el modelo · `ISS-###` incidencia · `IDEA-###` idea pendiente · `DEC-###` decisión.
- Citas: `[C#]` documentos del curso en [references/](references/README.md) (con página o sección), `[R#]` referencias externas (al final).
- Métricas crudas por run: [experiments/runs.csv](experiments/runs.csv) y `experiments/runs/<run_id>/` (`history.csv`, `confusion_matrix.csv/png`, `config.json`, `model_summary.txt`). El `EXP_ID` de la celda de tracking del notebook enlaza cada run con su entrada aquí.
- Un experimento cambia **un solo factor** respecto a su experimento base, para poder atribuirle el efecto.
- Lo que no se ha ejecutado figura como *pendiente*; nunca se anotan resultados de memoria.

---

## Estado actual

| | |
|---|---|
| Etapa del curso | ffNN (red feedforward) |
| Entorno | Celda 1 original restaurada (CHG-004): con el kernel global, Keras usa TensorFlow y entrena en CPU. El `.venv` con PyTorch + CUDA sigue instalado, pero el notebook ya no lo activa (ISS-003 reabierta) |
| Mejor experimento | EXP-000 (run `EXP-000_20261003-025155`) |
| Mejor val accuracy / mean recall | 0,385 / 0,340 (época 17 de 20; estimación optimista, ver EXP-000) |
| Referencia trivial | Clase mayoritaria (Building): 19,2 % de accuracy · azar: 7,7 % de mean recall (1/13) |
| Diagnóstico | Sin overfitting (val ≥ train). El cuello de botella es la optimización: softmax saturada (pérdida ~3 000) y val acc que oscila ±5 pp entre épocas |
| Próximo paso | IDEA-009 (caché del dataset, CHG: los runs pasarían de ~70 min a minutos) → EXP-001 = IDEA-003 (semilla y split estratificado, nueva base) → EXP-002 = IDEA-001 (normalizar la entrada) |

## Índice de experimentos

| ID | Fecha | Etapa | Cambio respecto a la base | Base | val acc | mean recall | Run | Estado |
|----|-------|-------|---------------------------|------|---------|-------------|-----|--------|
| EXP-000 | 2026-10-03 | ffNN | Baseline del enunciado, sin cambios | — | 0,385 | 0,340 | `EXP-000_20261003-025155` | ejecutado |

---

## Experimentos

### EXP-000: Baseline del enunciado
- **Fecha:** entrada 2026-10-02, ejecutado 2026-10-03 · **Etapa:** ffNN · **Base:** — · **Estado:** ejecutado
- **Hipótesis:** el modelo del enunciado es una cota inferior; sirve para medir cuánto aporta cada mejora posterior.
- **Configuración:** `Flatten(224×224×3) → ReLU → Dense(13) → softmax` (1 956 877 parámetros) · Adam con AMSGrad [R4] (lr = 1e-3, clipnorm = 1.0) · batch 16 · 20 épocas · split 90/10 aleatorio (`random_state=1`, 16 871 / 1 875 imágenes) · píxeles crudos en [0, 255], sin augmentation · **sin semilla fija** (no es reproducible exactamente).
- **Ejecución:** notebook completo, sin interfaz (nbclient, fuera del repo), con el kernel global: Python 3.12, TF 2.20, **CPU** (celda 1 original, CHG-004). En las salidas guardadas solo se colapsaron los redibujados `\r` de la barra de progreso, para dejar lo mismo que se vería en pantalla. `prediction.json` (test, 2 365 imágenes) se generó con el modelo de la época 17. El test no tiene etiquetas locales: su puntuación solo se obtiene en la competición.
- **Fundamento:** el ReLU sobre píxeles no negativos es la identidad (ver ISS-001), así que el modelo es una **regresión logística multinomial** sobre 150 528 píxeles: fronteras de decisión lineales en el espacio de píxeles, sin invarianza alguna a traslación ni rotación [R8 §6.1]. No se espera que generalice bien. La accuracy debe compararse con el 19,2 % de la clase mayoritaria y el mean recall con el 7,7 % del azar.
- **Resultados** (run `EXP-000_20261003-025155`, mejor época 17 de 20):
  - **val acc 0,385 · mean recall 0,340 · mean precision 0,286 · min recall 0,000** · train acc en la época 17: 0,371.
  - **Recall por clase** (casos en validación): Small car 0,81 (329) · Excavator 0,79 (81) · Pylon 0,56 (32) · Building 0,43 (363) · Storage tank 0,41 (152) · Cargo plane 0,41 (59) · Motorboat 0,30 (109) · Shipping container 0,30 (135) · Dump truck 0,20 (118) · Bus 0,20 (174) · **Truck 0 (241) · Fishing vessel 0 (73) · Helipad 0 (9)**.
  - **Truck y Fishing vessel no se predicen nunca.** Clases sobrepredichas: Small car (527 predicciones para 329 casos), Excavator (281 para 81) y Storage tank (288 para 152).
  - **Confusiones principales** (porcentaje de la clase real): Dump truck → Excavator 47 % · Motorboat → Small car 38 % · Bus → Small car 35 % · Truck → Small car 32 % · Building → Storage tank 19 % · Truck → Excavator 17 % · Truck → Storage tank 17 %.
  - **Curvas** (`history.csv`): la train acc sube despacio de 0,26 a 0,37. La pérdida se queda entre ~3 100 y ~4 300 en train y entre ~2 650 y ~5 700 en val durante todo el entrenamiento; lo razonable sería ≈ ln 13 ≈ 2,6 o menos. La val acc oscila entre 0,18 y 0,39 de una época a otra (σ = 0,048; media de las épocas 11–20: 0,317). La lr se mantuvo en 1e-3 las 20 épocas: ningún callback actuó (ISS-002 confirmada).
  - **Tiempos:** 70 min de entrenamiento (época 1: 446 s, 422 ms/paso con lecturas en frío; épocas 2–20: 111–401 s, media de 197 s, irregular). Predicción de validación: 133 s. Test: 174 s. RAM libre al empezar: 1,5 GB (ISS-005).
- **Conclusión:**
  - **Hipótesis confirmada:** es una cota inferior. Duplica la accuracy de la clase mayoritaria (0,385 frente a 0,192) y multiplica por 4,4 el mean recall del azar (0,340 frente a 0,077).
  - **No hay overfitting** (val ≥ train): el límite es de sesgo y, sobre todo, de **optimización**. La regularización (IDEA-007) no es prioritaria todavía.
  - **Por qué la pérdida es enorme y la val acc oscila:** con entradas no negativas, todos los pesos de una fila reciben gradientes del mismo signo [R1 §4.3]. Adam da pasos de tamaño ≈ lr por parámetro, sea cual sea la escala del gradiente (por eso `clipnorm=1.0` no lo limita) [R4 §2.1]. Un solo paso puede mover un logit unos 10⁻³ × 150 528 × 62 ≈ 9 300 (estimación con la media de píxel real). Con logits de ~10³, la softmax está saturada y la cross-entropy ≈ max_j z_j − z_y (log-sum-exp ≈ max). Solo aprenden los ejemplos mal clasificados, con un gradiente de magnitud fija: el entrenamiento se comporta como un perceptrón multiclase, que no converge si las clases no son linealmente separables [R17 §4.1.7]. Por eso la frontera salta de época en época. La oscilación (σ = 0,048) es unas 4 veces el error muestral de la validación (√(0,385·0,615/1875) ≈ 0,011), así que es inestabilidad del modelo, no ruido de la muestra. Todo esto apunta a **IDEA-001 como el cambio con más impacto esperado**.
  - **Clases que nunca se predicen:** que Helipad no se prediga encaja con el desbalance (IDEA-005). Truck, en cambio, es la 3.ª clase más frecuente (2 210 imágenes), así que su recall 0 no se explica por el desbalance. Hipótesis (sin verificar): con la softmax saturada, la fila de logits de Truck pierde siempre frente a las de Small car, Excavator y Storage tank, hacia las que van sus errores; para un clasificador lineal sobre píxeles crudos, esas clases ocupan regiones solapadas. Comprobarlo tras IDEA-001: si con la entrada normalizada Truck empieza a predecirse, la causa era la optimización.
  - **El 0,385 es optimista:** es el máximo de 20 épocas muy ruidosas, elegido y medido sobre la misma validación [R18]. Un valor más representativo es la media de las épocas 11–20 (0,317). Hasta que se estabilice el entrenamiento, comparar experimentos por su mejor época exagera las diferencias.
- **Siguiente paso:** IDEA-009 (caché, un CHG que no cambia el modelo) para pasar de ~70 min a minutos por run. Después EXP-001 = IDEA-003: el split estratificado cambia el conjunto de validación, así que EXP-001 pasa a ser la nueva base y los números de EXP-000 no son comparables con los siguientes. Luego EXP-002 = IDEA-001.

---

## Registro de cambios (CHG)

### CHG-001: Infraestructura de la bitácora (2026-10-02)
Creados `CLAUDE.md`, `LOGBOOK.md`, `references/`, `.gitignore`, `.gitattributes`, `tools/` (textconv para diffs de notebooks y `setup.sh`), `.githooks/pre-commit` y, en `.claude/`, los hooks de Claude Code y los skills `log-experiment` y `next-ideas`. Repositorio git inicializado en esta carpeta. Ver DEC-001 a DEC-003.

### CHG-002: Instrumentación de runs en el notebook (2026-10-02)
No cambia el modelo ni el entrenamiento: solo guarda evidencias de cada run.
- Nueva celda antes de los callbacks: `EXP_ID`, `run_id` y `run_dir`; guarda `model_summary.txt` y la configuración del optimizador.
- Callbacks: se añade `CSVLogger` → `history.csv` (loss/accuracy/lr de train y val por época).
- `draw_confusion_matrix(..., save_path=None)` guarda el PNG de la matriz de confusión.
- Nueva celda tras las métricas de validación: guarda `confusion_matrix.csv` y `config.json`, y añade una fila resumen a `experiments/runs.csv` (`val_accuracy`, `val_mean_recall`, `val_mean_precision`, `val_min_recall`, `train_accuracy` en la mejor época, etc.).
- Probado con un modelo diminuto y datos sintéticos. Ejecutar dos veces la celda de resumen duplica la fila: se toma `run_id` como clave.

### CHG-003: Entrenamiento en GPU con Keras 3 sobre PyTorch (2026-10-03)
No cambia el modelo ni el entrenamiento: misma API de Keras, otro motor de cálculo. Resuelve ISS-003; la elección está en DEC-004.
- **Entorno:** nuevo `.venv/` en la raíz del proyecto (Python 3.12, no versionado) con `torch 2.6.0+cu124`, `keras 3.13.0`, `numpy 2.5.3`, `rasterio 1.5.2`, `scikit-learn 1.9.1`, `matplotlib 3.11.2` e `ipykernel`. Registrado como kernel de Jupyter **"Python 3.12 (image_recognition, GPU)"**. El Python global no se toca.
- **Notebook:** la primera celda fija `os.environ['KERAS_BACKEND'] = 'torch'` antes de importar Keras e imprime el backend y la GPU. Los imports `tensorflow.keras.*` de las celdas del modelo, el optimizador y los callbacks pasan a `keras.*`.
- **`.gitignore`:** se añade `.venv/`.
- **Probado** (2026-10-03): `torch.cuda.is_available()` devuelve `True` y los pesos se crean en `cuda:0`. Se ejecutaron todas las celdas de código hasta el resumen del run con un 2 % de train, 2 épocas de 10 pasos y salidas en una carpeta temporal: `fit` con el generador, `ModelCheckpoint`, `ReduceLROnPlateau`, `CSVLogger`, `load_weights`, la matriz de confusión y la fila de `runs.csv` funcionan. No se escribió nada en `experiments/`.
- **Rendimiento medido** con el modelo de EXP-000 y batch 16 (1 600 imágenes de train al azar, con las imágenes ya en memoria): un paso de entrenamiento tarda **13,8 ms en GPU frente a 26,4 ms en CPU** (×1,9). Pero cargar las 16 imágenes de un batch con `load_geoimage` cuesta 63 ms con la caché de disco caliente y 232 ms en frío, así que la carga domina el tiempo por paso (ver ISS-005).
- **Revertido en parte por CHG-004:** la celda 1 vuelve a la original. Los imports `keras.*`, el `.venv` y el kernel registrado se mantienen.

### CHG-004: Se restaura la celda 1 original (2026-10-03)
A petición del usuario, la celda 1 vuelve a ser `import tensorflow as tf` / `print(tf.config.list_physical_devices('GPU'))`, y se borra su salida anterior. El resto de CHG-003 se mantiene: los imports `keras.*` de las celdas del modelo, el optimizador y los callbacks, el `.venv` y el kernel registrado.
- **Efecto:** sin `KERAS_BACKEND`, Keras 3 usa su backend por defecto, TensorFlow [R14]. Con el kernel global (TF 2.20) el notebook vuelve a entrenar en CPU. En TF 2.20, `keras.*` y `tensorflow.keras.*` son el mismo Keras 3.13, así que el modelo es el mismo. Con el kernel del `.venv` no funciona: allí no está instalado TensorFlow.
- **Contexto:** justo antes, el notebook con CHG-003 se había ejecutado con el kernel global en lugar del `.venv`. La celda 1 detectó la GPU, pero la celda del optimizador falló con `TypeError: len() of unsized object` en `keras/src/backend/torch/core.py`, porque el torch 2.3.1 global no puede usar su numpy 2.2.6 ("Failed to initialize NumPy: _ARRAY_API not found"). Con el kernel del `.venv` ese error no ocurre (prueba de CHG-003).

### CHG-005: Ejecutor de notebooks sin interfaz en `tools/` (2026-10-03)
`tools/run_notebook.py` es la versión generalizada del ejecutor temporal con el que se corrió EXP-000. No toca el notebook ni el modelo.
- Ejecuta las celdas en orden desde la carpeta del notebook, con el kernel del intérprete indicado en `--python` y un kernelspec desechable en un directorio temporal: no toca los kernels de Jupyter del usuario. Imprime un log por celda y guarda una copia de progreso tras cada una.
- Colapsa los redibujados `\r`/`\b` de la barra de progreso de Keras, que si no inflarían el notebook. Conserva el `kernelspec` original.
- Solo sobrescribe el notebook si no cambió durante la ejecución. Si falla una celda, el notebook queda intacto y las salidas parciales quedan en el temporal.
- Dependencias: `nbclient` y `nbformat`, instaladas en el `.venv` (añadidas a las instrucciones de `CLAUDE.md` para recrearlo). El Python global no se toca.
- **Probado** (2026-10-03) con notebooks pequeños y el kernel global: éxito con barra de progreso (queda "step 5/5", el kernel es el Python global y el directorio de trabajo es el del notebook), fallo en la 2.ª celda (notebook intacto, exit 1) y notebook modificado durante la ejecución (no se sobrescribe). En Windows fija `WindowsSelectorEventLoopPolicy`, porque `zmq` no funciona con el event loop Proactor.

---

## Incidencias (ISS)

### ISS-001: El ReLU tras `Flatten` no hace nada · abierta
`Activation('relu')` se aplica directamente a los píxeles, que son `uint8` en [0, 255] (verificado en 200 imágenes de train). ReLU(x) = x para x ≥ 0, así que no hay ninguna no linealidad y el modelo es lineal (EXP-000). Probablemente faltaba una `Dense` antes del ReLU. **Resolución:** IDEA-002.

### ISS-002: Los callbacks no llegan a actuar con 20 épocas · abierta
`EarlyStopping(patience=40)` no puede dispararse con `epochs=20`. `ReduceLROnPlateau(patience=10)` puede actuar como mucho una vez (en la época 11 o más tarde). Además, ambos vigilan `val_accuracy`, que es una señal ruidosa y escalonada. Vigilar `val_loss` es más suave. **Confirmada en EXP-000:** la lr fue 1e-3 las 20 épocas. La mejor val acc llegó en la época 17, así que ReduceLROnPlateau no tuvo margen para actuar, y EarlyStopping no podía. **Resolución:** IDEA-004.

### ISS-003: No hay GPU · reabierta (CHG-004; se había resuelto en CHG-003)
`tf.config.list_physical_devices('GPU')` devuelve `[]`. TensorFlow 2.10 fue la última versión con soporte de GPU en Windows nativo [R13], y el kernel usa TF 2.20, así que todo se entrena en CPU aunque CUDA 12.5 esté instalado. Esto condiciona el coste de cada experimento. **Opciones:** Google Colab (el notebook viene de Colab según sus metadatos), WSL2 con `tensorflow[and-cuda]`, o reducir el coste por época (IDEA-006). **Resolución:** Keras 3 sobre PyTorch con CUDA (CHG-003, DEC-004). El `torch 2.3.1+cu118` del Python global sí veía la GPU, pero falla al importar con su `numpy 2.2.6` ("A module that was compiled using NumPy 1.x cannot be run in NumPy 2.2.6"), por eso se usa un entorno aparte. **Reabierta** porque CHG-004 restaura la celda 1 original, que ya no activa el backend torch. Para volver a la GPU hay que fijar `os.environ['KERAS_BACKEND'] = 'torch'` antes del primer `import keras` y usar el kernel del `.venv`.

### ISS-004: Comentario engañoso sobre "resampling" · abierta (menor)
En la celda que construye `anns` aparece `# Resampling strategy to reduce training time`, pero no hay ningún remuestreo implementado. Hay que eliminarlo o implementar lo que promete (relacionado con IDEA-005).

### ISS-005: La carga de imágenes domina el tiempo de entrenamiento · abierta
El generador abre con rasterio cada TIF de 224×224 en cada batch y en cada época, en el mismo hilo que el entrenamiento. Medido (CHG-003): 63 ms por batch de 16 con la caché de disco caliente y 232 ms en frío, frente a 13,8 ms del paso de entrenamiento en GPU. Por la ley de Amdahl [R16], acelerar solo el cálculo da como mucho (63 + 26,4) / (63 + 13,8) ≈ ×1,16 por paso con la caché caliente, aunque la GPU haga el cálculo ×1,9 más rápido. En la prueba de CHG-003, con lecturas en frío, el paso completo fue de unos 240 ms. Mientras no se resuelva, la GPU apenas acorta las épocas de esta ffNN. Con modelos más pesados (un MLP grande o una CNN) el cálculo pesará más y la GPU se notará más, pero la carga seguirá costando unos 63 ms por paso. **Medido en EXP-000** (CPU, TF 2.20, 1,5 GB de RAM libre): de 105 a 422 ms por paso según la época, y 70 min en total. Las épocas varían mucho (111–446 s) sin ningún cambio en el cálculo, lo que delata que el tiempo lo marca la E/S y la caché de disco. **Resolución:** IDEA-009.

---

## Ideas pendientes (backlog)

Prioridad = impacto esperado / coste. Al probar una idea se crea su EXP y la idea pasa a *probada → EXP-###*.

### IDEA-001: Normalizar la entrada · prioridad alta · coste mínimo
- **Qué:** escalar los píxeles a [0, 1], o estandarizar por canal con la media y la desviación típica de train.
- **Por qué:** (1) si todas las entradas son positivas, los gradientes de los pesos que llegan a una neurona tienen todos el mismo signo y el descenso avanza en zigzag; centrar en cero mejora el condicionamiento [R1 §4.3]. Adam reescala cada parámetro (precondicionador diagonal), lo que compensa la escala pero no la correlación que introduce una media no nula. (2) Con `glorot_uniform` y 150 528 entradas, la desviación típica inicial de los logits es ≈ 110 con píxeles crudos frente a ≈ 1,4 con entradas estandarizadas (estimación con las estadísticas reales: media ≈ 62, desviación ≈ 47,5). Con logits ≈ 10² la softmax está saturada desde el principio.
- **Éxito:** la pérdida inicial ≈ ln 13 ≈ 2,56 y una convergencia más rápida que en EXP-000.
- **Evidencia de EXP-000:** pérdida de ~3 000 durante las 20 épocas, val acc oscilando ±5 pp y dos clases frecuentes que nunca se predicen. Es lo esperable con logits saturados (análisis en la conclusión de EXP-000). Es el cambio de mayor impacto esperado. Además de lo anterior, comprobar que la val acc deja de oscilar (σ entre épocas ≪ 0,048).

### IDEA-002: Añadir capas ocultas reales (MLP) e inicialización He · prioridad alta
- **Qué:** `Flatten → Dense(n) → ReLU → … → Dense(13) → softmax`, con `kernel_initializer='he_normal'`.
- **Por qué:** sin capa oculta el modelo es lineal. Un MLP con una sola capa oculta ya es un aproximador universal [R3; R8 §6.4.1]. Para ReLU, la inicialización He conserva la varianza de las activaciones entre capas [R2]; `glorot_uniform` (la de Keras por defecto) está pensada para activaciones simétricas.
- **Cuidado:** con 150 528 entradas, una capa de 512 neuronas tiene ≈ 77 M de parámetros frente a 16 871 imágenes de train: hay alto riesgo de overfitting. En float32, entre pesos, gradientes y los tres estados de Adam con AMSGrad, ocupa ≈ 5 × 308 MB ≈ 1,5 GB, que caben en los 6 GB de la GPU (CHG-003). Combinar con IDEA-006 e IDEA-007.

### IDEA-003: Semilla fija y split estratificado · prioridad alta (prerrequisito)
- **Qué:** `keras.utils.set_random_seed(...)` al inicio y `train_test_split(..., stratify=<labels>)`.
- **Por qué:** sin semilla, la inicialización y el barajado cambian entre runs, y diferencias de ~1 pp pueden ser solo ruido [R9]. Con un split no estratificado, Helipad (111 imágenes en total) tiene unas 11 en validación: la desviación típica de su recall estimado es √(p(1−p)/n) ≈ 0,15 para p = 0,5, así que las métricas por clase son muy ruidosas.
- **Ojo:** estratificar cambia el conjunto de validación, así que el primer run con este cambio (EXP-001) es la **nueva base** y sus números no se comparan con los de EXP-000.
- **En GPU** (CHG-003): con el backend torch, `keras.utils.set_random_seed` también siembra `torch` (lo hace `keras/src/utils/rng_utils.py`). Aun así, algunas operaciones de CUDA no son deterministas; `torch.use_deterministic_algorithms(True)` lo fuerza a costa de velocidad [R15].

### IDEA-004: Ajustar callbacks y número de épocas · prioridad media
- **Qué:** `EarlyStopping(monitor='val_loss', patience≈5–8, restore_best_weights=True)`, `ReduceLROnPlateau(monitor='val_loss', patience≈3)` y `epochs` lo bastante grande para que sea EarlyStopping quien termine.
- **Por qué:** ISS-002. La parada temprana es una forma de regularización [R8 §7.8].

### IDEA-005: Tratar el desbalance de clases · prioridad media
- **Qué:** `class_weight` inversamente proporcional a la frecuencia (o según el "effective number" [R11]), u oversampling de las clases minoritarias en el generador.
- **Por qué:** el desbalance es de 32:1 (Building 3 594 frente a Helipad 111). Sin corregirlo, la pérdida está dominada por las clases frecuentes. En CNNs, el oversampling suele ser la estrategia más robusta [R10]. La métrica guía debe ser el mean recall (balanced accuracy), no la accuracy.
- **Evidencia de EXP-000:** Helipad (minoritaria) tiene recall 0, pero también Truck, que es la 3.ª clase más frecuente. No todo es desbalance: aplicar esta idea después de IDEA-001 para no confundir los dos efectos.

### IDEA-006: Reducir la resolución de entrada para la ffNN · prioridad media
- **Qué:** redimensionar a 64×64 o 32×32 antes de `Flatten`.
- **Por qué:** divide la dimensión de entrada por ≈ 12 o ≈ 49, y con ella los parámetros de la primera capa: menos varianza (mejor relación datos/parámetros) y menos cálculo por paso. Esto último importa poco ahora que hay GPU (CHG-003) y que la carga de imágenes domina el tiempo por paso (ISS-005), porque el redimensionado se hace después de leer cada imagen. **Coste:** se pierde detalle en objetos pequeños (Small car, Pylon).

### IDEA-007: Regularización (dropout, L2, BatchNorm) · prioridad media, tras IDEA-002
- **Por qué:** cuando el MLP tenga capacidad de sobreajustar (hueco entre `train_accuracy` y `val_accuracy` en `runs.csv`). Dropout [R5], penalización L2 [R8 §7.1], BatchNorm para estabilizar y acelerar el entrenamiento [R6].

### IDEA-008: Data augmentation con rotaciones de 90° y flips · prioridad media (clave en la etapa CNN)
- **Por qué:** en imágenes aéreas no hay un "arriba" canónico, así que las 8 transformaciones del grupo diedral D4 conservan la etiqueta. Es aumentar los datos codificando una invarianza conocida [R8 §7.4]. Para una ffNN, que no comparte pesos espacialmente, el beneficio será menor que en una CNN.

### IDEA-009: Leer cada imagen una sola vez (caché del dataset) · prioridad alta · coste bajo
- **Qué:** (a) volcar las imágenes a un único `.npy` en `uint8` y leerlo con `np.load(..., mmap_mode='r')`, o cargarlo entero en RAM. Ocupa 18 746 × 150 528 B ≈ 2,8 GB. (b) Como alternativa, o además, pasar el generador a `keras.utils.PyDataset` con `workers > 1` para que la carga se solape con el cálculo.
- **Por qué:** ISS-005. La parte serie (la E/S) limita la aceleración total [R16]. Con la caché, el paso bajaría hacia los ~14 ms del cálculo en GPU: unas 1 055 × 14 ms ≈ 15 s de entrenamiento por época, más la validación.
- **Coste/riesgo:** 2,8 GB de disco para el `.npy`. El equipo tiene 15,7 GB de RAM, pero solo había unos 2,5 GB libres al medir, así que es mejor memmap que cargarlo todo. Es un cambio del pipeline, no del modelo (CHG): hay que comprobar que los batches salen idénticos (mismos píxeles, mismo orden con la misma semilla) para que los runs sigan siendo comparables.
- **Éxito:** tiempo por paso cercano a 14 ms en GPU, medido en la barra de progreso de `fit` o en `history.csv`.

### Etapas futuras del curso (para no perderlas de vista)
- **CNN:** conectividad local, compartición de pesos y equivarianza a traslaciones reducen los parámetros en órdenes de magnitud frente a la ffNN [R8 §9.2].
- **Arquitecturas conocidas desde cero frente a transfer learning:** la transferibilidad de las features baja con la distancia entre dominios [R12]. El paso de ImageNet (fotos naturales) a imágenes aéreas es un salto de dominio grande, así que conviene comparar congelar capas frente a fine-tuning completo.

---

## Decisiones (DEC)

### DEC-001: La raíz del proyecto es `image_recognition/` (2026-10-02)
Aquí están el repositorio git, `CLAUDE.md`, la bitácora y la documentación. No se usa material de la carpeta padre: la documentación necesaria se añade a `references/`.

### DEC-002: Tracking de runs en CSV, no MLflow/W&B (2026-10-02)
Un CSV y una carpeta por run: sin dependencias, versionable y legible por Claude. Se revisará si el número de runs lo hace inmanejable (alternativa: TensorBoard o MLflow).

### DEC-003: Qué no se versiona (2026-10-02)
El dataset (`xview_recognition/`, se descarga del enlace del enunciado), los pesos (`*.keras`, `*.h5`), `prediction.json` y los PDF de `references/` (copyright). Las evidencias ligeras de cada run (`experiments/`) sí se versionan.

### DEC-004: GPU con Keras 3 sobre PyTorch en un `.venv` del proyecto (2026-10-03)
**Decisión:** entrenar con Keras 3 usando el backend PyTorch (`KERAS_BACKEND=torch`) y `torch 2.6.0+cu124`, en un `.venv` propio del proyecto. CUDA 12.4 es compatible con el driver 555.85 (CUDA 12.5).
**Por qué:** PyTorch publica ruedas con CUDA para Windows nativo, y Keras 3 ejecuta el mismo código con TensorFlow, JAX o PyTorch [R14]. El enunciado pide Keras, y el código del notebook sigue siendo Keras; solo cambian los imports. Como aún no hay ningún run, cambiar de backend ahora no rompe la comparabilidad entre experimentos.
**Alternativas descartadas:**
- **WSL2 con `tensorflow[and-cuda]`** [R13]: mantiene TensorFlow, pero obliga a montar otro Python en Ubuntu, conectar VS Code a WSL y leer el dataset desde `/mnt/c` (lento con miles de archivos pequeños) o copiarlo.
- **Google Colab:** GPU gratis, pero con sesiones limitadas y el dataset que subir en cada sesión.
- **Bajar a TF 2.10**, la última versión con GPU en Windows nativo [R13] (evaluado el 2026-10-03 a propuesta del usuario). En PyPI solo hay ruedas de `tensorflow 2.10.1` para cp37–cp310, así que hace falta instalar Python 3.10 (aquí solo hay 3.11 y 3.12). Trae Keras 2.10 en lugar de Keras 3 (`keras<2.11`). Hay que instalar aparte CUDA 11.2 y cuDNN 8.1 [R13], y es anterior a numpy 2, así que en la práctica exige numpy 1.x: sigue haciendo falta un entorno aparte. Además, la RTX 4050 (arquitectura Ada) es posterior a CUDA 11.2, así que dependería de que el driver compile el PTX (sin verificar). El código actual sí funcionaría: en Keras 2.10, `model.keras` se guarda como HDF5 (`is_hdf5_filepath` en `keras/saving/saving_utils.py`). Pero quedaría anclado a una API de 2022 durante el resto del curso.
- **Actualizar el torch del Python global:** `torchvision 0.18.1` y `torchaudio 2.3.1` fijan `torch==2.3.1+cu118`, así que habría que actualizarlos también, y podría afectar a otros proyectos que usan ese Python.

**Riesgo:** si la asignatura exige TensorFlow como motor y no solo la API de Keras, hay que pasar a WSL2. El código no cambia: basta con quitar `KERAS_BACKEND` y volver a `tensorflow.keras`.

### DEC-005: Repositorio remoto en GitHub (2026-10-03)
El remoto es `https://github.com/alexsierra45/image-recognition` (público), elegido por el usuario. Ya tenía un commit inicial con `LICENSE`: el primer commit local se hizo encima de él, sin forzar el push. Al ser público, todo lo versionado (notebook con salidas, bitácora, evidencias de los runs) queda visible. El dataset, los pesos y los PDF siguen fuera (DEC-003).

---

## Referencias externas

- **[R1]** LeCun, Bottou, Orr, Müller (1998). *Efficient BackProp*. En *Neural Networks: Tricks of the Trade*, Springer.
- **[R2]** He, Zhang, Ren, Sun (2015). *Delving Deep into Rectifiers: Surpassing Human-Level Performance on ImageNet Classification*. ICCV.
- **[R3]** Cybenko (1989). *Approximation by superpositions of a sigmoidal function*. MCSS · Hornik (1991). *Approximation capabilities of multilayer feedforward networks*. Neural Networks.
- **[R4]** Kingma, Ba (2015). *Adam: A Method for Stochastic Optimization*. ICLR · Reddi, Kale, Kumar (2018). *On the Convergence of Adam and Beyond* (AMSGrad). ICLR.
- **[R5]** Srivastava et al. (2014). *Dropout: A Simple Way to Prevent Neural Networks from Overfitting*. JMLR.
- **[R6]** Ioffe, Szegedy (2015). *Batch Normalization: Accelerating Deep Network Training by Reducing Internal Covariate Shift*. ICML.
- **[R7]** Lam et al. (2018). *xView: Objects in Context in Overhead Imagery*. arXiv:1802.07856.
- **[R8]** Goodfellow, Bengio, Courville (2016). *Deep Learning*. MIT Press. (§6 MLP, §7 regularización, §8 optimización, §9 CNN)
- **[R9]** Bouthillier et al. (2021). *Accounting for Variance in Machine Learning Benchmarks*. MLSys.
- **[R10]** Buda, Maki, Mazurowski (2018). *A systematic study of the class imbalance problem in convolutional neural networks*. Neural Networks.
- **[R11]** Cui et al. (2019). *Class-Balanced Loss Based on Effective Number of Samples*. CVPR.
- **[R12]** Yosinski et al. (2014). *How transferable are features in deep neural networks?* NeurIPS.
- **[R13]** TensorFlow. *Install TensorFlow with pip* (sección Windows Native: "TensorFlow 2.10 was the last TensorFlow release that supported GPU on native Windows"). https://www.tensorflow.org/install/pip
- **[R14]** Keras. *Getting started with Keras: configuring your backend*. https://keras.io/getting_started/
- **[R15]** PyTorch. *Reproducibility* (notas de la documentación). https://pytorch.org/docs/stable/notes/randomness.html
- **[R16]** Amdahl (1967). *Validity of the single processor approach to achieving large scale computing capabilities*. AFIPS Spring Joint Computer Conference.
- **[R17]** Bishop (2006). *Pattern Recognition and Machine Learning*. Springer. (§4.1.7: el perceptrón no converge si los datos no son linealmente separables)
- **[R18]** Cawley, Talbot (2010). *On Over-fitting in Model Selection and Subsequent Selection Bias in Performance Evaluation*. JMLR 11.

---

## Plantillas

<details>
<summary>EXP</summary>

```markdown
### EXP-###: <título>
- **Fecha:** AAAA-MM-DD · **Etapa:** ffNN/CNN/… · **Base:** EXP-### · **Estado:** pendiente / ejecutado / descartado
- **Hipótesis:** qué se espera y por qué.
- **Cambio:** el único factor que cambia respecto a la base.
- **Fundamento:** teoría con cita [C#, p. X] / [R#].
- **Configuración:** solo lo que difiere de la base.
- **Resultados:** run `<run_id>` · val acc · mean recall · min recall · hueco train-val · observaciones de la matriz de confusión y de las curvas.
- **Conclusión:** ¿se confirma la hipótesis? ¿por qué sí o no?
- **Siguiente paso:**
```
</details>

<details>
<summary>CHG / ISS / IDEA / DEC</summary>

```markdown
### CHG-###: <título> (AAAA-MM-DD)
Qué cambió, dónde (celdas) y por qué. Si afecta al modelo, es un EXP.

### ISS-###: <título> · abierta / resuelta (→ CHG/EXP-###)
Síntoma, causa y evidencia. Resolución.

### IDEA-###: <título> · prioridad alta/media/baja
- **Qué:** · **Por qué:** con cita · **Coste/riesgo:** · **Éxito:**

### DEC-###: <título> (AAAA-MM-DD)
Qué se decidió, alternativas descartadas y por qué.
```
</details>
