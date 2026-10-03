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
| Etapa del curso | ffNN (red feedforward), plan de mejora en 9 fases (DEC-006) |
| Entorno | WSL2, `.venv` con TF 2.21 en **GPU** (CHG-006); caché `uint8` del dataset y barridos con `sweep.py` (CHG-007) |
| Fase en curso | Fase 3: arquitectura (EXP-006) |
| Mejor experimento | `EXP-005_r16`: 16×16 `/255` → `Dense(256, relu)` → `Dense(13)`, Adam lr = 1e-3, callbacks ajustados |
| Mejor val accuracy / mean recall | 0,600 / 0,549 (min recall 0,182; split estratificado, no comparable con EXP-000) |
| Referencia | EXP-001 (baseline con el split nuevo): 0,389 / 0,282 · clase mayoritaria: 19,2 % de accuracy · azar: 7,7 % de mean recall |
| Diagnóstico | Brecha train-val moderada (≈ 6 pp de mean recall). La lr óptima depende del número de entradas (EXP-005): reajustarla si cambia la resolución |
| Próximo paso | Fases 2 → 9 del plan (DEC-006) |

## Índice de experimentos

Las métricas de cada variante están en [experiments/experiments.csv](experiments/experiments.csv) (EXP-001 en adelante). Gráficas e interpretación por fase: [hyperparameter_sweep.ipynb](hyperparameter_sweep.ipynb).

| ID | Fecha | Fase | Cambio respecto a la base | Base | val acc | mean recall | Run | Estado |
|----|-------|------|---------------------------|------|---------|-------------|-----|--------|
| EXP-000 | 2026-10-03 | — | Baseline del enunciado, sin cambios | — | 0,385 | 0,340 | `EXP-000_20261003-025155` | ejecutado (split antiguo) |
| EXP-001 | 2026-10-03 | 0 | Semilla 1 y split estratificado | EXP-000 | 0,389 | 0,282 | `EXP-001` | ejecutado (nueva base) |
| EXP-002 | 2026-10-03 | 1 | `Rescaling(1/255)` | EXP-001 | 0,418 | 0,365 | `EXP-002` | ejecutado |
| EXP-003 | 2026-10-03 | 1 | Callbacks ES 10 / RLR 4, 60 épocas; métrica vigilada (mejor: `val_mean_recall`) | CHG-008 | 0,413 | 0,382 | `EXP-003_val_mean_recall` | ejecutado |
| EXP-004 | 2026-10-03 | 1 | Estandarización por canal en lugar de `/255` | EXP-003 | 0,367 | 0,344 | `EXP-004` | ejecutado (descartado) |
| EXP-005 | 2026-10-03 | 2 | Resolución de entrada (8–224) con `Dense(256)`, y rejilla resolución × lr | EXP-003 | 0,600 | 0,549 | `EXP-005_r16` | ejecutado |
| EXP-006 | 2026-10-03 | 3 | Profundidad y anchura de las capas ocultas (16×16) | EXP-005_r16 | | | `EXP-006_h*` | en ejecución |

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

### EXP-001: Baseline con semilla fija y split estratificado (fase 0)
- **Fecha:** 2026-10-03 · **Etapa:** ffNN · **Base:** EXP-000 · **Estado:** ejecutado · **Idea:** IDEA-003 (probada)
- **Hipótesis:** mismo modelo y entrenamiento que EXP-000; solo cambia el conjunto de validación (estratificado) y el run pasa a ser reproducible. Es la referencia del plan de fases (DEC-006).
- **Cambio:** `train_test_split(..., stratify=labels)` y semilla 1 (`keras.utils.set_random_seed` + determinismo de TF, CHG-007). Imágenes desde la caché (CHG-007).
- **Fundamento:** IDEA-003. Con estratificación, cada clase tiene en validación exactamente su 10 % (Helipad: 11 casos frente a los 9 de EXP-000), y la semilla elimina la varianza entre runs que hace inconcluyentes diferencias de ~1 pp [R9].
- **Configuración:** `Flatten → ReLU → Dense(13) → softmax` (1 956 877 parámetros), píxeles crudos, Adam AMSGrad lr = 1e-3 con clipnorm = 1, batch 16, 20 épocas, callbacks originales (ReduceLROnPlateau con paciencia 10, EarlyStopping con paciencia 40, checkpoint sobre `val_accuracy`). Run `EXP-001` en `experiments/experiments.csv`, que es la evidencia de este y de todos los EXP siguientes.
- **Resultados:** mejor época 12 de 20 · **val acc 0,389 · mean recall 0,282 · min recall 0** (Fishing vessel y Helipad nunca se predicen; Motorboat 0,02, Excavator 0,06) · train acc 0,438 (en inferencia) · pérdida de train ~3 200–3 800 durante todo el run · 290 s (14,5 s por época, GPU).
- **Conclusión:** reproduce el comportamiento de EXP-000: softmax saturada (pérdida de ~10³), oscilación de la val acc (σ = 0,041 en las últimas 10 épocas) y clases que no se predicen nunca. La val acc es similar (0,389 frente a 0,385), pero el mean recall es menor (0,282 frente a 0,340): las validaciones son distintas, y con estas oscilaciones el mejor valor de 20 épocas depende mucho de la suerte. No se comparan. **Esta es la línea base del plan.**
- **Siguiente paso:** fase 1 (EXP-002).

### EXP-002: Normalizar la entrada con `Rescaling(1/255)` (fase 1)
- **Fecha:** 2026-10-03 · **Etapa:** ffNN · **Base:** EXP-001 · **Estado:** ejecutado · **Idea:** IDEA-001 (probada; la estandarización por canal va en EXP-004)
- **Hipótesis:** escalar a [0, 1] reduce la magnitud de los logits y desatura la softmax, así que mejora mucho la optimización (IDEA-001, conclusión de EXP-000).
- **Cambio:** `Input(224, 224, 3) → Rescaling(1/255)` como primera capa del modelo. Al ir dentro del modelo, train, validación y test reciben el mismo preprocesado sin tocar el resto del código.
- **Fundamento:** IDEA-001 [R1 §4.3].
- **Resultados:** mejor época 20 de 20 · **val acc 0,418 · mean recall 0,365 (+8,3 pp frente a EXP-001) · min recall 0** (Helipad) · train acc 0,550 · pérdida de train: 13,6 en la época 1 y 4,2 en la 20 (de ~3 500 en EXP-001). ReduceLROnPlateau actuó una vez (lr 1e-3 → 1e-4). 271 s.
- **Conclusión:** **se confirma** en parte. Mejora clara del mean recall; Fishing vessel (0,45) y Motorboat (0,42) pasan a predecirse. Pero la pérdida de la época 1 es 13,6 y no ln 13 ≈ 2,6: con 150 528 entradas en [0, 1] (media ≈ 0,24), un paso de Adam de tamaño ≈ lr por peso mueve un logit hasta ≈ 1e-3 × 150 528 × 0,24 ≈ 36, así que la softmax se vuelve a saturar enseguida [R4 §2.1]. Apunta a que lr = 1e-3 es alta para este modelo (fase 5.1) y a que centrar la entrada debería ayudar (EXP-004). La val acc sigue oscilando (σ = 0,064 en las últimas 10 épocas) y la mejor época es la última: el entrenamiento no había terminado (ISS-002, EXP-003). Aparece una brecha train-val de 13 pp: el modelo lineal ya empieza a memorizar.
- **Siguiente paso:** CHG-008 y EXP-003.

### EXP-003: Callbacks acoplados a la duración real del entrenamiento y métrica vigilada (fase 1)
- **Fecha:** 2026-10-03 · **Etapa:** ffNN · **Base:** CHG-008 (= EXP-002 sin el ReLU tras `Flatten`) · **Estado:** ejecutado · **Idea:** IDEA-004 (probada) · **Resuelve:** ISS-002
- **Hipótesis:** con paciencias acordes al número de épocas, ReduceLROnPlateau baja la lr cuando la validación se estanca, lo que estabiliza el final del entrenamiento, y EarlyStopping lo termina. Vigilar una señal más suave que `val_accuracy` debería elegir mejor la época.
- **Cambio:** `epochs` 20 → 60, `EarlyStopping` con paciencia 40 → 10 (el mejor modelo se recupera del checkpoint), `ReduceLROnPlateau` con paciencia 10 → 4 (factor 0,1). Se prueban tres métricas vigiladas (checkpoint, EarlyStopping y ReduceLROnPlateau a la vez): `val_loss`, `val_accuracy` y `val_mean_recall`.
- **Fundamento:** parada temprana como regularización [R8 §7.8]; bajar la lr al estancarse reduce el ruido del gradiente estocástico cerca del mínimo [R8 §8.3.1]. Las paciencias (4 y 10) son heurísticas.
- **Resultados:**

| Métrica vigilada | Mejor época / épocas | val acc | mean recall | min recall | train acc | Tiempo |
|---|---|---|---|---|---|---|
| `val_loss` | 50 / 60 | 0,410 | 0,365 | 0,182 | 0,576 | 823 s |
| `val_accuracy` | 10 / 20 | 0,417 | 0,338 | 0,081 | 0,513 | 253 s |
| **`val_mean_recall`** | **24 / 34** | **0,413** | **0,382** | **0,199** | 0,560 | 434 s |

  En las tres, ReduceLROnPlateau actúa varias veces (lr hasta 1e-6 o menos) y la val acc se estabiliza: σ de las últimas 10 épocas 0,001–0,013, frente a 0,064 en EXP-002.
- **Conclusión:**
  - **Se confirma:** los callbacks ya actúan, y con la lr reducida desaparecen las oscilaciones y las clases con recall 0 (min recall de 0 a ~0,19).
  - Vigilando `val_accuracy`, EarlyStopping corta en la época 20 tras un máximo temprano y ruidoso (época 10).
  - `val_mean_recall` da el mejor mean recall (+1,7 pp sobre `val_loss`, poco concluyente con una semilla) y en menos tiempo. **Se elige `val_mean_recall`**, porque además coincide con la métrica de decisión (DEC-006). Contrapartida: la época se elige con la misma métrica con la que se compara, así que la cifra es algo optimista [R18].
- **Siguiente paso:** EXP-004 (estandarizar en lugar de escalar).

### EXP-004: Estandarización por canal en lugar de `/255` (fase 1)
- **Fecha:** 2026-10-03 · **Etapa:** ffNN · **Base:** EXP-003 (`val_mean_recall`) · **Estado:** ejecutado
- **Hipótesis:** centrar en cero y escalar a varianza 1 mejora el condicionamiento [R1 §4.3] y debería superar a `/255`, que deja las entradas en [0, 1] con media ≈ 0,24.
- **Cambio:** `keras.layers.Normalization(mean, variance)` por canal, con la media y la varianza de **train** (sin información de validación), en lugar de `Rescaling(1/255)`.
- **Resultados:** mejor época 12 de 22 · **val acc 0,367 · mean recall 0,344 (−3,8 pp frente a EXP-003) · min recall 0,181** · train acc 0,501 · pérdida de train en la época 1: **60,2** (13,6 con `/255`) · 277 s.
- **Conclusión:** **se refuta** con este optimizador. Con Adam, cada peso se mueve ≈ lr por paso, sea cual sea su gradiente [R4 §2.1], así que el cambio de un logit en un paso escala con Σ|x_i|. Las entradas estandarizadas tienen |x| medio ≈ 0,8, frente a ≈ 0,24 con `/255`: los logits se mueven unas 3 veces más por paso y la softmax se satura antes (pérdida inicial 60 frente a 13,6). La ventaja de centrar queda tapada por una lr demasiado alta. **Se mantiene `Rescaling(1/255)`.** Volver a probar cuando la lr esté ajustada (IDEA-010).
- **Siguiente paso:** fase 2, la resolución de entrada (EXP-005).

### EXP-005: Resolución de entrada (fase 2)
- **Fecha:** 2026-10-03 · **Etapa:** ffNN · **Base:** EXP-003 (`val_mean_recall`) + una capa oculta `Dense(256, relu, he_normal)` · **Estado:** ejecutado · **Idea:** IDEA-006 (probada; primera prueba de IDEA-002)
- **Hipótesis:** a 224×224 la primera capa tiene 150 528 × 256 ≈ 38,5 M de pesos para 16 871 imágenes: mucha varianza. Reducir la imagen (con promedio por área, CHG-007) divide los parámetros de la primera capa por 4 (112), 12 (64) o 49 (32) a cambio de perder detalle fino (Small car, Pylon). Se espera un óptimo en 32–64.
- **Cambio:** solo la resolución de entrada, con la misma arquitectura `Flatten → Dense(256) → Dense(13)` en todas. Rejilla del plan: 32, 64, 112, 224. Ampliada a 8, 16, 24 y 48 porque el óptimo caía en el borde. Además, una rejilla conjunta resolución × lr (16, 32 y 64 con 3–4 lr cada una), porque la accuracy de train también caía con la resolución (ver conclusión).
- **Fundamento:** compromiso sesgo-varianza: menos parámetros libres por ejemplo reducen la varianza del estimador [R8 §5.4]. Un MLP con capa oculta es un aproximador universal [R3; R8 §6.4.1]; He init para ReLU [R2]. El submuestreo por área es un filtro paso bajo, que evita el aliasing [R19, cap. 4].
- **Resultados** (mean recall de validación; ids `EXP-005_r<res>[_lr<lr>]` en `experiments.csv`):

| lr \ resolución | 8 | 16 | 24 | 32 | 48 | 64 | 112 | 224 |
|---|---|---|---|---|---|---|---|---|
| 3e-3 | | 0,510 | | | | | | |
| **1e-3** | 0,499 | **0,549** | 0,535 | 0,497 | 0,469 | 0,427 | 0,325 | 0,129 |
| 3e-4 | | 0,511 | | **0,524** | | 0,463 | | |
| 1e-4 | | 0,342 | | 0,474 | | **0,510** | | |
| 3e-5 | | | | | | 0,395 | | |

  - Mejor run: **`EXP-005_r16`** (16×16, lr = 1e-3, 200 205 parámetros): **val acc 0,600 · mean recall 0,549 · min recall 0,182** (Helipad: 2 de 11) · train acc 0,661, train mean recall 0,609 · mejor época 26 de 36 · 279 s (7,8 s por época).
  - A lr = 1e-3, la **métrica de train baja con la resolución, igual que la de validación**: 0,609 (16) → 0,469 (64) → 0,133 (224). A 224, la pérdida de train se queda en ≈ 2,28 desde la época 2 (≈ la entropía de las frecuencias de clase) y la accuracy en ≈ 0,21 (clase mayoritaria: 19 %): la red ha colapsado a la distribución a priori.
  - Recall por clase en `EXP-005_r16`: Small car 0,83 · Cargo plane 0,77 · Building 0,76 · Excavator 0,67 · Fishing vessel 0,62 · Shipping container 0,59 · Pylon 0,58 · Storage tank 0,52 · Bus 0,45 · Motorboat 0,45 · Dump truck 0,44 · Truck 0,27 · Helipad 0,18.
- **Conclusión:**
  - **Se confirma que conviene reducir mucho la resolución, pero no por el motivo previsto.** A lr fija, el empeoramiento con la resolución es sobre todo un **problema de optimización** (subajuste: train y val bajan juntas), no de varianza. Con Adam, cada peso se mueve ≈ lr por paso [R4 §2.1], así que una preactivación se mueve ≈ lr · Σ|x_i|, que crece con el número de entradas. A 224, unos primeros pasos de ese tamaño son compatibles con dejar muertas casi todas las ReLU (hipótesis, no medida; se comprueba en la fase 4) [R8 §6.3.1]. La lr óptima cae ≈ ×3 cada vez que se dobla el lado (×4 entradas): 16 → 1e-3, 32 → 3e-4, 64 → 1e-4.
  - Comparando cada resolución con su mejor lr, **16×16 sigue ganando** (0,549 frente a 0,524 de 32×32 y 0,510 de 64×64), con el modelo más pequeño y las épocas más rápidas. 8×8 pierde detalle (0,499). La diferencia 16 frente a 32 (2,5 pp) es moderada con una sola semilla.
  - **Se elige 16×16 con lr = 1e-3** como base de las fases siguientes; 32×32 con lr = 3e-4 queda como segundo candidato, a revisar en la fase 9.
  - La primera capa oculta (IDEA-002) ya aporta: a 224, la regresión softmax de EXP-003 daba 0,382, y a 16×16 con `Dense(256)` se llega a 0,549 (dos factores a la vez; no se atribuye por separado).
- **Siguiente paso:** fase 3, la arquitectura (EXP-006).

### EXP-006: Profundidad y anchura de la red (fase 3)
- **Fecha:** 2026-10-03 · **Etapa:** ffNN · **Base:** `EXP-005_r16` · **Estado:** en ejecución · **Idea:** IDEA-002
- **Hipótesis:** con 768 entradas, ensanchar o añadir capas aumenta la capacidad; mientras la brecha train-val sea pequeña (≈ 6 pp en la base), más capacidad debería mejorar la validación hasta que empiece el sobreajuste.
- **Cambio:** solo las capas ocultas: 1 capa (64, 128, 256, 512, 1024), 2 capas (256-128, 512-128, 512-256, 1024-256), 3 en embudo (512-256-128, 1024-512-256) y una estrecha y profunda (256-256-256-256) para comparar con presupuestos de parámetros parecidos.
- **Fundamento:** [R3; R8 §6.4.1]: con una capa basta en teoría, pero la profundidad puede representar algunas funciones con exponencialmente menos unidades [R8 §6.4.1]. Capacidad frente a generalización [R8 §5.2].
- **Resultados:** *pendiente*.

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

### CHG-006: El proyecto pasa a WSL2 con TensorFlow y GPU (2026-10-03)
Hecho por el usuario: clonó el repositorio en WSL2 (Ubuntu, `~/muia/vision-computer/image-recognition`) y creó un `.venv` con `pip install "tensorflow[and-cuda]"`. Al final de `.venv/bin/activate` se añadió un `export LD_LIBRARY_PATH` con las librerías CUDA que instala pip (`site-packages/nvidia/*/lib`). Sin él, TF no encuentra la GPU.
- **Entorno resultante:** Python 3.10.12, TensorFlow 2.21.0, Keras 3.12.4 (backend TensorFlow), cuDNN 9.27, RTX 4050 Laptop (6 GB) visible: `tf.config.list_physical_devices('GPU')` devuelve la GPU en la celda 1 del notebook. Se añadieron al `.venv` `rasterio 1.4.4`, `scikit-learn 1.7.2`, `matplotlib 3.10.9`, `pandas 2.3.3`, `nbclient 0.11.0` y `nbformat`.
- **Efecto:** la celda 1 original (CHG-004) ya entrena en GPU sin ningún cambio de código. Resuelve ISS-003. El backend torch de CHG-003/DEC-004 deja de ser necesario (en este `.venv` no está torch).
- **Dataset:** copiado por el usuario a `xview_recognition/` (ver ISS-006). WSL tiene 7,6 GB de RAM (unos 4–5 GB libres), así que la caché de 224 se lee con memmap (CHG-007).
- `CLAUDE.md` actualizado con el entorno nuevo.

### CHG-007: Caché del dataset e infraestructura de barridos (2026-10-03)
No cambia el modelo: cambia de dónde salen los píxeles y añade la maquinaria del plan de fases (DEC-006). Resuelve ISS-005 e IDEA-009.
- **`xview_data.py`:** lee cada TIF una sola vez con el mismo `load_geoimage` del notebook y lo guarda en `cache/train_<res>.npy` y `cache/test_<res>.npy` (`uint8`, no versionado). Las resoluciones menores se generan desde la de 224 con `tf.image.resize(method='area')`, que promedia los píxeles cubiertos, es decir, filtra paso bajo antes de submuestrear y evita el aliasing [R19, cap. 4]. También incluye el split (`train_test_split(..., random_state=1, stratify=labels)`) y la lista de test filtrada a `*.tif` (ISS-006).
- **`experiment_utils.py`:** `run_experiment(run_id, exp_id, phase, change, cfg)` construye el modelo a partir de un diccionario de configuración (`DEFAULT_CFG`), siembra todo (`keras.utils.set_random_seed`) con `tf.config.experimental.enable_op_determinism()` [R20], entrena con un `PyDataset` que lee de la caché y convierte a `float32` por batch (con augmentation, oversampling y pesos por muestra opcionales), y selecciona la mejor época por `cfg['monitor']` con `ModelCheckpoint`. La validación se calcula una vez por época en un callback (`EpochMetrics`), en batches de 256: `val_loss`, `val_accuracy` y `val_mean_recall`, el recall macro que es la métrica guía, para que EarlyStopping y ReduceLROnPlateau puedan vigilarlo. Al terminar evalúa con los pesos de la mejor época en validación **y en todo train** (en modo inferencia, sin dropout ni augmentation) para medir la brecha train-val. Escribe una fila en `experiments/experiments.csv`, el historial completo en `experiments/history/<id>.json` (con recall por clase y matriz de confusión) y el modelo en `experiments/models/<id>.keras`.
- **`plot_utils.py`:** `plot_learning_curves`, `plot_sweep`, `plot_capacity`, `plot_per_class_recall`, `plot_confusion_matrix` y `plot_bars`. Guardan en `experiments/plots/` con prefijo de fase.
- **`sweep.py`:** una función por fase; `python sweep.py phase1 phase2`. Salta los ids que ya están en el CSV, así que se puede reanudar.
- **Verificado:**
  - La caché es idéntica a leer los TIF (300 imágenes de train y 100 de test comparadas con `np.array_equal`). Construirla tarda 43 s (train y test, 224).
  - Determinismo: dos runs con la misma semilla dan métricas idénticas hasta el 4.º decimal.
  - Una época a 224 con el modelo de EXP-000 tarda 19 s (frente a 111–446 s en EXP-000). A 64 con `Dense(256)`, ~7 s: con batch 16 el límite ya no es la E/S sino la sobrecarga por paso de la GPU (~6 ms por paso de `fit`; ni `steps_per_execution` ni `tf.data` la bajan).
- **Diferencias con el generador del notebook:** el orden de barajado sale de otro RNG (`np.random.default_rng(seed)`) y dentro de cada batch las muestras se ordenan por índice (la lectura del memmap es más rápida), lo que no cambia el gradiente del batch. Por lo demás, mismos píxeles y mismas etiquetas.

### CHG-008: Se quita el ReLU tras `Flatten` y se usa `keras.Input` (2026-10-03)
Desde la fase 1, los modelos empiezan con `keras.Input(shape=(res, res, 3))`, lo que elimina el warning de Keras 3 por `input_shape`, y no llevan el `Activation('relu')` tras `Flatten`. Resuelve ISS-001. **Verificado con un run** (`CHG-008` en `experiments.csv`, base EXP-002): métricas idénticas a EXP-002 (val acc 0,418, mean recall 0,365, mismas épocas), como se esperaba, porque ReLU(x) = x para x ≥ 0.

---

## Incidencias (ISS)

### ISS-001: El ReLU tras `Flatten` no hace nada · resuelta (→ CHG-008)
`Activation('relu')` se aplica directamente a los píxeles, que son `uint8` en [0, 255] (verificado en 200 imágenes de train). ReLU(x) = x para x ≥ 0, así que no hay ninguna no linealidad y el modelo es lineal (EXP-000). Probablemente faltaba una `Dense` antes del ReLU. **Resolución:** IDEA-002.

### ISS-002: Los callbacks no llegan a actuar con 20 épocas · resuelta (→ EXP-003)
`EarlyStopping(patience=40)` no puede dispararse con `epochs=20`. `ReduceLROnPlateau(patience=10)` puede actuar como mucho una vez (en la época 11 o más tarde). Además, ambos vigilan `val_accuracy`, que es una señal ruidosa y escalonada. Vigilar `val_loss` es más suave. **Confirmada en EXP-000:** la lr fue 1e-3 las 20 épocas. La mejor val acc llegó en la época 17, así que ReduceLROnPlateau no tuvo margen para actuar, y EarlyStopping no podía. **Resolución:** IDEA-004.

### ISS-003: No hay GPU · resuelta (→ CHG-006: WSL2 + TF 2.21 con CUDA)
`tf.config.list_physical_devices('GPU')` devuelve `[]`. TensorFlow 2.10 fue la última versión con soporte de GPU en Windows nativo [R13], y el kernel usa TF 2.20, así que todo se entrena en CPU aunque CUDA 12.5 esté instalado. Esto condiciona el coste de cada experimento. **Opciones:** Google Colab (el notebook viene de Colab según sus metadatos), WSL2 con `tensorflow[and-cuda]`, o reducir el coste por época (IDEA-006). **Resolución:** Keras 3 sobre PyTorch con CUDA (CHG-003, DEC-004). El `torch 2.3.1+cu118` del Python global sí veía la GPU, pero falla al importar con su `numpy 2.2.6` ("A module that was compiled using NumPy 1.x cannot be run in NumPy 2.2.6"), por eso se usa un entorno aparte. **Reabierta** porque CHG-004 restaura la celda 1 original, que ya no activa el backend torch. Para volver a la GPU hay que fijar `os.environ['KERAS_BACKEND'] = 'torch'` antes del primer `import keras` y usar el kernel del `.venv`.

### ISS-004: Comentario engañoso sobre "resampling" · abierta (menor)
En la celda que construye `anns` aparece `# Resampling strategy to reduce training time`, pero no hay ningún remuestreo implementado. Hay que eliminarlo o implementar lo que promete (relacionado con IDEA-005).

### ISS-005: La carga de imágenes domina el tiempo de entrenamiento · resuelta (→ CHG-007)
El generador abre con rasterio cada TIF de 224×224 en cada batch y en cada época, en el mismo hilo que el entrenamiento. Medido (CHG-003): 63 ms por batch de 16 con la caché de disco caliente y 232 ms en frío, frente a 13,8 ms del paso de entrenamiento en GPU. Por la ley de Amdahl [R16], acelerar solo el cálculo da como mucho (63 + 26,4) / (63 + 13,8) ≈ ×1,16 por paso con la caché caliente, aunque la GPU haga el cálculo ×1,9 más rápido. En la prueba de CHG-003, con lecturas en frío, el paso completo fue de unos 240 ms. Mientras no se resuelva, la GPU apenas acorta las épocas de esta ffNN. Con modelos más pesados (un MLP grande o una CNN) el cálculo pesará más y la GPU se notará más, pero la carga seguirá costando unos 63 ms por paso. **Medido en EXP-000** (CPU, TF 2.20, 1,5 GB de RAM libre): de 105 a 422 ms por paso según la época, y 70 min en total. Las épocas varían mucho (111–446 s) sin ningún cambio en el cálculo, lo que delata que el tiempo lo marca la E/S y la caché de disco. **Resolución:** IDEA-009.

### ISS-006: Archivos `:Zone.Identifier` en la copia del dataset · resuelta (→ CHG-007)
Al copiar el dataset desde Windows a WSL, cada TIF llegó acompañado de un `<nombre>.tif:Zone.Identifier` (el flujo alternativo "Mark of the Web" de NTFS convertido en archivo): 21 112 archivos de más. La celda de test del notebook recorre `xview_test/` con `os.walk` sin filtrar la extensión, así que habría contado 4 730 imágenes de test y fallado al abrir los `Zone.Identifier` con rasterio. **Resolución:** el usuario pidió borrarlos (`find xview_recognition -name '*:Zone.Identifier' -delete`; quedan 18 746 + 2 365 TIF y el JSON) y, además, `xview_data.list_test_files()` filtra por `*.tif`. Pendiente aplicar el mismo filtro en la celda de test del notebook (fase 9).

---

## Ideas pendientes (backlog)

Prioridad = impacto esperado / coste. Al probar una idea se crea su EXP y la idea pasa a *probada → EXP-###*.

### IDEA-001: Normalizar la entrada · probada → EXP-002, EXP-004
- **Qué:** escalar los píxeles a [0, 1], o estandarizar por canal con la media y la desviación típica de train.
- **Por qué:** (1) si todas las entradas son positivas, los gradientes de los pesos que llegan a una neurona tienen todos el mismo signo y el descenso avanza en zigzag; centrar en cero mejora el condicionamiento [R1 §4.3]. Adam reescala cada parámetro (precondicionador diagonal), lo que compensa la escala pero no la correlación que introduce una media no nula. (2) Con `glorot_uniform` y 150 528 entradas, la desviación típica inicial de los logits es ≈ 110 con píxeles crudos frente a ≈ 1,4 con entradas estandarizadas (estimación con las estadísticas reales: media ≈ 62, desviación ≈ 47,5). Con logits ≈ 10² la softmax está saturada desde el principio.
- **Éxito:** la pérdida inicial ≈ ln 13 ≈ 2,56 y una convergencia más rápida que en EXP-000.
- **Evidencia de EXP-000:** pérdida de ~3 000 durante las 20 épocas, val acc oscilando ±5 pp y dos clases frecuentes que nunca se predicen. Es lo esperable con logits saturados (análisis en la conclusión de EXP-000). Es el cambio de mayor impacto esperado. Además de lo anterior, comprobar que la val acc deja de oscilar (σ entre épocas ≪ 0,048).

### IDEA-002: Añadir capas ocultas reales (MLP) e inicialización He · prioridad alta
- **Qué:** `Flatten → Dense(n) → ReLU → … → Dense(13) → softmax`, con `kernel_initializer='he_normal'`.
- **Por qué:** sin capa oculta el modelo es lineal. Un MLP con una sola capa oculta ya es un aproximador universal [R3; R8 §6.4.1]. Para ReLU, la inicialización He conserva la varianza de las activaciones entre capas [R2]; `glorot_uniform` (la de Keras por defecto) está pensada para activaciones simétricas.
- **Cuidado:** con 150 528 entradas, una capa de 512 neuronas tiene ≈ 77 M de parámetros frente a 16 871 imágenes de train: hay alto riesgo de overfitting. En float32, entre pesos, gradientes y los tres estados de Adam con AMSGrad, ocupa ≈ 5 × 308 MB ≈ 1,5 GB, que caben en los 6 GB de la GPU (CHG-003). Combinar con IDEA-006 e IDEA-007.

### IDEA-003: Semilla fija y split estratificado · probada → EXP-001
- **Qué:** `keras.utils.set_random_seed(...)` al inicio y `train_test_split(..., stratify=<labels>)`.
- **Por qué:** sin semilla, la inicialización y el barajado cambian entre runs, y diferencias de ~1 pp pueden ser solo ruido [R9]. Con un split no estratificado, Helipad (111 imágenes en total) tiene unas 11 en validación: la desviación típica de su recall estimado es √(p(1−p)/n) ≈ 0,15 para p = 0,5, así que las métricas por clase son muy ruidosas.
- **Ojo:** estratificar cambia el conjunto de validación, así que el primer run con este cambio (EXP-001) es la **nueva base** y sus números no se comparan con los de EXP-000.
- **En GPU** (CHG-003): con el backend torch, `keras.utils.set_random_seed` también siembra `torch` (lo hace `keras/src/utils/rng_utils.py`). Aun así, algunas operaciones de CUDA no son deterministas; `torch.use_deterministic_algorithms(True)` lo fuerza a costa de velocidad [R15].

### IDEA-004: Ajustar callbacks y número de épocas · probada → EXP-003
- **Qué:** `EarlyStopping(monitor='val_loss', patience≈5–8, restore_best_weights=True)`, `ReduceLROnPlateau(monitor='val_loss', patience≈3)` y `epochs` lo bastante grande para que sea EarlyStopping quien termine.
- **Por qué:** ISS-002. La parada temprana es una forma de regularización [R8 §7.8].

### IDEA-005: Tratar el desbalance de clases · prioridad media
- **Qué:** `class_weight` inversamente proporcional a la frecuencia (o según el "effective number" [R11]), u oversampling de las clases minoritarias en el generador.
- **Por qué:** el desbalance es de 32:1 (Building 3 594 frente a Helipad 111). Sin corregirlo, la pérdida está dominada por las clases frecuentes. En CNNs, el oversampling suele ser la estrategia más robusta [R10]. La métrica guía debe ser el mean recall (balanced accuracy), no la accuracy.
- **Evidencia de EXP-000:** Helipad (minoritaria) tiene recall 0, pero también Truck, que es la 3.ª clase más frecuente. No todo es desbalance: aplicar esta idea después de IDEA-001 para no confundir los dos efectos.

### IDEA-006: Reducir la resolución de entrada para la ffNN · probada → EXP-005
- **Qué:** redimensionar a 64×64 o 32×32 antes de `Flatten`.
- **Por qué:** divide la dimensión de entrada por ≈ 12 o ≈ 49, y con ella los parámetros de la primera capa: menos varianza (mejor relación datos/parámetros) y menos cálculo por paso. Esto último importa poco ahora que hay GPU (CHG-003) y que la carga de imágenes domina el tiempo por paso (ISS-005), porque el redimensionado se hace después de leer cada imagen. **Coste:** se pierde detalle en objetos pequeños (Small car, Pylon).

### IDEA-007: Regularización (dropout, L2, BatchNorm) · prioridad media, tras IDEA-002
- **Por qué:** cuando el MLP tenga capacidad de sobreajustar (hueco entre `train_accuracy` y `val_accuracy` en `runs.csv`). Dropout [R5], penalización L2 [R8 §7.1], BatchNorm para estabilizar y acelerar el entrenamiento [R6].

### IDEA-008: Data augmentation con rotaciones de 90° y flips · prioridad media (clave en la etapa CNN)
- **Por qué:** en imágenes aéreas no hay un "arriba" canónico, así que las 8 transformaciones del grupo diedral D4 conservan la etiqueta. Es aumentar los datos codificando una invarianza conocida [R8 §7.4]. Para una ffNN, que no comparte pesos espacialmente, el beneficio será menor que en una CNN.

### IDEA-009: Leer cada imagen una sola vez (caché del dataset) · hecha → CHG-007
- **Qué:** (a) volcar las imágenes a un único `.npy` en `uint8` y leerlo con `np.load(..., mmap_mode='r')`, o cargarlo entero en RAM. Ocupa 18 746 × 150 528 B ≈ 2,8 GB. (b) Como alternativa, o además, pasar el generador a `keras.utils.PyDataset` con `workers > 1` para que la carga se solape con el cálculo.
- **Por qué:** ISS-005. La parte serie (la E/S) limita la aceleración total [R16]. Con la caché, el paso bajaría hacia los ~14 ms del cálculo en GPU: unas 1 055 × 14 ms ≈ 15 s de entrenamiento por época, más la validación.
- **Coste/riesgo:** 2,8 GB de disco para el `.npy`. El equipo tiene 15,7 GB de RAM, pero solo había unos 2,5 GB libres al medir, así que es mejor memmap que cargarlo todo. Es un cambio del pipeline, no del modelo (CHG): hay que comprobar que los batches salen idénticos (mismos píxeles, mismo orden con la misma semilla) para que los runs sigan siendo comparables.
- **Éxito:** tiempo por paso cercano a 14 ms en GPU, medido en la barra de progreso de `fit` o en `history.csv`.

### IDEA-010: Volver a probar la estandarización por canal con la lr ajustada · prioridad baja
- **Qué:** repetir EXP-004 sobre el mejor modelo de la fase 5 (lr ya ajustada).
- **Por qué:** EXP-004 empeoró porque, con Adam a lr = 1e-3, una entrada de mayor magnitud satura antes la softmax. No se probó la ventaja de centrar en cero [R1 §4.3] en igualdad de condiciones. Con capas ocultas y BatchNorm el efecto puede ser distinto.
- **Coste:** un run. **Éxito:** mean recall ≥ el de `/255` con la misma configuración.

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

### DEC-006: Se adopta el plan de mejora de la ffNN en 9 fases (2026-10-03)
El usuario aporta un plan de mejora de la ffNN en 9 fases y pide ejecutarlo junto con la recomendación previa (IDEA-009 → EXP-001 → normalizar). Decisiones para encajarlo en este proyecto:
- **Restricciones del plan:** solo capas densas (`Dense`, `Flatten`, `Dropout`, `BatchNormalization`, `Rescaling`, `Resizing`, activaciones, regularizadores), optimizadores, schedulers, class weights y augmentation con numpy. Sin convoluciones ni modelos preentrenados. **El test no se usa para decidir nada**; solo genera `prediction.json` al final.
- **Métrica guía:** mean recall (macro) en validación, porque las clases están desbalanceadas. Secundarias: accuracy, `val_loss`, recall por clase (Helipad, Pylon, Cargo plane), parámetros y tiempo por época. Se mide además la brecha con train para diagnosticar subajuste o sobreajuste (`train_acc` y `train_mean_recall`, evaluados en modo inferencia).
- **Split estratificado** (`stratify=labels`, `random_state=1`). El plan pedía no tocar el split, pero el usuario eligió la recomendación EXP-001 = IDEA-003. Por eso el baseline se vuelve a ejecutar con el split nuevo (EXP-001), y es la referencia de todo lo que sigue. EXP-000 deja de ser comparable.
- **Backend:** el plan suponía Keras sobre torch; tras CHG-006 se usa TensorFlow con GPU en WSL2. La semilla se fija con `keras.utils.set_random_seed`, que siembra `random`, numpy y TF; `torch.manual_seed` no aplica.
- **Archivos** (nombres en inglés, como pide el plan), todos dentro de `experiments/` para no dispersar las evidencias: `experiments.csv`, `history/<id>.json`, `models/<id>.keras` (no versionados, regla `*.keras`), `plots/phase<N>_*.png` y `logs/`. Código: `xview_data.py`, `experiment_utils.py`, `plot_utils.py`, `sweep.py`, `hyperparameter_sweep.ipynb` (notebook auxiliar con gráficas e interpretación por fase). `experiments/runs.csv` sigue siendo el registro de los runs del notebook entregable.
- **IDs:** una entrada EXP por barrido de un solo factor (por ejemplo, "anchura de la capa oculta"), y una fila por variante en `experiments.csv`, con id `EXP-###_<variante>`. Los cambios que no deberían alterar el resultado (por ejemplo quitar el ReLU tras `Flatten`) son CHG, pero se verifican igual con un run.
- **Fase 1 en escalera:** el plan aplica juntas todas las correcciones de base, pero la regla de la bitácora es un factor por experimento. Se encadenan (EXP-002 normalizar → CHG-008 quitar el ReLU → EXP-003 callbacks → EXP-004 estandarizar), cada paso sobre el anterior, de modo que también se ve el efecto acumulado frente a EXP-001.
- **Elección entre variantes:** la de mayor mean recall. Con una sola semilla, diferencias de ~1 pp no son concluyentes [R9]; en ese caso se prefiere la variante más simple o con menos parámetros, y se dice. La mejor época se elige y se mide sobre la misma validación, así que todas las cifras son algo optimistas [R18]; afecta igual a todas las variantes. La fase 9 repite el modelo final con 3 semillas.
- **Commits:** uno por fase (elección del usuario), sin push.
- **Notebook entregable:** se mantiene su estructura y se actualiza en la fase 9 con el modelo final (caché, `Rescaling`, predicción por lotes, filtro `*.tif`) y una tabla resumen. Los barridos viven fuera (`sweep.py` + notebook auxiliar).

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
- **[R19]** Gonzalez, Woods (2018). *Digital Image Processing*, 4.ª ed. Pearson. (cap. 4: muestreo, aliasing y filtrado paso bajo antes de submuestrear)
- **[R20]** TensorFlow. *tf.config.experimental.enable_op_determinism* (documentación de la API). https://www.tensorflow.org/api_docs/python/tf/config/experimental/enable_op_determinism

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
