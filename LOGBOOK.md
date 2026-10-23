# LOGBOOK: xview_recognition

Clasificación de 13 tipos de objeto en recortes 224×224 de imágenes aéreas de xView [R7].

Bitácora viva del notebook [xview_recognition.ipynb](xview_recognition.ipynb) (hasta la entrega 2 se llamaba `ffNN_example.ipynb`, nombre que conservan las entradas antiguas: DEC-014): todo cambio, experimento, idea, incidencia y decisión, con su justificación teórica. Es la memoria del proyecto: si no está aquí, no pasó.

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
| Etapa del curso | **Entrega 3: CNN propias frente a arquitecturas conocidas, desde cero** (plan en DEC-017). Entregas 1 y 2 cerradas (etiquetas `delivery-01`, `delivery-02`) |
| Entorno | WSL2, `.venv` con TF 2.21 en **GPU** (CHG-006); caché `uint8` del dataset y barridos con `sweep.py` (CHG-007) |
| Mejor experimento | **EXP-035_shift2**, la CNN propia final de la entrega 3: 32×32, 4 bloques 2 × [Conv3×3 → BN → LeakyReLU 0,3] (32-64-128-256) + MaxPool, GAP → `Dense(13)`; logit adjustment τ 1; rotaciones + desplazamientos ±2 px; Adam lr 2,55e-3, β₁ 0,95, batch 128. Mejor ffNN: EXP-018 (etiqueta `ffnn-full-plan`) |
| Entrega 2 (DEC-009) | Regularización; modelo `EXP-021_final` (BN + rotaciones de 90°): 0,643 ± 0,011 / 0,646 ± 0,015. **`ffNN_example.ipynb` entrena ahora este modelo** (CHG-019, run idéntico a la semilla 1). Documentos ampliados al nivel de la entrega 1 (CHG-020): paquete listo para Overleaf |
| Entrega 1 (DEC-008) | Fases 0-5; modelo `EXP-010_bs256_sqrt_60ep` (0,616 ± 0,006 / 0,578 ± 0,007), el que entrenaba `ffNN_example.ipynb` hasta CHG-019 (versión en el commit `d557b09`). Documentos ampliados con las ediciones del usuario y la fase 4 (LeakyReLU) con figura (CHG-018) |
| Mejor val accuracy / mean recall | **0,779 ± 0,007 / 0,821 ± 0,007** (EXP-035_shift2: CNN propia final, 3 semillas). InceptionV3 desde cero empata: 0,795 ± 0,002 / 0,825 ± 0,005 (EXP-039). Mejor ffNN: EXP-018, 0,635 / 0,651. Split estratificado, no comparable con EXP-000 |
| Referencia | EXP-001 (baseline con el split nuevo): 0,389 / 0,282 · clase mayoritaria: 19,2 % de accuracy · azar: 7,7 % de mean recall |
| Diagnóstico | Entrega 2: la regularización explícita tampoco ayuda con redes más anchas (EXP-024); desplazamientos y TTA no mejoran (EXP-025..EXP-027); label smoothing descartado (EXP-022) y promedio de semillas no concluyente (EXP-023, +0,6 pp); errores restantes en vehículos entre sí y Building como cajón de objetos pequeños. Ruido entre semillas σ ≈ 1–2 pp (EXP-009). Las mejoras grandes: resolución 16×16 (fase 2), desbalance (fase 7) y D4 (fase 8). La regularización clásica no ayuda (EXP-013). Límite: el sesgo del MLP sobre píxeles; los vehículos se confunden entre sí y Truck se queda en recall 0,28 |
| Próximo paso | Entrega 3: decidir qué hacer con VGG16 tras el colapso de la semilla 3 (ISS-014: relanzar, plan B con BN o 2 semillas). Después, MixUp, comparación final (D3-8), notebook y documentos. IDEA-010 queda para la entrega 5 |

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
| EXP-006 | 2026-10-03 | 3 | Profundidad y anchura de las capas ocultas (16×16) | EXP-005_r16 | 0,604 | 0,572 | `EXP-006_h256-256-256-256` | ejecutado |
| EXP-007 | 2026-10-03 | 4 | Activación de las capas ocultas | EXP-006_h256-256-256-256 | 0,635 | 0,591 | `EXP-007_leaky0.3` | ejecutado |
| EXP-008 | 2026-10-03 | 5.1 | Learning rate (Adam y SGD con momentum), runs de ≤ 20 épocas | EXP-007_leaky0.3 | 0,609 | 0,582 | `EXP-008_sgdm_lr1e-2` | ejecutado |
| EXP-009 | 2026-10-03 | 5.2 | Momentum, β de Adam, RMSprop (≤ 20 épocas); ruido entre 3 semillas | EXP-008 | 0,608 ± 0,019 | 0,571 ± 0,009 | `EXP-009_adam_b1_0.95*` | ejecutado |
| EXP-010 | 2026-10-03 | 5.3 | Tamaño de batch con lr reescalada; batch 256 con 60 épocas, 3 semillas | EXP-009_adam_b1_0.95 | 0,616 ± 0,006 | 0,578 ± 0,007 | `EXP-010_bs256_sqrt_60ep*` | ejecutado |
| EXP-011 | 2026-10-03 | 5.4 | Schedule de la lr (3 semillas en los finalistas) | EXP-010_bs256_sqrt_60ep | — | sin mejora | `EXP-011_*` | ejecutado |
| EXP-012 | 2026-10-03 | 5.5 | Clipnorm, AMSGrad, AdamW (3 semillas) | EXP-010_bs256_sqrt_60ep | — | sin mejora | `EXP-012_*` | ejecutado |
| EXP-013 | 2026-10-03 | 6 | Regularización: dropout, L2, BatchNorm (3 semillas); gana BN con lr ×3 | EXP-010_bs256_sqrt_60ep | 0,612 ± 0,005 | 0,590 ± 0,009 | `EXP-013_bn_before_lr3x*` | ejecutado |
| EXP-014 | 2026-10-03 | 7 | Desbalance: class weights y sobremuestreo (3 semillas) | EXP-013_bn_before_lr3x | 0,614 ± 0,011 | 0,622 ± 0,005 | `EXP-014_cw_sqrt*` | ejecutado |
| EXP-015 | 2026-10-03 | 8 | Data augmentation D4 y fotométrica (3 semillas) | EXP-014_cw_sqrt | 0,625 ± 0,003 | 0,656 ± 0,006 | `EXP-015_d4*` | ejecutado |
| EXP-016 | 2026-10-03 | 9 | Ajuste fino y revisión (lr, dropout, anchura, resolución 32, batch 16) | EXP-015_d4 | — | sin mejora | `EXP-016_*` | ejecutado |
| EXP-017 | 2026-10-03 | 9 | Ablación del modelo combinado; ReLU empata con LeakyReLU | EXP-015_d4 | 0,638 ± 0,002 | 0,652 ± 0,004 | `EXP-017_relu*` | ejecutado |
| **EXP-018** | 2026-10-03 | 9 | **Modelo final** (ReLU, entrenamiento más largo) | EXP-017_relu | **0,635 ± 0,003** | **0,651 ± 0,009** | `EXP-018_final*`, `EXP-018_20261003-142325` | ejecutado |
| EXP-019 | 2026-10-03 | E2 | Augmentation sobre BN sin pesos de clase (3 semillas) | EXP-013_bn_before_lr3x | 0,632 ± 0,005 | 0,617 ± 0,015 | `EXP-019_rot90*` | ejecutado |
| EXP-020 | 2026-10-03 | E2 | Dropout, L2 y ablación de BN sobre BN + rotaciones (3 semillas) | EXP-019_rot90 | — | sin mejora | `EXP-020_*` | ejecutado |
| **EXP-021** | 2026-10-03 | E2 | **Modelo de la entrega 2**: BN + rotaciones, entrenamiento más largo | EXP-019_rot90 | **0,643 ± 0,011** | **0,646 ± 0,015** | `EXP-021_final*` | ejecutado |
| EXP-022 | 2026-10-15 | E2 | Label smoothing ε ∈ {0,05; 0,1; 0,2} (3 semillas) | EXP-021_final | 0,641 ± 0,003 | 0,617 ± 0,006 | `EXP-022_ls0.1*` | ejecutado (descartado) |
| EXP-023 | 2026-10-15 | E2 | Promedio de las probabilidades de las 3 semillas (sin entrenar) | EXP-021_final | 0,651 | 0,652 | `ensembles/EXP-023_ens_EXP-021` | ejecutado (no concluyente) |
| EXP-024 | 2026-10-16 | E2 | Capacidad × regularización: anchura {256, 512, 1024} × {nada, Dropout 0,2/0,5, L2 1e-4} (DEC-013) | EXP-021_final | 0,643 ± 0,006 | 0,645 ± 0,003 (mejor celda: 512 + L2) | `EXP-024_*` | ejecutado (se mantiene 4 × 256) |
| EXP-025 | 2026-10-16 | E2 | Desplazamientos aleatorios de ≤ 1 / ≤ 2 px | EXP-021_final | 0,600 ± 0,001 | 0,585 ± 0,021 | `EXP-025_shift1*` | ejecutado (descartado) |
| EXP-026 | 2026-10-16 | E2 | TTA con las 4 rotaciones (sin entrenar), por semilla | EXP-021_final | 0,647 ± 0,007 | 0,643 ± 0,010 | `ensembles/EXP-026_tta_*` | ejecutado (descartado) |
| EXP-027 | 2026-10-16 | E2 | Medio píxel: entrada de 32 + promedio 2×2, con y sin desplazamiento de 1 px | EXP-021_final | 0,626 ± 0,006 | 0,620 ± 0,012 (control 0,631) | `EXP-027_*` | ejecutado (descartado) |
| EXP-028 | 2026-10-16 | E2 | Ablación de las paciencias de EXP-021: solo RLR 6 → 8 / solo ES 15 → 25 | EXP-019_rot90 | 0,643 ± 0,011 / 0,632 ± 0,005 | 0,646 ± 0,015 / 0,617 ± 0,015 | `EXP-028_*` | ejecutado (todo el salto es la paciencia de RLR) |
| EXP-029 | 2026-10-17 | D3-1 | CNN propia (3 bloques 32-64-128, GAP) a 16×16, batch 256 / 64 (3 semillas) | EXP-021_final | 0,746 ± 0,003 / 0,748 ± 0,006 | 0,741 ± 0,011 / **0,751 ± 0,007** | `EXP-029_cnn16_*` | ejecutado |
| EXP-030 | 2026-10-17 | D3-2 | Resolución de la CNN propia: 32 / 64 / 112 (16 = EXP-029), batch 64 | EXP-029_cnn16_bs64 | **0,769 ± 0,010** (32) | **0,770 ± 0,006** (32) | `EXP-030_r*` | ejecutado |
| EXP-031 | 2026-10-18 | D3-3 | Profundidad {3, 4, 5} bloques × resolución {32, 64, 112} (DEC-018) | EXP-030 | 0,772 ± 0,006 (32, 4 bloques) | **0,784 ± 0,010** (32, 4 bloques; empata con 64/5 y 112/5) | `EXP-031_*` | ejecutado |
| EXP-032 | 2026-10-18 | D3-4 | Optimización: lr ×0,5 / ×2; batch 32 / 128 (regla de la raíz) | EXP-031_r32_b4 | 0,775 ± 0,006 (batch 128) | 0,785 ± 0,012 (batch 128; todas empatan) | `EXP-032_*` | ejecutado |
| EXP-033 | 2026-10-18 | D3-5 | Capacidad: anchura ×0,5 / ×2, 1 / 3 convs por bloque, AveragePool / stride 2, cabeza Flatten | EXP-032_bs128 | — | sin mejora (se mantiene la base) | `EXP-033_*` | ejecutado |
| EXP-034 | 2026-10-18 | D3-6 | Desbalance: pesos raíz / inverso, sobremuestreo 0,5 / 1, logit adjustment | EXP-032_bs128 | 0,762 ± 0,015 (logit adj.) | **0,803 ± 0,007** (logit adj.) | `EXP-034_*` | ejecutado |
| EXP-035 | 2026-10-18 | D3-7 | Regularizadores: criba con 1 semilla y las 3 mejores con 3 semillas | EXP-034_logitadj1 | 0,779 ± 0,007 (desplazamientos) | **0,821 ± 0,007** (desplazamientos ±2 px) | `EXP-035_*` | ejecutado |
| EXP-036 | 2026-10-18 | D3-7 | AdamW + desplazamientos + borrado combinados | EXP-034_logitadj1 | 0,780 ± 0,013 | 0,820 ± 0,010 | `EXP-036_combo*` | ejecutado (no suma) |
| EXP-037 | 2026-10-19 | D3-7 | Receta final de la CNN propia a 112 px (5 bloques, desplazamientos ±7 px, batch 64) | EXP-035_shift2 | 0,781 ± 0,018 | 0,814 ± 0,021 (empata con 32 px) | `EXP-037_cnn112*` | ejecutado |
| EXP-038 | 2026-10-19 | D3-8 | Criba de lr de VGG16, ResNet50 e InceptionV3 desde cero a 112 px (1 semilla, ≤ 40 épocas) | EXP-037 | — | VGG16 1e-3: 0,761; ResNet50 3e-4: 0,824; InceptionV3 3e-4: 0,781 | `EXP-038_*` | ejecutado |
| EXP-039 | 2026-10-19 | D3-8 | VGG16, ResNet50 e InceptionV3 desde cero, receta completa, 3 semillas | EXP-038 | 0,795 ± 0,002 (InceptionV3) | **0,825 ± 0,005** (InceptionV3; ResNet50 0,818, VGG16 0,758 con 2 semillas: ISS-014) | `EXP-039_*` | ejecutado (empate con la CNN propia) |

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
  - **Se confirma que conviene reducir mucho la resolución, pero no por el motivo previsto.** A lr fija, el empeoramiento con la resolución es sobre todo un **problema de optimización** (subajuste: train y val bajan juntas), no de varianza. Con Adam, cada peso se mueve ≈ lr por paso [R4 §2.1], así que una preactivación se mueve ≈ lr · Σ|x_i|, que crece con el número de entradas. Pasos de ese tamaño dejan muertas las ReLU, que ya no reciben gradiente [R8 §6.3.1]. **Medido** con los modelos guardados (unidades de la capa oculta que no se activan con ninguna imagen de validación): **28 % a 16×16, 74 % a 64, 79 % a 112 y 93 % a 224** (activación media: 20,9 %, 3,1 %, 1,1 % y 0,1 %). La lr óptima cae ≈ ×3 cada vez que se dobla el lado (×4 entradas): 16 → 1e-3, 32 → 3e-4, 64 → 1e-4.
  - Comparando cada resolución con su mejor lr, **16×16 sigue ganando** (0,549 frente a 0,524 de 32×32 y 0,510 de 64×64), con el modelo más pequeño y las épocas más rápidas. 8×8 pierde detalle (0,499). La diferencia 16 frente a 32 (2,5 pp) es moderada con una sola semilla.
  - **Se elige 16×16 con lr = 1e-3** como base de las fases siguientes; 32×32 con lr = 3e-4 queda como segundo candidato, a revisar en la fase 9.
  - La primera capa oculta (IDEA-002) ya aporta: a 224, la regresión softmax de EXP-003 daba 0,382, y a 16×16 con `Dense(256)` se llega a 0,549 (dos factores a la vez; no se atribuye por separado).
  - *Añadido con CHG-017 (2026-10-04):* las unidades inactivas de los 16 runs, medidas con los modelos guardados, ya están en `history/EXP-005_*.json` (`val.inactive_units`). Confirman las cifras de arriba y muestran que **la fracción crece también con la lr**: a 16×16, 5 % (1e-4), 13 % (3e-4), 28 % (1e-3) y 59 % (3e-3); a 64×64, 17 % (3e-5), 32 % (1e-4), 54 % (3e-4) y 74 % (1e-3). Con la mejor lr de cada tamaño queda en un nivel parecido: 28 % (16, 1e-3), 25 % (32, 3e-4) y 32 % (64, 1e-4). Encaja con que lo que mata las ReLU es el tamaño del paso en la preactivación, ≈ lr · Σ|x_i| [R4 §2.1; R8 §6.3.1]. Ordenadas por lr × número de entradas, las 16 medidas crecen de forma monótona; es una observación sobre estos runs, no una ley comprobada.
  - *Corrección (ISS-008):* a 224×224 el colapso a la distribución a priori dura unas 11 épocas (pérdida de train 2,27–2,32, accuracy 0,21–0,22). Después se recupera en parte: la pérdida baja a 2,03 y la accuracy de train llega a 0,315 al final del run.
- **Siguiente paso:** fase 3, la arquitectura (EXP-006).

### EXP-006: Profundidad y anchura de la red (fase 3)
- **Fecha:** 2026-10-03 · **Etapa:** ffNN · **Base:** `EXP-005_r16` · **Estado:** ejecutado · **Idea:** IDEA-002 (probada)
- **Hipótesis:** con 768 entradas, ensanchar o añadir capas aumenta la capacidad; mientras la brecha train-val sea pequeña (≈ 6 pp en la base), más capacidad debería mejorar la validación hasta que empiece el sobreajuste.
- **Cambio:** solo las capas ocultas: 1 capa (64, 128, 256, 512, 1024), 2 capas (256-128, 512-128, 512-256, 1024-256), 3 en embudo (512-256-128, 1024-512-256) y una estrecha y profunda (256-256-256-256) para comparar con presupuestos de parámetros parecidos.
- **Fundamento:** con una capa basta en teoría [R3], pero la profundidad puede representar algunas funciones con exponencialmente menos unidades [R8 §6.4.1]. Capacidad frente a generalización [R8 §5.2].
- **Resultados** (ids `EXP-006_h<capas>`; "inactivas" = unidades de la 1.ª capa oculta muertas en validación, CHG-009):

| Capas ocultas | Parámetros | train acc | val acc | train mean recall | **val mean recall** | min recall | Inactivas |
|---|---|---|---|---|---|---|---|
| 64 | 50 061 | 0,573 | 0,526 | 0,507 | 0,468 | 0,091 | 31 % |
| 128 | 100 109 | 0,611 | 0,569 | 0,558 | 0,514 | 0,182 | 31 % |
| 256 | 200 205 | 0,661 | 0,600 | 0,609 | 0,549 | 0,182 | 28 % |
| 512 | 400 397 | 0,671 | 0,595 | 0,629 | 0,543 | 0,182 | 27 % |
| 1024 | 800 781 | 0,698 | 0,602 | 0,659 | 0,562 | 0,272 | 25 % |
| 256-128 | 231 437 | 0,717 | 0,599 | 0,674 | 0,565 | 0,244 | 23 % |
| 512-128 | 461 069 | 0,723 | 0,605 | 0,684 | 0,560 | 0,182 | 23 % |
| 512-256 | 528 397 | 0,649 | 0,590 | 0,602 | 0,548 | 0,182 | 20 % |
| 1024-256 | 1 053 197 | 0,755 | 0,610 | 0,716 | 0,561 | 0,182 | 25 % |
| 512-256-128 | 559 629 | 0,789 | 0,603 | 0,762 | 0,568 | 0,312 | 25 % |
| 1024-512-256 | 1 446 925 | 0,976 | 0,596 | 0,968 | 0,572 | 0,281 | 33 % |
| **256-256-256-256** | 397 581 | 0,796 | 0,604 | 0,777 | 0,572 | 0,294 | 34 % |

- **Conclusión:**
  - **Se confirma en parte.** Más capacidad ayuda hasta ~200 k parámetros. A partir de ahí la validación se estanca en 0,56–0,57 (todas dentro de ~1 pp, no concluyente con una semilla), mientras **train sigue subiendo** (accuracy 0,66 → 0,98): la capacidad extra se gasta en memorizar. Es el paso del buen ajuste al **sobreajuste**, que tratará la fase 6.
  - Con el mismo presupuesto (~400 k), **la red estrecha y profunda gana a la ancha**: 256×4 da 0,572 y una capa de 512 da 0,543 (+2,9 pp), en línea con [R8 §6.4.1].
  - **Se elige 256-256-256-256**: empata en lo más alto con 1024-512-256 con 3,6 veces menos parámetros, mucha menos brecha (train acc 0,80 frente a 0,98) y el mejor min recall (0,294). La elección es por simplicidad, no por rendimiento.
  - Entre el 20 % y el 34 % de las ReLU de la primera capa están muertas en todas las arquitecturas: lo pone a prueba la fase 4.
- **Siguiente paso:** fase 4, las activaciones (EXP-007).

### EXP-007: Función de activación de las capas ocultas (fase 4)
- **Fecha:** 2026-10-03 · **Etapa:** ffNN · **Base:** `EXP-006_h256-256-256-256` (ReLU + He) · **Estado:** ejecutado
- **Hipótesis:** con un 34 % de ReLU muertas en la primera capa, las activaciones que dejan pasar gradiente para entradas negativas (LeakyReLU, ELU, GELU, SELU) deberían aprovechar mejor la red. tanh, con saturación en los dos lados, se espera peor.
- **Cambio:** solo la activación de las capas ocultas, cada una con su inicializador recomendado: LeakyReLU (α = 0,1 y 0,3), ELU y GELU con `he_normal`; SELU con `lecun_normal`; tanh con `glorot_uniform`.
- **Fundamento:** LeakyReLU [R21], ELU [R22], GELU [R23], SELU con `lecun_normal` (red autonormalizante) [R24], Glorot para activaciones simétricas [R25], He para la familia ReLU [R2]. ReLU muertas [R8 §6.3.1].
- **Resultados** (ids `EXP-007_<activación>`; la fila de ReLU es la base):

| Activación | Mejor época / épocas | train acc | val acc | train mean recall | **val mean recall** | min recall | Inactivas |
|---|---|---|---|---|---|---|---|
| ReLU (base) | 21 / 31 | 0,796 | 0,604 | 0,777 | 0,572 | 0,294 | 33,6 % |
| LeakyReLU α = 0,1 | 19 / 29 | 0,717 | 0,596 | 0,676 | 0,544 | 0,182 | 0,4 % |
| **LeakyReLU α = 0,3** | 30 / 40 | 0,819 | 0,635 | 0,813 | 0,591 | 0,273 | 0,0 % |
| ELU | 27 / 37 | 0,969 | 0,595 | 0,967 | 0,561 | 0,344 | 0,0 % |
| GELU | 32 / 42 | 0,999 | 0,601 | 0,999 | 0,583 | 0,344 | 5,9 % |
| SELU (`lecun_normal`) | 28 / 38 | 0,923 | 0,601 | 0,923 | 0,569 | 0,273 | 0,0 % |
| tanh (`glorot_uniform`) | 20 / 30 | 0,764 | 0,574 | 0,726 | 0,526 | 0,182 | 1,2 % |

- **Conclusión:**
  - **Se confirma en parte.** Todas las alternativas eliminan las unidades muertas (34 % con ReLU → ≈ 0 %; GELU 5,9 %), y ninguna muestra una pérdida que no baje al principio. **LeakyReLU con α = 0,3 es la mejor** en mean recall (0,591, +1,9 pp) y en accuracy (0,635, +3,1 pp).
  - Pero quitar las neuronas muertas no basta: LeakyReLU con α = 0,1 queda *por debajo* de ReLU (0,544), así que parte de estas diferencias de 1–3 pp es ruido de una sola semilla [R9]. No hay que sobreinterpretar la ventaja de α = 0,3.
  - GELU, ELU y SELU llegan a una train acc de 0,92–1,00 y su pérdida de validación **sube** desde la época ~12: memorizan más fácilmente sin generalizar mejor. tanh es la peor, como se esperaba de una activación que satura por los dos lados [R25].
  - **Se elige LeakyReLU con α = 0,3** (`he_normal`).
  - *Añadido al cerrar la fase 5:* repetido con las semillas 2 y 3, da **0,587 ± 0,004** de mean recall y 0,626 ± 0,008 de val acc.
- **Siguiente paso:** fase 5.1, la learning rate (EXP-008).

### EXP-008: Learning rate de Adam y de SGD con momentum (fase 5.1)
- **Fecha:** 2026-10-03 · **Etapa:** ffNN · **Base:** `EXP-007_leaky0.3` con **≤ 20 épocas** · **Estado:** ejecutado
- **Hipótesis:** la lr es el hiperparámetro más influyente [R26 §3.1.1]. Para Adam se espera el óptimo cerca de 1e-3, el valor que ya mostró ser bueno a 16×16 en la rejilla de EXP-005. SGD con momentum, al no normalizar por la magnitud del gradiente, necesitará lr mayores (1e-2 a 1e-1).
- **Cambio:** Adam (AMSGrad, clipnorm = 1) con lr ∈ {1e-4, 3e-4, 1e-3, 3e-3, 1e-2}; SGD con momentum 0,9 y lr ∈ {1e-3, 1e-2, 1e-1}. Rejilla logarítmica.
- **Presupuesto:** la fase 5 explora con runs cortos (máximo 20 épocas, con EarlyStopping 10 y ReduceLROnPlateau 4), como pide el plan. Por eso la referencia es `EXP-008_adam_lr1e-3` (la configuración de EXP-007 con 20 épocas), no EXP-007: con menos épocas las cifras son más bajas y favorecen a las lr altas, que convergen antes. El modelo final se reentrena más tiempo en la fase 9.
- **Fundamento:** [R26 §3.1.1] (la lr como primer hiperparámetro a ajustar, en escala logarítmica); Adam [R4]; momentum [R8 §8.3.2].
- **Resultados** (ids `EXP-008_<optimizador>_lr<lr>`):

| Variante | Mejor época / épocas | train acc | val acc | train mean recall | **val mean recall** | min recall | s/época |
|---|---|---|---|---|---|---|---|
| Adam lr 1e-4 | 20 / 20 | 0,657 | 0,581 | 0,605 | 0,527 | 0,182 | 10,5 s |
| Adam lr 3e-4 | 16 / 20 | 0,687 | 0,570 | 0,660 | 0,535 | 0,226 | 10,5 s |
| Adam lr 1e-3 | 20 / 20 | 0,665 | 0,559 | 0,652 | 0,518 | 0,222 | 10,4 s |
| Adam lr 3e-3 | 20 / 20 | 0,592 | 0,539 | 0,535 | 0,480 | 0,000 | 10,7 s |
| Adam lr 1e-2 | 17 / 20 | 0,642 | 0,578 | 0,593 | 0,520 | 0,182 | 10,6 s |
| SGD momentum 0.9 lr 1e-3 | 20 / 20 | 0,571 | 0,533 | 0,498 | 0,455 | 0,000 | 7,5 s |
| SGD momentum 0.9 lr 1e-2 | 20 / 20 | 0,773 | 0,609 | 0,747 | 0,582 | 0,364 | 7,4 s |
| SGD momentum 0.9 lr 1e-1 | 20 / 20 | 0,622 | 0,547 | 0,589 | 0,512 | 0,182 | 7,6 s |

- **Conclusión:**
  - **Adam es casi insensible a la lr entre 1e-4 y 1e-2** con este presupuesto: todo cae en 0,52–0,535 salvo 3e-3 (0,480), y esas diferencias son del orden del ruido de una semilla. No diverge ni a 1e-2, probablemente por `clipnorm = 1` y AMSGrad. El mejor punto es 3e-4 (0,535).
  - **SGD con momentum 0,9 y lr = 1e-2 es claramente el mejor** (0,582, val acc 0,609, min recall 0,364): +4,7 pp sobre el mejor Adam con el mismo presupuesto, y cada época es un 30 % más rápida (menos estado del optimizador). A 1e-3 avanza demasiado despacio (0,455) y a 1e-1 es inestable (0,512).
  - **El presupuesto de 20 épocas pesa:** la referencia (Adam 1e-3) da 0,518 frente a 0,591 en EXP-007, cuya mejor época fue la 30. Varias variantes tienen su mejor época en la 20.ª, así que este barrido mide también la velocidad de convergencia. Al cerrar la fase 5 se comprueba la configuración ganadora con el presupuesto completo frente a EXP-007.
  - Se siguen ambos: **SGD con lr = 1e-2** para el barrido de momentum, y **Adam con lr = 3e-4** para sus β.
- **Siguiente paso:** fase 5.2, momentum y comparación de optimizadores (EXP-009).

### EXP-009: Momentum y comparación de optimizadores (fase 5.2)
- **Fecha:** 2026-10-03 · **Etapa:** ffNN · **Base:** `EXP-008_sgdm_lr1e-2` (SGD) y `EXP-008_adam_lr3e-4` (Adam), ≤ 20 épocas · **Estado:** ejecutado
- **Hipótesis:** momentum alto (0,9–0,95) acelera SGD al promediar gradientes en direcciones consistentes; 0,99 puede volverse inestable a lr = 1e-2 (la lr efectiva es lr/(1 − m) = 1). Nesterov suele ayudar un poco. En Adam, β₁ hace el papel del momentum; β₂ = 0,99 adapta antes la escala.
- **Cambio:** SGD (lr = 1e-2) con momentum ∈ {0; 0,5; 0,9; 0,95; 0,99}, con y sin Nesterov; Adam (lr = 3e-4) con β₁ ∈ {0,8; 0,95} y β₂ = 0,99; RMSprop con lr ∈ {1e-4, 3e-4, 1e-3, 3e-3}. Cada optimizador se compara con su mejor lr.
- **Fundamento:** momentum y Nesterov [R8 §8.3.2–8.3.3; R27]; Adam [R4]; RMSprop [R8 §8.5.2].
- **Resultados** (ids `EXP-009_*`; SGD con momentum 0,9 sin Nesterov es `EXP-008_sgdm_lr1e-2`: 0,582):

| Variante | Mejor época / épocas | train acc | val acc | **val mean recall** | min recall |
|---|---|---|---|---|---|
| SGD lr 1e-2 momentum 0.0 | 14 / 20 | 0,542 | 0,519 | 0,459 | 0,000 |
| SGD lr 1e-2 momentum 0.5 | 20 / 20 | 0,642 | 0,576 | 0,524 | 0,182 |
| SGD lr 1e-2 momentum 0.95 | 20 / 20 | 0,614 | 0,537 | 0,508 | 0,182 |
| SGD lr 1e-2 momentum 0.99 | 20 / 20 | 0,572 | 0,520 | 0,451 | 0,182 |
| SGD lr 1e-2 momentum 0.5 Nesterov | 20 / 20 | 0,645 | 0,573 | 0,526 | 0,182 |
| SGD lr 1e-2 momentum 0.9 Nesterov | 18 / 20 | 0,695 | 0,584 | 0,559 | 0,249 |
| SGD lr 1e-2 momentum 0.95 Nesterov | 14 / 20 | 0,681 | 0,601 | 0,572 | 0,182 |
| SGD lr 1e-2 momentum 0.99 Nesterov | 20 / 20 | 0,611 | 0,553 | 0,506 | 0,182 |
| Adam beta_1 0.8 | 17 / 20 | 0,703 | 0,595 | 0,553 | 0,213 |
| Adam beta_1 0.95 | 20 / 20 | 0,761 | 0,614 | 0,569 | 0,273 |
| Adam beta_2 0.99 | 20 / 20 | 0,747 | 0,603 | 0,558 | 0,273 |
| RMSprop lr 1e-4 | 20 / 20 | 0,636 | 0,554 | 0,528 | 0,182 |
| RMSprop lr 3e-4 | 14 / 20 | 0,697 | 0,601 | 0,562 | 0,182 |
| RMSprop lr 1e-3 | 18 / 20 | 0,629 | 0,551 | 0,516 | 0,091 |
| RMSprop lr 3e-3 | 20 / 20 | 0,647 | 0,562 | 0,536 | 0,182 |
| SGD lr 1e-2 momentum 0.9, seed 2 | 20 / 20 | 0,718 | 0,621 | 0,568 | 0,182 |
| Adam beta_1 0.95, seed 2 | 20 / 20 | 0,758 | 0,625 | 0,582 | 0,273 |
| SGD lr 1e-2 momentum 0.9, seed 3 | 17 / 20 | 0,682 | 0,593 | 0,545 | 0,273 |
| Adam beta_1 0.95, seed 3 | 16 / 20 | 0,687 | 0,587 | 0,564 | 0,226 |

  **Ruido entre semillas** (los dos finalistas, semillas 1, 2 y 3): SGD con momentum 0,9 → 0,582 / 0,568 / 0,545 = **0,565 ± 0,018**; Adam con β₁ = 0,95 → 0,569 / 0,582 / 0,564 = **0,571 ± 0,009** (val acc 0,608 ± 0,014 y 0,608 ± 0,019).
- **Conclusión:**
  - **El momentum importa:** SGD sin momentum se queda en 0,459 y con 0,9 llega a 0,582. A partir de 0,95 empeora (0,99: 0,451), porque la lr efectiva lr/(1 − m) se vuelve demasiado grande [R27]. Nesterov no da una ventaja consistente (mejor con 0,95, peor con 0,9).
  - En Adam, β₁ = 0,95 (0,569) y β₂ = 0,99 (0,558) mejoran la referencia de β = (0,9; 0,999) (0,535), pero β₁ = 0,8 también (0,553): los resultados no son monótonos.
  - RMSprop rinde parecido a Adam en su mejor lr (3e-4: 0,562).
  - **La medida de ruido cambia la lectura de todo el barrido:** con σ ≈ 1–2 pp entre semillas, las diferencias de menos de ~3 pp con una sola semilla no son concluyentes, tanto aquí como en las fases 3–4. Los dos finalistas **empatan** (0,565 ± 0,018 frente a 0,571 ± 0,009).
  - **Se elige Adam con lr = 3e-4 y β₁ = 0,95**: tiene la mitad de varianza entre semillas, en la 5.1 fue casi insensible a la lr (importante para reescalarla en la 5.3), y los ajustes de la 5.5 son de Adam. A cambio, SGD es un 30 % más rápido por época. Elección por robustez, no por rendimiento.
- **Siguiente paso:** fase 5.3, el tamaño de batch (EXP-010).

### EXP-010: Tamaño de batch (fase 5.3)
- **Fecha:** 2026-10-03 · **Etapa:** ffNN · **Base:** `EXP-009_adam_b1_0.95` (Adam lr = 3e-4, β₁ = 0,95, batch 16), ≤ 20 épocas · **Estado:** ejecutado
- **Hipótesis:** batches mayores dan gradientes menos ruidosos y épocas más rápidas en GPU (menos sobrecarga por paso, CHG-007), pero menos actualizaciones por época; con un presupuesto fijo de 20 épocas pueden quedarse cortos. El ruido de los batches pequeños puede además actuar como regularizador.
- **Cambio:** batch ∈ {32, 64, 128, 256} con lr = 3e-4 · √(batch/16) (regla de la raíz), y con lr = 3e-4 · batch/16 (regla lineal) para 64 y 256.
- **Fundamento:** la regla lineal mantiene la escala del paso de SGD por muestra [R28]; para optimizadores adaptativos se suele proponer la raíz cuadrada (heurística, sin un resultado tan establecido; ver [R29]). Que el batch grande generaliza peor es una observación empírica discutida [R30].
- **Resultados** (ids `EXP-010_bs<batch>_<regla>`):

| Variante | Mejor época / épocas | val acc | **val mean recall** | min recall | s/época | Hasta val acc ≥ 0,58 |
|---|---|---|---|---|---|---|
| batch 16, lr 3e-4 (= `EXP-009_adam_b1_0.95`) | 20 / 20 | 0,614 | 0,569 | 0,273 | 10,6 s | 159 s |
| batch 32, lr 0.00042 (sqrt rule) | 20 / 20 | 0,578 | 0,552 | 0,273 | 4,45 s | 68 s |
| batch 64, lr 0.0006 (sqrt rule) | 20 / 20 | 0,614 | 0,571 | 0,253 | 2,52 s | 34 s |
| batch 128, lr 0.00085 (sqrt rule) | 19 / 20 | 0,611 | 0,566 | 0,273 | 1,44 s | 26 s |
| batch 256, lr 0.0012 (sqrt rule) | 20 / 20 | 0,609 | 0,574 | 0,240 | 0,68 s | 14 s |
| batch 64, lr 0.0012 (linear rule) | 17 / 20 | 0,582 | 0,563 | 0,113 | 2,8 s | 45 s |
| batch 256, lr 0.0048 (linear rule) | 20 / 20 | 0,591 | 0,565 | 0,258 | 0,66 s | 12 s |

  Con el presupuesto completo (60 épocas, EarlyStopping 10), batch 256 da lo mismo (`EXP-010_bs256_sqrt_60ep`: mejor época 20, parada en la 30) y, con 3 semillas, **0,578 ± 0,007** de mean recall y 0,616 ± 0,006 de val acc.
- **Conclusión:**
  - **Todas las variantes caen dentro del ruido** (0,552–0,574 frente a σ ≈ 1–2 pp, EXP-009), con cualquiera de las dos reglas. No se observa la peor generalización de los batches grandes [R30] en este rango.
  - **El coste cae drásticamente:** 10,6 s por época con batch 16 frente a 0,7 s con 256 (×15; con batch 16 manda la sobrecarga por paso de la GPU, CHG-007), y val acc 0,58 se alcanza en 14 s en lugar de 159 s.
  - **Se elige batch 256 con lr = 1,2e-3** (regla de la raíz). Como un run baja de 1 min, se abandona el presupuesto corto de exploración: **vuelta a 60 épocas con EarlyStopping 10, y 3 semillas en cada decisión** a partir de la 5.4.
- **Siguiente paso:** fase 5.4, el schedule de la lr (EXP-011).

### EXP-011: Schedule de la learning rate (fase 5.4)
- **Fecha:** 2026-10-03 · **Etapa:** ffNN · **Base:** `EXP-010_bs256_sqrt_60ep` (ReduceLROnPlateau, lr = 1,2e-3, 60 épocas) · **Estado:** ejecutado
- **Hipótesis:** un decaimiento suave (coseno) o un warmup que permita una lr inicial mayor pueden encontrar mejores mínimos que reducir la lr a saltos.
- **Cambio:** coseno sobre 60 épocas; decaimiento exponencial (×0,9 por época); warmup lineal de 3 épocas + coseno con lr ×2; warmup de 3 épocas + plateau con lr ×2. Para separar factores, también plateau con lr ×2 sin warmup. Los finalistas, con 3 semillas.
- **Fundamento:** decaimiento de la lr en SGD [R8 §8.3.1]; coseno [R31]; warmup para lr altas o batches grandes [R28].
- **Resultados:**

| Variante | Semillas | val mean recall | val acc |
|---|---|---|---|
| ReduceLROnPlateau, lr 1,2e-3 (base, `EXP-010_bs256_sqrt_60ep`) | 3 | 0,578 ± 0,007 | 0,616 ± 0,006 |
| Coseno | 1 | 0,570 | 0,594 |
| Exponencial 0,9/época | 1 | 0,554 | 0,589 |
| Warmup 3 + coseno, lr ×2 | 1 | 0,581 | 0,604 |
| Plateau, lr ×2 (sin warmup) | 1 | 0,590 | 0,615 |
| Warmup 3 + plateau, lr ×2 | 3 | 0,584 ± 0,028 | 0,613 ± 0,018 |

- **Conclusión:** **ningún schedule supera a la base más allá del ruido.** Con una semilla, warmup + plateau con lr ×2 parecía el mejor (0,594), pero sin warmup da lo mismo (0,590), y con 3 semillas empata con la base con cuatro veces más varianza. El coseno y el exponencial no mejoran (y con EarlyStopping el coseno no llega a completar su ciclo). **Se mantiene ReduceLROnPlateau con lr = 1,2e-3 y sin warmup**, la opción más simple y estable.
- **Siguiente paso:** fase 5.5 (EXP-012).

### EXP-012: Clipnorm, AMSGrad y weight decay desacoplado (fase 5.5)
- **Fecha:** 2026-10-03 · **Etapa:** ffNN · **Base:** `EXP-010_bs256_sqrt_60ep` · **Estado:** ejecutado
- **Hipótesis:** `clipnorm = 1` y AMSGrad venían del enunciado sin justificar; puede que no aporten. El weight decay desacoplado de AdamW regulariza de forma distinta a la L2 dentro de Adam.
- **Cambio:** sin clipnorm; sin AMSGrad; AdamW con weight decay ∈ {1e-4, 1e-3, 1e-2}. 3 semillas cada uno.
- **Fundamento:** AMSGrad [R4]; AdamW [R32].
- **Resultados:**

| Variante | Semillas | val mean recall | val acc |
|---|---|---|---|
| Base (Adam, AMSGrad, clipnorm 1) | 3 | 0,578 ± 0,007 | 0,616 ± 0,006 |
| Sin clipnorm | 3 | 0,565 ± 0,032 | 0,605 ± 0,017 |
| Sin AMSGrad | 3 | 0,572 ± 0,006 | 0,608 ± 0,003 |
| AdamW, weight decay 1e-4 | 3 | 0,565 ± 0,017 | 0,606 ± 0,011 |
| AdamW, weight decay 1e-3 | 3 | 0,579 ± 0,023 | 0,611 ± 0,010 |
| AdamW, weight decay 1e-2 | 3 | 0,566 ± 0,010 | 0,606 ± 0,008 |

- **Conclusión:** **nada cambia más allá del ruido.** Quitar clipnorm aumenta la varianza entre semillas (σ = 0,032), lo que sugiere que estabiliza algo el entrenamiento, aunque no mejora la media. Se mantiene la base.
- **Cierre de la fase 5, comprobación con 3 semillas frente a EXP-007:** EXP-007 (Adam, lr = 1e-3, β₁ = 0,9, batch 16) da **0,587 ± 0,004** (val acc 0,626 ± 0,008), frente a **0,578 ± 0,007** de la base de la fase 5. La fase 5 cuesta **~1 pp** (algo más de una σ: probablemente real, pero pequeño) a cambio de épocas 15 veces más rápidas. Hipótesis: el ruido del batch pequeño regulariza. Se sigue con batch 256 en las fases 6–8 (3 semillas por variante serían inviables con batch 16), y el modelo final se vuelve a probar con batch 16 en la fase 9 (IDEA-011).
- **Siguiente paso:** fase 6, la regularización (EXP-013).

### EXP-013: Regularización (fase 6)
- **Fecha:** 2026-10-03 · **Etapa:** ffNN · **Base:** `EXP-010_bs256_sqrt_60ep` (3 semillas: 0,578 ± 0,007) · **Estado:** ejecutado · **Idea:** IDEA-007
- **Hipótesis:** la brecha train-val de la base (train acc ≈ 0,76 frente a 0,62 en val) indica sobreajuste; regularizar debería cerrarla y, con suerte, subir la validación.
- **Cambio:** una técnica cada vez: Dropout ∈ {0,1; 0,2; 0,3; 0,5} tras cada capa oculta; L2 ∈ {1e-5, 1e-4, 1e-3} en todas las `Dense`; BatchNormalization antes de la activación (sin sesgo en la `Dense`) o después. 3 semillas cada una.
- **Fundamento:** Dropout [R5]; penalización L2 [R8 §7.1]; BatchNorm [R6].
- **Cambio añadido durante la fase:** el tope de épocas sube de 60 a 150 en toda la fase, porque con regularización el entrenamiento es más lento (en un primer intento, dropout 0,2 tuvo su mejor época en la 60 de 60). La base no cambia: sus 3 semillas pararon por EarlyStopping en las épocas 30, 41 y 42, y los runs son deterministas. Ese primer intento (4 runs) se descartó.
- **Resultados** (3 semillas por variante, media ± desviación típica):

| Variante | val mean recall | val acc | min recall | train acc | Épocas |
|---|---|---|---|---|---|
| Base (sin regularización) | 0,578 ± 0,007 | 0,616 ± 0,006 | 0,270 | 0,756 | 38 |
| Dropout 0,1 | 0,555 ± 0,016 | 0,605 ± 0,012 | 0,206 | 0,672 | 41 |
| Dropout 0,2 | 0,559 ± 0,011 | 0,605 ± 0,011 | 0,182 | 0,669 | 57 |
| Dropout 0,3 | 0,530 ± 0,016 | 0,579 ± 0,013 | 0,155 | 0,615 | 51 |
| Dropout 0,5 | 0,394 ± 0,014 | 0,479 ± 0,010 | 0,000 | 0,489 | 28 |
| L2 1e-5 | 0,575 ± 0,021 | 0,614 ± 0,012 | 0,299 | 0,775 | 45 |
| L2 1e-4 | 0,577 ± 0,007 | 0,611 ± 0,006 | 0,278 | 0,769 | 44 |
| L2 1e-3 | 0,469 ± 0,012 | 0,550 ± 0,008 | 0,030 | 0,581 | 34 |
| BN antes de la activación | 0,578 ± 0,014 | 0,608 ± 0,002 | 0,281 | 0,836 | 32 |
| BN después de la activación | 0,578 ± 0,006 | 0,613 ± 0,004 | 0,277 | 0,821 | 35 |
| **BN antes, lr ×3 (3,6e-3)** | 0,590 ± 0,009 | 0,612 ± 0,005 | 0,323 | 0,797 | 32 |
| BN después, lr ×3 | 0,569 ± 0,011 | 0,602 ± 0,006 | 0,244 | 0,700 | 29 |

- **Conclusión:**
  - **Se refuta para dropout y L2:** al aumentar la regularización, **train y validación bajan juntas** (dropout 0,5: train acc 0,489, val 0,479). No hay una varianza que recortar: la regularización solo empuja hacia el subajuste. La brecha de la base (train 0,76 frente a val 0,62) no se cierra subiendo la validación, sino bajando train. El límite parece ser el sesgo del modelo: un MLP sobre píxeles no tiene ninguna invarianza espacial [R8 §9.2]. Es lo que pueden aportar la augmentation (fase 8) y, en la etapa siguiente del curso, las CNN.
  - **BatchNorm** con la misma lr empata con la base (0,578 en los dos órdenes). Con **lr ×3 y BN antes de la activación** llega a **0,590 ± 0,009** (+1,2 pp, min recall 0,32 frente a 0,27): BN permite una lr mayor [R6]. Con 3 semillas por grupo es una mejora **marginal** (t ≈ 2), pero las tres semillas superan o igualan a la mejor de la base. **Se adopta BN antes de la activación con lr = 3,6e-3.**
- **Siguiente paso:** fase 7, el desbalance de clases (EXP-014).

### EXP-014: Tratamiento del desbalance de clases (fase 7)
- **Fecha:** 2026-10-03 · **Etapa:** ffNN · **Base:** `EXP-013_bn_before_lr3x` (3 semillas) · **Estado:** ejecutado · **Idea:** IDEA-005 (probada)
- **Hipótesis:** con un desbalance de 32:1, la pérdida está dominada por las clases frecuentes. Reponderar o sobremuestrear sube el recall de las minoritarias y, con él, el mean recall, a costa de algo de accuracy global.
- **Cambio:** pesos por muestra n_total / (n_clases · n_c) (inverso) o su raíz cuadrada, normalizados para que su media sobre las muestras sea 1 (misma escala de pérdida); o sobremuestreo dentro del generador con probabilidad por clase ∝ n_c^(1−α): α = 1 balanceado, α = 0,5 con raíz. 3 semillas.
- **Fundamento:** IDEA-005 [R10, R11]. Reponderar la pérdida y sobremuestrear son equivalentes en esperanza, pero el sobremuestreo cambia la varianza del gradiente (repite ejemplos).
- **Resultados** (3 semillas por variante):

| Variante | val mean recall | val acc | min recall | train acc | Épocas |
|---|---|---|---|---|---|
| Base (sin tratamiento) | 0,590 ± 0,009 | 0,612 ± 0,005 | 0,323 | 0,797 | 32 |
| `class_weight` inverso | 0,625 ± 0,010 | 0,588 ± 0,025 | 0,214 | 0,746 | 42 |
| **`class_weight` con raíz** | 0,622 ± 0,005 | 0,614 ± 0,011 | 0,252 | 0,795 | 40 |
| Sobremuestreo balanceado (α = 1) | 0,619 ± 0,002 | 0,598 ± 0,010 | 0,241 | 0,774 | 37 |
| Sobremuestreo con raíz (α = 0,5) | 0,620 ± 0,003 | 0,604 ± 0,014 | 0,249 | 0,778 | 31 |

  Recall por clase (media de 3 semillas), base → `class_weight` con raíz, de mayor a menor ganancia: Helipad 0,455 → 0,636; Pylon 0,731 → 0,839; Dump truck 0,403 → 0,454; Motorboat 0,495 → 0,545; Bus 0,397 → 0,437; Storage tank 0,626 → 0,660; Excavator 0,650 → 0,684; Cargo plane 0,740 → 0,760; Shipping container 0,678 → 0,697; Small car 0,791 → 0,794; Fishing vessel 0,615 → 0,610; Building 0,752 → 0,717; Truck 0,342 → 0,252.
- **Conclusión:**
  - **Se confirma, y con claridad:** las cuatro técnicas suben el mean recall unos **3 pp** (0,619–0,625 frente a 0,590), más de tres σ por encima de la base. Es la mejora más clara desde la fase 2. Entre ellas, las diferencias están dentro del ruido.
  - Los pesos inversos bajan la accuracy (0,588 frente a 0,612), como se esperaba; con la raíz no baja (0,614).
  - La ganancia se concentra en las clases minoritarias (Helipad 0,455 → 0,636, Pylon 0,731 → 0,839), pero **Truck baja** (0,342 → 0,252) y Building algo (0,752 → 0,717): los pesos desplazan predicciones de las clases grandes hacia las pequeñas, y el **min recall baja** (0,323 → 0,252), ahora en Truck. Truck ya era la clase más difícil desde EXP-000; se revisa en la matriz de confusión.
  - **Se elige `class_weight` con raíz:** el mismo mean recall que los pesos inversos (0,622 ± 0,005 frente a 0,625 ± 0,010) sin perder accuracy, y con la menor varianza entre semillas.
- **Siguiente paso:** fase 8, la augmentation (EXP-015).

### EXP-015: Data augmentation (fase 8)
- **Fecha:** 2026-10-03 · **Etapa:** ffNN · **Base:** `EXP-014_cw_sqrt` con paciencia 15 / 6 (`EXP-015_none_patience15`) · **Estado:** ejecutado · **Idea:** IDEA-008
- **Hipótesis:** las imágenes aéreas no tienen un "arriba" canónico, así que las 8 transformaciones del grupo diedral D4 conservan la etiqueta. Mostrarlas en train es codificar esa invarianza, que el MLP no tiene, y debería reducir la brecha train-val. Las variaciones suaves de brillo y contraste imitan cambios de iluminación y sensor.
- **Cambio:** solo en train, dentro del generador y sobre la caché: volteos horizontal y vertical aleatorios (`flip`); rotación aleatoria de 0/90/180/270° (`rot90`); ambas (`d4`, el grupo completo); brillo ±20 niveles y contraste ×[0,8; 1,2] (`color`); y `d4` + `color`. Con augmentation se necesitan más épocas: en toda la fase EarlyStopping 15 y ReduceLROnPlateau 6, también en la referencia sin augmentation, para que solo cambie la augmentation. 3 semillas.
- **Fundamento:** IDEA-008 [R8 §7.4]. Para una ffNN, que no comparte pesos entre posiciones, la invarianza se tiene que aprender a partir de los ejemplos, así que el beneficio será menor que en una CNN [R8 §9.2].
- **Resultados** (3 semillas por variante):

| Variante | val mean recall | val acc | min recall | train acc | Épocas |
|---|---|---|---|---|---|
| Sin augmentation (referencia, paciencia 15/6) | 0,615 ± 0,007 | 0,606 ± 0,011 | 0,288 | 0,837 | 44 |
| Volteos | 0,647 ± 0,003 | 0,623 ± 0,007 | 0,176 | 0,707 | 71 |
| Rotaciones de 90° | 0,654 ± 0,015 | 0,629 ± 0,009 | 0,202 | 0,684 | 59 |
| **D4 (volteos + rotaciones)** | 0,656 ± 0,006 | 0,625 ± 0,003 | 0,186 | 0,694 | 66 |
| Brillo y contraste | 0,596 ± 0,009 | 0,594 ± 0,007 | 0,279 | 0,788 | 49 |
| D4 + brillo y contraste | 0,619 ± 0,007 | 0,599 ± 0,006 | 0,136 | 0,633 | 62 |

  Recall por clase (media de 3 semillas), sin augmentation → D4: Helipad 0,576 → 0,818; Motorboat 0,539 → 0,664; Dump truck 0,406 → 0,503; Fishing vessel 0,582 → 0,676; Cargo plane 0,792 → 0,849; Bus 0,458 → 0,497; Small car 0,752 → 0,775; Storage tank 0,628 → 0,642; Building 0,706 → 0,719; Excavator 0,692 → 0,700; Shipping container 0,717 → 0,715; Pylon 0,860 → 0,785; Truck 0,288 → 0,186.
  Confusiones principales con D4 (semilla 1): Truck → Small car 0,20; Dump truck → Excavator 0,16; Truck → Dump truck 0,16; Bus → Small car 0,16; Dump truck → Truck 0,15; Storage tank → Building 0,15.
- **Conclusión:**
  - **Se confirma, y es la mayor mejora del plan:** D4 sube el mean recall de 0,615 ± 0,007 a **0,656 ± 0,006** (+4,1 pp, unas seis σ). Los volteos solos (0,647) y las rotaciones solas (0,654) dan casi lo mismo. La brecha se cierra por el lado bueno: train acc baja de 0,84 a 0,69 y val acc sube de 0,606 a 0,625. La red deja de memorizar píxeles ligados a una orientación concreta, justo la invarianza que la fase 6 apuntaba que faltaba.
  - **La augmentation fotométrica empeora** (0,596 sola, 0,619 con D4): a 16×16, el brillo y el contraste son información útil (p. ej. Storage tank frente a Building), no ruido. Sin verificar por clase.
  - La ganancia se concentra en Helipad, Motorboat, Dump truck y Fishing vessel, pero **Truck vuelve a bajar** (0,288 → 0,186) y es la clase de min recall: se confunde con Small car, Dump truck y Shipping container. Hipótesis: a 16×16 se pierde el detalle que distingue los vehículos; se prueba 32×32 en la fase 9.
  - **Se elige D4 sin augmentation fotométrica**, con paciencia 15 / 6.
  - La referencia con paciencia 15/6 (0,615 ± 0,007) es igual, dentro del ruido, a la base con paciencia 10/4 (0,622 ± 0,005): el cambio de paciencia por sí solo no influye.
- **Siguiente paso:** fase 9, combinación, ajuste fino, ablación y modelo final (EXP-016 en adelante).

### EXP-016: Ajuste fino y revisión de decisiones con el modelo combinado (fase 9)
- **Fecha:** 2026-10-03 · **Etapa:** ffNN · **Base:** `EXP-015_d4` (= combinación de todas las fases; 3 semillas: 0,656 ± 0,006) · **Estado:** ejecutado
- **Por qué:** el plan se ejecutó de forma acumulativa (cada fase parte de la ganadora de la anterior), así que la combinación ya es `EXP-015_d4` y no empeora nada. Pero la augmentation cambia el régimen (menos sobreajuste), y algunas decisiones tomadas antes pueden cambiar de signo: una red más ancha o una resolución mayor (más detalle para Truck, la clase más difícil) podrían aprovecharse ahora; algo de dropout también.
- **Cambio** (cada uno por separado, 3 semillas): lr ×0,5 y ×2 (ajuste fino); dropout 0,1; capas 512×4; resolución 32×32 (Truck, segundo candidato de EXP-005); batch 16 con lr = 3,6e-3·√(16/256) = 9e-4 (IDEA-011).
- **Fundamento:** [R26 §3.1.1] (ajustar la lr alrededor del óptimo); interacción entre regularización implícita y explícita [R8 §7.4, §7.12]; IDEA-011 [R30].
- **Resultados** (3 semillas por variante):

| Variante | val mean recall | Δ frente a `EXP-015_d4` | val acc | min recall | train acc | Épocas |
|---|---|---|---|---|---|---|
| Modelo combinado (`EXP-015_d4`) | 0,656 ± 0,006 | — | 0,625 ± 0,003 | 0,186 | 0,694 | 66 |
| lr ×0,5 (1,8e-3) | 0,656 ± 0,014 | 0,0 pp | 0,625 ± 0,010 | 0,190 | 0,693 | 68 |
| lr ×2 (7,2e-3) | 0,656 ± 0,010 | 0,0 pp | 0,623 ± 0,007 | 0,184 | 0,678 | 73 |
| Dropout 0,1 | 0,617 ± 0,019 | -3,9 pp | 0,598 ± 0,018 | 0,151 | 0,624 | 61 |
| Capas 512×4 | 0,647 ± 0,003 | -0,9 pp | 0,619 ± 0,003 | 0,175 | 0,668 | 60 |
| Resolución 32×32 | 0,634 ± 0,008 | -2,2 pp | 0,615 ± 0,008 | 0,170 | 0,651 | 53 |
| Batch 16, lr 9e-4 (15,9 s/época frente a 1,0) | 0,619 ± 0,008 | -3,7 pp | 0,592 ± 0,010 | 0,093 | 0,624 | 75 |

- **Conclusión:** **nada mejora el modelo combinado.** Con BatchNorm la lr deja de ser crítica (×0,5 y ×2 dan lo mismo). El dropout vuelve a empeorar (−3,9 pp), una red más ancha no aporta (−0,9 pp) y 32×32 tampoco (−2,2 pp; Truck no mejora: min recall 0,170). **IDEA-011 queda refutada:** con augmentation, el batch 16 es peor (−3,7 pp, min recall 0,093) y 16 veces más lento; la ventaja del batch pequeño en la fase 5 desaparece cuando ya hay otra fuente de regularización.
- **Siguiente paso:** EXP-017 (ablación).

### EXP-017: Ablación del modelo combinado (fase 9)
- **Fecha:** 2026-10-03 · **Etapa:** ffNN · **Base:** `EXP-015_d4` · **Estado:** ejecutado
- **Por qué:** cuantificar cuánto aporta cada ingrediente dentro del modelo final, porque las ganancias medidas en cada fase pueden no sumarse.
- **Cambio** (quitando uno cada vez, 3 semillas): sin D4 (`EXP-015_none_patience15`, ya ejecutado); sin `class_weight`; sin BN (y lr de vuelta a 1,2e-3, la de antes de BN); ReLU en lugar de LeakyReLU(0,3); una sola capa de 256; Adam con β₁ = 0,9.
- **Resultados** (3 semillas por variante; ordenadas por pérdida):

| Variante | val mean recall | Δ frente a `EXP-015_d4` | val acc | min recall | train acc | Épocas |
|---|---|---|---|---|---|---|
| Modelo combinado (`EXP-015_d4`) | 0,656 ± 0,006 | — | 0,625 ± 0,003 | 0,186 | 0,694 | 66 |
| Sin profundidad (una capa de 256) | 0,576 ± 0,032 | -8,0 pp | 0,554 ± 0,017 | 0,116 | 0,572 | 74 |
| Sin BatchNorm (lr 1,2e-3) | 0,603 ± 0,031 | -5,3 pp | 0,592 ± 0,016 | 0,167 | 0,636 | 62 |
| Sin `class_weight` | 0,606 ± 0,008 | -5,0 pp | 0,630 ± 0,005 | 0,247 | 0,671 | 61 |
| Sin D4 (`EXP-015_none_patience15`) | 0,615 ± 0,007 | -4,1 pp | 0,606 ± 0,011 | 0,288 | 0,837 | 44 |
| Adam β₁ = 0,9 | 0,641 ± 0,021 | -1,5 pp | 0,611 ± 0,017 | 0,163 | 0,674 | 58 |
| ReLU en lugar de LeakyReLU(0,3) | 0,652 ± 0,004 | -0,4 pp | 0,638 ± 0,002 | 0,264 | 0,790 | 59 |

- **Conclusión:**
  - Por orden de aportación dentro del modelo final: **profundidad** (−8,0 pp), **BatchNorm** (−5,3 pp), **pesos de clase** (−5,0 pp de mean recall, aunque la accuracy sube un poco sin ellos, como se espera), **augmentation D4** (−4,1 pp) y β₁ = 0,95 (−1,5 pp, ~1 σ).
  - **BatchNorm pesa mucho más aquí que en la fase 6** (+1,2 pp entonces): interactúa con la augmentation. Sin BN, la varianza entre semillas se dispara (σ = 0,031).
  - **ReLU empata con LeakyReLU(0,3)** en mean recall (0,652 ± 0,004 frente a 0,656 ± 0,006, dentro del ruido) y es mejor en accuracy (0,638 frente a 0,625) y en min recall (0,264 frente a 0,186). Con BN antes de la activación, las preactivaciones están centradas y el problema de las ReLU muertas de la fase 4 desaparece [R6]. **El modelo final usa ReLU**: es más simple y gana en las métricas secundarias.
- **Siguiente paso:** EXP-018 (modelo final).

### EXP-018: Modelo final (fase 9)
- **Fecha:** 2026-10-03 · **Etapa:** ffNN · **Base:** `EXP-017_relu` · **Estado:** ejecutado
- **Configuración:** 16×16 (caché con promedio por área) → `Rescaling(1/255)` → `Flatten` → 4 × [`Dense(256, he_normal, sin sesgo)` → `BatchNormalization` → ReLU] → `Dense(13, softmax)` (400 653 parámetros). Adam con AMSGrad, lr = 3,6e-3, β₁ = 0,95, clipnorm = 1; batch 256; `class_weight` con raíz; augmentation D4 en train; ReduceLROnPlateau (×0,1) y EarlyStopping sobre `val_mean_recall`.
- **Cambio:** entrenar más tiempo, como pide el plan: paciencias 25 / 8 (antes 15 / 6) y hasta 300 épocas. 3 semillas.
- **Resultados** (ids `EXP-018_final[_seed2|_seed3]`):
  - Semilla 1: mejor época 47 de 72 · val acc 0,637 · **mean recall 0,644** · min recall 0,294. Semilla 2: 0,636 / 0,661. Semilla 3: 0,632 / 0,648.
  - **Media de 3 semillas: mean recall 0,651 ± 0,009 · val acc 0,635 ± 0,003 · min recall 0,276.** Frente a la base EXP-001 (0,282 / 0,389): ×2,3 el mean recall y +24,6 pp de accuracy.
  - Recall por clase (media de 3 semillas): Cargo plane 0,885 · Small car 0,777 · Pylon 0,731 · Building 0,728 · Shipping container 0,711 · Helipad 0,697 · Fishing vessel 0,685 · Excavator 0,671 · Motorboat 0,660 · Storage tank 0,655 · Dump truck 0,535 · Bus 0,454 · **Truck 0,276**.
  - **Reproducción en el notebook entregable** (CHG-010): run `EXP-018_20261003-142325` en `experiments/runs.csv`, idéntico al run `EXP-018_final` del barrido (época 47 de 72, mean recall 0,6443, val acc 0,6373, min recall 0,2941). Entrenamiento de 79 s; el notebook completo tarda ~1,5 min (EXP-000 tardaba ~75 min). `model.keras` y `prediction.json` (2 365 imágenes de test) salen de este run. La `train_accuracy` de `runs.csv` (0,828) es la media durante la época con augmentation; el `train_acc` de `experiments.csv` (0,851) se evalúa en inferencia sobre todo train.
- **Conclusión:** entrenar más no cambia nada (EarlyStopping corta hacia la época 70, igual que con paciencia 15): el modelo final rinde lo mismo que `EXP-017_relu`. **Es el mejor modelo de la etapa ffNN.** El límite sigue siendo el sesgo de una red densa sobre píxeles: los vehículos (Truck, Bus, Dump truck) se confunden entre sí, y Truck se queda en 0,28. Las mejoras grandes vinieron de la resolución (fase 2), el desbalance (fase 7) y la invarianza D4 (fase 8). El siguiente salto natural es la etapa CNN, que incorpora por arquitectura la conectividad local y la equivarianza a traslaciones [R8 §9.2].
- **Siguiente paso:** etapa CNN del curso. Pendiente de la ffNN: IDEA-010 (estandarización con la lr ajustada, prioridad baja).

### EXP-019: Data augmentation como regularización, sin pesos de clase (entrega 2)
- **Fecha:** 2026-10-03 · **Etapa:** ffNN, regularización (entrega 2, DEC-009) · **Base:** `EXP-013_bn_before_lr3x` (BatchNorm antes de la activación, lr = 3,6e-3; sobre el modelo final de la entrega 1) con paciencias 15 / 6 · **Estado:** ejecutado
- **Por qué:** EXP-015 midió la augmentation sobre el modelo con pesos de clase (EXP-014). El desbalance pasa a la entrega 5, así que para la entrega 2 la augmentation se mide sobre el ganador de la regularización sin pesos de clase.
- **Hipótesis:** como en EXP-015, D4 debería reducir la brecha train-val y subir la validación; la variación fotométrica, empeorarla.
- **Cambio:** las mismas variantes que EXP-015 (sin augmentation con paciencia 15/6 como referencia, volteos, rotaciones de 90°, D4, brillo y contraste, D4 + brillo y contraste), 3 semillas.
- **Fundamento:** IDEA-008 [R8 §7.4].
- **Resultados** (3 semillas por variante; base de la fase 6 con paciencia 10/4: 0,590 ± 0,009):

| Variante | val mean recall | val acc | min recall | train acc | Épocas |
|---|---|---|---|---|---|
| Sin augmentation (referencia, paciencia 15/6) | 0,595 ± 0,010 | 0,615 ± 0,005 | 0,327 | 0,817 | 39 |
| Volteos | 0,609 ± 0,018 | 0,629 ± 0,005 | 0,213 | 0,694 | 63 |
| **Rotaciones de 90°** | 0,617 ± 0,015 | 0,632 ± 0,005 | 0,244 | 0,694 | 58 |
| D4 (volteos + rotaciones) | 0,606 ± 0,008 | 0,630 ± 0,005 | 0,247 | 0,671 | 61 |
| Brillo y contraste | 0,571 ± 0,009 | 0,601 ± 0,003 | 0,315 | 0,821 | 50 |
| D4 + brillo y contraste | 0,591 ± 0,011 | 0,615 ± 0,002 | 0,181 | 0,652 | 64 |

  Recall por clase (media de 3 semillas), sin augmentation → rotaciones: Cargo plane 0,740 → 0,839; Motorboat 0,530 → 0,626; Dump truck 0,355 → 0,449; Bus 0,412 → 0,471; Fishing vessel 0,587 → 0,638; Building 0,745 → 0,784; Small car 0,817 → 0,823; Helipad 0,485 → 0,485; Storage tank 0,599 → 0,594; Pylon 0,731 → 0,720; Shipping container 0,708 → 0,691; Excavator 0,696 → 0,654; **Truck 0,327 → 0,244**.
- **Conclusión:**
  - **Se confirma que la augmentation geométrica regulariza:** la accuracy de train baja de 0,817 a ~0,67–0,69 y la de validación sube de 0,615 a ~0,63 con muy poca dispersión (σ = 0,005). En mean recall la ganancia es de +1,1 a +2,2 pp, menor que en EXP-015 (+4,1 pp con pesos de clase): parte de aquella ganancia era interacción con el tratamiento del desbalance.
  - Volteos, rotaciones y D4 empatan dentro del ruido; la variación fotométrica vuelve a empeorar (0,571).
  - El min recall baja (0,327 → 0,244): Truck pierde recall, como en EXP-015, mientras Cargo plane, Motorboat y Dump truck ganan.
  - **Se elige rotaciones de 90°** (0,617 ± 0,015 de mean recall, 0,632 de accuracy): la mejor media en las dos métricas y un subconjunto más simple que D4; la diferencia con D4 no es concluyente.
- **Siguiente paso:** EXP-020 (penalizaciones sobre BN + rotaciones, y ablación de BN).

### EXP-020: Penalizaciones y BatchNorm sobre BN + rotaciones (entrega 2)
- **Fecha:** 2026-10-03 · **Etapa:** ffNN, regularización (entrega 2) · **Base:** `EXP-019_rot90` · **Estado:** ejecutado
- **Hipótesis:** con la augmentation cerrando parte de la brecha, una penalización explícita suave podría sumar; y BatchNorm, que en la fase 6 aportó poco (+1,2 pp), puede pesar más con augmentation (como en la ablación EXP-017).
- **Cambio:** dropout 0,1 y 0,2; L2 1e-5 y 1e-4; sin BatchNorm (con la lr de la fase 5, 1,2e-3). 3 semillas.
- **Fundamento:** Dropout [R5], L2 [R8 §7.1], BatchNorm [R6].
- **Resultados:**

| Variante | val mean recall | val acc | min recall | train acc | Épocas |
|---|---|---|---|---|---|
| BN + rotaciones (base, `EXP-019_rot90`) | 0,617 ± 0,015 | 0,632 ± 0,005 | 0,244 | 0,694 | 58 |
| + dropout 0,1 | 0,577 ± 0,010 | 0,616 ± 0,009 | 0,177 | 0,645 | 73 |
| + dropout 0,2 | 0,550 ± 0,039 | 0,593 ± 0,033 | 0,151 | 0,622 | 68 |
| + L2 1e-5 | 0,603 ± 0,006 | 0,630 ± 0,001 | 0,256 | 0,677 | 64 |
| + L2 1e-4 | 0,608 ± 0,012 | 0,630 ± 0,009 | 0,247 | 0,681 | 62 |
| Sin BatchNorm (lr 1,2e-3) | 0,569 ± 0,023 | 0,600 ± 0,017 | 0,205 | 0,658 | 64 |

- **Conclusión:** **ninguna penalización mejora.** El dropout vuelve a bajar train y validación a la vez (subajuste); L2 deja la accuracy igual (0,630) y el mean recall algo por debajo, dentro del ruido. **Sin BatchNorm se pierden 4,8 pp**: con augmentation, BatchNorm es el ingrediente de regularización/optimización que más aporta, igual que en la ablación EXP-017. Se mantiene BN + rotaciones, sin dropout ni L2.
- **Siguiente paso:** EXP-021 (el modelo de la entrega 2 entrenado más tiempo).

### EXP-021: Modelo final de la entrega 2 (BN + rotaciones, entrenamiento más largo)
- **Fecha:** 2026-10-03 · **Etapa:** ffNN, regularización (entrega 2) · **Base:** `EXP-019_rot90` · **Estado:** ejecutado
- **Configuración:** 16×16 `/255` → 4 × [`Dense(256)` → BatchNorm → LeakyReLU(0,3)] → `Dense(13)`; Adam AMSGrad, lr = 3,6e-3, β₁ = 0,95; batch 256; rotaciones aleatorias de 90° en train; sin pesos de clase.
- **Cambio:** paciencias 25 / 8 (antes 15 / 6) y hasta 300 épocas, como en EXP-018. 3 semillas.
- **Resultados:** semilla 1: época 82 de 107, mean recall 0,660, val acc 0,642; semilla 2: época 65 de 90, mean recall 0,647, val acc 0,654; semilla 3: época 55 de 80, mean recall 0,630, val acc 0,632. **Media: mean recall 0,646 ± 0,015 · val acc 0,643 ± 0,011 · min recall 0,258** · train acc 0,764.
- **Conclusión:** a diferencia de EXP-018, **entrenar más sí ayuda aquí** (+2,9 pp sobre `EXP-019_rot90`, con las mejores épocas entre la 55 y la 82): con augmentation la validación sigue subiendo más tarde. Frente al modelo de la entrega 1 (`EXP-010_bs256_sqrt_60ep`: 0,578 ± 0,007 / 0,616 ± 0,006), la regularización suma +6,8 pp de mean recall y +2,7 pp de accuracy. **Es el modelo final de la entrega 2.**
- **Siguiente paso:** documentos de la entrega 2 (`reports/delivery_02/`).
- **Corrección (2026-10-16, EXP-028):** «entrenar más» era una mala descripción del cambio. EXP-021 cambió a la vez dos factores (paciencia de EarlyStopping 15 → 25 y de ReduceLROnPlateau 6 → 8; el tope de 300 épocas nunca se alcanza). La ablación muestra que **todo el +2,9 pp es la paciencia de ReduceLROnPlateau**: la lr baja más tarde. La de EarlyStopping no cambia nada.

### EXP-022: Label smoothing sobre el modelo de la entrega 2
- **Fecha:** 2026-10-15 · **Etapa:** ffNN, regularización (entrega 2) · **Base:** `EXP-021_final` · **Estado:** ejecutado (descartado)
- **Por qué:** a petición del usuario ("no estoy obteniendo buenos resultados"). Es la propuesta 4 de `next-ideas`: una regularización de la salida que todavía no se había probado.
- **Hipótesis:** penalizar la sobreconfianza en entrenamiento reduce la brecha de 12,1 pp sin perder validación.
- **Cambio:** objetivos $(1-ε)$·one-hot + ε/K (`CategoricalCrossentropy(label_smoothing=ε)`) con ε ∈ {0,05; 0,1; 0,2}. Todo lo demás, como `EXP-021_final`. 3 semillas.
- **Fundamento:** Szegedy et al. 2016, §7 [R38]; Müller, Kornblith, Hinton 2019 [R39].
- **Resultados** (3 semillas; brecha = train acc − val acc):

| Variante | val mean recall | val acc | min recall | train acc | Brecha (pp) | Mejor época / épocas |
|---|---|---|---|---|---|---|
| Sin label smoothing (base) | 0,646 ± 0,015 | 0,643 ± 0,011 | 0,258 | 0,764 | 12,1 | 67 / 92 |
| ε = 0,05 | 0,603 ± 0,004 | 0,638 ± 0,008 | 0,255 | 0,695 | 5,7 | 42 / 67 |
| ε = 0,1 | 0,617 ± 0,006 | 0,641 ± 0,003 | 0,262 | 0,721 | 8,0 | 59 / 84 |
| ε = 0,2 | 0,593 ± 0,011 | 0,640 ± 0,005 | 0,212 | 0,699 | 5,9 | 49 / 74 |

  Recall por clase, base → ε = 0,1: **Helipad 0,697 → 0,424** (11 imágenes de validación), Storage tank 0,660 → 0,590, Excavator 0,700 → 0,650, Dump truck 0,478 → 0,430; Bus 0,443 → 0,482, Small car 0,812 → 0,846.
- **Conclusión:**
  - **Se refuta.** La brecha baja, pero porque baja la accuracy de train. La de validación no cambia y el mean recall baja entre 2,9 y 5,3 pp con las tres intensidades, con poca dispersión entre semillas: es el mismo patrón que Dropout y L2 (EXP-013, EXP-020).
  - La pérdida se concentra en pocas clases. Solo Helipad aporta unos 2 pp de los 2,9 de ε = 0,1, y en esa clase la diferencia son unas 3 imágenes.
  - El entrenamiento para antes (mejor época 42-59 frente a 67).
  - Una posible explicación, no verificada: repartir ε entre todas las clases resta margen a las que tienen pocos ejemplos.
  - Se mantiene `EXP-021_final`.

### EXP-023: Promedio de las 3 semillas del modelo de la entrega 2
- **Fecha:** 2026-10-15 · **Etapa:** ffNN, regularización (entrega 2) · **Base:** `EXP-021_final` (3 semillas por separado) · **Estado:** ejecutado (no concluyente)
- **Por qué:** a petición del usuario; es la propuesta 1 de `next-ideas`. Las semillas varían mucho: mean recall 0,630-0,660 y train acc 0,73-0,81.
- **Hipótesis:** promediar los modelos reduce la varianza entre runs [R8 §7.11] y debería superar en más de 1 pp a la media de los modelos por separado.
- **Cambio:** se promedian las probabilidades softmax de los 3 modelos guardados (`experiments/models/EXP-021_final*.keras`), sin entrenar nada (`tools/ensemble_eval.py`).
- **Resultados** (`experiments/ensembles/EXP-023_ens_EXP-021.json`):
  - **Promedio:** val acc **0,651**, mean recall **0,652**, min recall 0,262, train acc 0,786 (brecha 13,5 pp).
  - **Modelos por separado** (recalculados): 0,660 / 0,647 / 0,630 de mean recall (media 0,646) y 0,642 / 0,654 / 0,633 de accuracy (media 0,643). Coinciden con `experiments.csv`; la semilla 3 difiere en 0,0005 (inferencia en CPU frente a GPU).
- **Conclusión:**
  - Mejora +0,6 pp de mean recall y +0,8 pp de accuracy sobre la media de los modelos, pero no llega a la mejor semilla (0,660). No supera el criterio de éxito de 1 pp.
  - Es una sola medida, con una ganancia menor que el ruido entre semillas (1,5 pp): **no concluyente**.
  - Triplica el coste de predecir. El modelo entregable sigue siendo uno solo (el del notebook), y el `prediction.json` de la entrega 2 sigue saliendo de él.
- **Siguiente paso:** quedan dos propuestas que no requieren cambiar el alcance de la entrega: TTA con las 4 rotaciones (IDEA-012) y traslaciones aleatorias (IDEA-013).

### EXP-024: Capacidad × regularización (entrega 2)
- **Fecha:** 2026-10-16 · **Etapa:** ffNN, regularización (entrega 2) · **Base:** `EXP-021_final` · **Estado:** ejecutado
- **Por qué:** DEC-013. Dropout y L2 solo se habían juzgado sobre 4 × 256, una red elegida en la entrega 1 en parte porque las mayores memorizaban.
- **Hipótesis:** con más capacidad, Dropout y L2 dejan de empujar al subajuste y ayudan [R5].
- **Cambio** (dos factores a propósito): anchura de las 4 capas ocultas ∈ {256, 512, 1024} × {ninguna, Dropout 0,2, Dropout 0,5, L2 1e-4}. Todo lo demás, como EXP-021. 3 semillas; la celda 256 sin regularización es EXP-021.
- **Resultados** (val mean recall, media ± desviación típica; entre paréntesis, brecha en accuracy en pp):

| Anchura (parámetros) | Ninguna | Dropout 0,2 | Dropout 0,5 | L2 1e-4 |
|---|---|---|---|---|
| 256 (400 653) | **0,646 ± 0,015** (12,1) | 0,561 ± 0,030 (3,1) | 0,445 ± 0,011 (1,8) | 0,625 ± 0,009 (7,8) |
| 512 (1 194 509) | 0,644 ± 0,005 (11,4) | 0,594 ± 0,013 (3,5) | 0,506 ± 0,007 (2,5) | 0,645 ± 0,003 (8,5) |
| 1024 (3 961 869) | 0,640 ± 0,013 (8,8) | 0,624 ± 0,008 (5,3) | 0,545 ± 0,014 (3,0) | 0,638 ± 0,007 (8,4) |

- **Conclusión:**
  - **La interacción existe:** el coste de Dropout frente a la misma anchura sin regularizar baja al crecer la red. Con 0,2 pasa de −8,4 a −5,0 y a −1,7 pp; con 0,5, de −20,1 a −9,5 pp.
  - **Pero ninguna celda supera a EXP-021** (criterio de DEC-013).
    - Las redes anchas sin regularizar empatan con ella (0,644 y 0,640) y no memorizan más: con BN y rotaciones, la brecha baja de 12,1 a 8,8 pp con 1024 unidades.
    - La mejor celda regularizada, 512 + L2 (0,645 ± 0,003), también empata, con menos dispersión entre semillas.
  - **Se mantiene 4 × 256 sin Dropout ni L2**, que iguala a redes con 3 a 10 veces más parámetros. La conclusión de la entrega ya no depende de la red pequeña.
  - **Corrección (2026-10-17, CHG-027):** «no memorizan más» era una lectura excesiva de la brecha.
    - La brecha media de 4 × 256 (12,1 pp) se debe sobre todo a la semilla 1 (17,2 pp, mejor época 82); las otras dos dan 9,6 y 9,5. Las de 4 × 1024 son 10,8, 10,2 y 5,5. La brecha de cada run sigue sobre todo a la época conservada.
    - La diferencia de brecha entre anchuras no es concluyente. Con la validación empatada, una brecha menor solo significa ajustar menos el entrenamiento, no generalizar mejor. Se elige 4 × 256 por ser el modelo más simple.

### EXP-025: Desplazamientos aleatorios (entrega 2)
- **Fecha:** 2026-10-16 · **Etapa:** ffNN, regularización (entrega 2) · **Base:** `EXP-021_final` · **Estado:** ejecutado (descartado) · **Idea:** IDEA-013
- **Hipótesis:** desplazar la imagen enseña la invarianza a traslaciones [R8 §7.4, §9.2], igual que las rotaciones enseñaron la de orientación.
- **Cambio:** se añaden a las rotaciones desplazamientos aleatorios de hasta 1 o 2 píxeles en cada eje, con relleno por reflejo (`augment_batch`, op `shiftK`). 3 semillas.
- **Resultados:**
  - ≤ 1 px: mean recall 0,585 ± 0,021, val acc 0,600, train acc 0,638, brecha 3,8 pp.
  - ≤ 2 px: mean recall 0,552 ± 0,018, val acc 0,572, train acc 0,589, brecha 1,6 pp.
  - Pierden casi todas las clases; Helipad (−24 / −33 pp), Storage tank y Shipping container son las que más.
- **Conclusión:** **se refuta.** El mean recall cae 6,0 y 9,4 pp y la accuracy de train baja con él: subajuste. A 16 × 16, un píxel es el 6 % del lado y los recortes están encuadrados sobre el objeto, así que el desplazamiento recorta parte de él (explicación plausible, no verificada). EXP-027 comprueba si es el tamaño del desplazamiento.

### EXP-026: Test-time augmentation con las 4 rotaciones (entrega 2)
- **Fecha:** 2026-10-16 · **Etapa:** ffNN (entrega 2) · **Base:** `EXP-021_final` · **Estado:** ejecutado (descartado) · **Idea:** IDEA-012
- **Cambio:** cada modelo predice la imagen y sus 3 rotaciones y se promedian las probabilidades (`tools/ensemble_eval.py --tta rot90`). Se mide por semilla; no hay que entrenar. Fundamento: [R40].
- **Resultados** (`experiments/ensembles/EXP-026_tta_*`):
  - Mean recall por semilla, sin → con TTA: 0,660 → 0,643, 0,647 → 0,653 y 0,630 → 0,634. Media: 0,646 → **0,643 ± 0,010** (−0,3 pp).
  - Accuracy: 0,643 → **0,647 ± 0,007** (+0,4 pp).
- **Conclusión:** no mejora. La semilla 1, la que baja, es la que tiene la época elegida como máximo del mean recall sin TTA: parte de su 0,660 es efecto de esa selección, y cualquier cambio en la predicción lo hace volver hacia la media. Que TTA apenas cambie las métricas sugiere que la red ya responde de forma parecida a las cuatro orientaciones.

### EXP-027: Desplazamiento de medio píxel (entrega 2)
- **Fecha:** 2026-10-16 · **Etapa:** ffNN, regularización (entrega 2) · **Base:** `EXP-021_final` · **Estado:** ejecutado (descartado)
- **Hipótesis:** el problema de EXP-025 es el tamaño del desplazamiento; medio píxel podría ayudar.
- **Cambio:**
  - Reducir de 224 a 32 por área y promediar después bloques de 2 × 2 equivale a reducir de 224 a 16 (224 / 16 = 14 = 7 × 2). Por tanto, entrada de 32 + `AveragePooling2D(2)` delante de la red (`pool`, clave nueva) equivale a la entrada de 16.
  - Desplazar 1 px a 32 es desplazar medio píxel a 16.
  - Control: 32 + pooling sin desplazamiento. Variante: lo mismo con `shift1`. 3 semillas cada una.
- **Resultados:**
  - Control: 0,631 ± 0,005, val acc 0,637, train acc 0,718.
  - Medio píxel: 0,620 ± 0,012, val acc 0,626, train acc 0,692.
- **Conclusión:**
  - El control queda 1,5 pp por debajo de EXP-021, del orden del ruido entre semillas. Difieren el redondeo `uint8` de las cachés y la trayectoria del entrenamiento.
  - Frente al control, el medio píxel cuesta 1,1 pp (dentro del ruido) y no mejora.
  - **El patrón es monótono:** cuanto mayor el desplazamiento, peor, y siempre bajando el train. A esta resolución, la invarianza a traslaciones no se le puede dar a una red densa a través de los datos; es lo que una CNN aporta en su arquitectura (pesos compartidos y pooling [R8 §9.2]). Puente a la entrega 3.
- **Siguiente paso:** el modelo final de la entrega 2 sigue siendo `EXP-021_final`. Las vías dentro del alcance de la entrega están agotadas.

### EXP-028: Ablación de las paciencias del modelo final (entrega 2)
- **Fecha:** 2026-10-16 · **Etapa:** ffNN (entrega 2) · **Base:** `EXP-019_rot90` (EarlyStopping 15, ReduceLROnPlateau 6) · **Estado:** ejecutado
- **Por qué:** el usuario preguntó de dónde salía el salto de 0,617 a 0,646 tras la augmentation. Era EXP-021, que cambiaba **dos** factores a la vez y lo llamaba «entrenamiento más largo».
- **Diagnóstico previo, en los historiales:** con rotaciones, la validación oscila mucho entre épocas. Con paciencia 6, ReduceLROnPlateau bajaba la lr en las épocas 14, 27 y 37 (una por semilla); con 8, en la 71, la 43 y la 39. Que el modelo final suba también la media de las 5 últimas épocas (semilla 1: 0,594 → 0,646; semilla 2: 0,608 → 0,646) descarta que el salto sea solo un efecto de elegir el máximo de una curva ruidosa.
- **Cambio:** un factor cada vez, con 3 semillas: (a) solo ReduceLROnPlateau 6 → 8, con EarlyStopping 15; (b) solo EarlyStopping 15 → 25, con ReduceLROnPlateau 6.
- **Resultados** (los runs son deterministas):
  - (a) mean recall 0,646 ± 0,015, val acc 0,643 ± 0,011, mejores épocas 82 / 65 / 55. Idéntico a EXP-021, salvo que para 10 épocas antes.
  - (b) mean recall 0,617 ± 0,015, val acc 0,632 ± 0,005, mejores épocas 40 / 42 / 48. Idéntico a EXP-019_rot90, salvo que entrena 10 épocas más.
- **Conclusión:**
  - **Todo el salto (+2,9 pp) es la paciencia de ReduceLROnPlateau.** Con una validación ruidosa, una paciencia corta confunde el ruido con una meseta y baja la lr antes de tiempo. La paciencia de EarlyStopping no influye.
  - La ganancia se apoya en las semillas 1 y 2. En la 3, la primera bajada ya llegaba casi en la misma época (37 frente a 39) y no gana nada.
  - El modelo final no cambia (EXP-021 y la variante (a) dan las mismas mejores épocas; solo difiere cuándo para). Lo que cambia es cómo se explica.
  - Fundamento: ReduceLROnPlateau reacciona a una métrica ruidosa, y la paciencia es su único filtro frente a ese ruido. Es una explicación empírica a partir de los historiales: no hay una referencia específica.

### EXP-029: La convolución a igual entrada que la ffNN (entrega 3, D3-1)
- **Fecha:** 2026-10-17 · **Etapa:** CNN (entrega 3, DEC-017) · **Base:** `EXP-021_final` (ffNN de la entrega 2, 16×16) · **Estado:** ejecutado
- **Hipótesis:** el límite de la ffNN era su sesgo: cada píxel tiene su peso y no hay invarianza a traslaciones (EXP-024..027). Una CNN, con conectividad local, pesos compartidos y pooling [R8 §9.2], debería mejorar a igual entrada y con parámetros parecidos.
- **Cambio:**
  - (a) Las capas densas se sustituyen por la CNN propia: 3 bloques de 2 × [Conv3×3 → BN → LeakyReLU 0,3] con 32, 64 y 128 filtros, MaxPool 2×2 tras cada bloque, global average pooling [R45] y Dense(13). El resto de la receta de EXP-021 no cambia: 16×16, rotaciones, Adam β₁ 0,95, batch 256, lr 3,6e-3 y paciencias 25/8.
  - (b) Igual que (a), con batch 64 y lr 1,8e-3 (regla de la raíz, como en EXP-010). Es el batch que exigirán las resoluciones mayores, por memoria.
  - 3 semillas cada una.
- **Resultados:**

| Variante | val mean recall | val acc | min recall | train mean recall | Brecha (acc, pp) | Mejor época | s/época | Parámetros |
|---|---|---|---|---|---|---|---|---|
| ffNN EXP-021 (base) | 0,646 ± 0,015 | 0,643 ± 0,011 | 0,258 | 0,774 | 12,1 | 67 | 1,0 | 400 653 |
| (a) CNN, batch 256 | 0,741 ± 0,011 | 0,746 ± 0,003 | 0,425 | 0,870 | 11,3 | 41 | 1,6 | 290 029 |
| (b) CNN, batch 64 | **0,751 ± 0,007** | 0,748 ± 0,006 | 0,394 | 0,910 | 15,2 | 48 | 6,2 | 290 029 |

- **Conclusión:**
  - **Se confirma, y con margen.** Solo con la arquitectura, a la misma entrada de 16×16 y con menos parámetros, el mean recall sube 9,5 pp y la accuracy 10,3. Es más que toda la entrega 2 junta (+6,8 pp), y confirma el diagnóstico: lo que fallaba era el sesgo de la red densa.
  - El recall mínimo pasa de 0,258 a 0,425.
  - **El batch 64 empata con el 256** (+1,0 pp, dentro del ruido), así que se puede usar en las resoluciones mayores sin coste en resultados, aunque cada época tarda 4 veces más.
  - La brecha no cambia mucho: con batch 64 el train sube más.
- **Siguiente paso:** D3-2, la resolución (EXP-030), con batch 64.

### EXP-030: Resolución de entrada de la CNN propia (entrega 3, D3-2)
- **Fecha:** 2026-10-17 · **Etapa:** CNN (entrega 3) · **Base:** `EXP-029_cnn16_bs64` · **Estado:** ejecutado
- **Hipótesis:** una CNN comparte pesos entre posiciones, así que más píxeles no le multiplican los parámetros como a la ffNN y debería aprovechar más detalle (DEC-016).
- **Cambio:** solo la resolución de entrada, 32, 64 y 112 (16 es EXP-029), con la misma red de 3 bloques, batch 64 y 3 semillas.
- **Resultados:**

| Entrada | val mean recall | val acc | min recall | train mean recall | Brecha (acc, pp) | Mejor época | s/época |
|---|---|---|---|---|---|---|---|
| 16 (EXP-029) | 0,751 ± 0,007 | 0,748 ± 0,006 | 0,394 | 0,910 | 15,2 | 48 | 6,2 |
| **32** | **0,770 ± 0,006** | 0,769 ± 0,010 | 0,401 | 0,871 | 9,0 | 58 | 6,7 |
| 64 | 0,745 ± 0,017 | 0,745 ± 0,007 | 0,357 | 0,784 | 3,5 | 61 | 14,0 |
| 112 | 0,675 ± 0,018 | 0,689 ± 0,016 | 0,330 | 0,706 | 2,4 | 66 | 42,8 |

- **Conclusión:**
  - **32 × 32 es la mejor** (+1,9 pp sobre 16). Por encima, más píxeles empeoran.
  - **No es sobreajuste, es subajuste:** al subir la resolución baja también el mean recall de train (0,871 → 0,706) y la brecha casi desaparece.
  - **Explicación (cálculo, no medida):** el campo receptivo teórico de la última capa es de 36 px con 3 bloques. Cada Conv3×3 suma 2 × el paso acumulado, y cada MaxPool 2×2 suma 1 × el paso y lo duplica. A 32 px ese campo cubre toda la imagen; a 112 solo un trozo, y el global average pooling promedia rasgos locales sin ver nunca el objeto entero. El campo receptivo efectivo es además menor que el teórico [R52].
  - **Resolución y profundidad interactúan:** una imagen mayor necesita una red más profunda. Con 4 bloques el campo es de 76 px y con 5, de 156.
- **Siguiente paso:** la rejilla profundidad × resolución (EXP-031), adelantada a la optimización (DEC-018).

### EXP-031: Profundidad × resolución de la CNN propia (entrega 3, D3-3)
- **Fecha:** 2026-10-18 · **Etapa:** CNN (entrega 3, DEC-018) · **Base:** `EXP-030` (3 bloques) · **Estado:** ejecutado
- **Hipótesis:** con más bloques, el campo receptivo (76 px con 4 y 156 con 5) cubre las imágenes grandes y desaparece el subajuste de EXP-030 a 64 y 112.
- **Cambio:** rejilla de dos factores a propósito (DEC-013, DEC-018): bloques {4, 5} (filtros 32 → 256 o 32 → 512, duplicándose; 1 178 477 y 4 724 845 parámetros) × resolución {32, 64, 112}. Las celdas de 3 bloques son EXP-030. 3 semillas, cada run en su propio proceso.
- **Resultados** (val mean recall, media ± desviación típica; entre paréntesis, val acc, mean recall de train y s/época):

| Entrada | 3 bloques (290 029) | 4 bloques (1 178 477) | 5 bloques (4 724 845) |
|---|---|---|---|
| 32 | 0,770 ± 0,006 (0,769; 0,871; 6,7 s) | **0,784 ± 0,010** (0,772; 0,916; 8,2 s) | 0,781 ± 0,004 (0,764; 0,920; 9,9 s) |
| 64 | 0,745 ± 0,017 (0,745; 0,784; 14,0 s) | 0,778 ± 0,011 (0,776; 0,895; 15,4 s) | 0,787 ± 0,013 (0,776; 0,904; 17,6 s) |
| 112 | 0,675 ± 0,018 (0,689; 0,706; 42,8 s) | 0,775 ± 0,024 (0,767; 0,835; 48,1 s) | 0,788 ± 0,002 (0,781; 0,916; 51,4 s) |

- **Conclusión:**
  - **Se confirma la interacción.** A 112, la profundidad recupera +11,3 pp (de 0,675 con 3 bloques a 0,788 con 5), y el mean recall de train sube con ella (0,706 → 0,916): el subajuste de EXP-030 era de campo receptivo. A 32, la profundidad aporta menos (+1,4 pp con 4 bloques), porque 4 bloques ya cubren la imagen.
  - **Las tres mejores celdas empatan dentro del ruido:** 32/4 (0,784), 64/5 (0,787) y 112/5 (0,788), a 0,4 pp entre ellas.
  - **Se elige 32×32 con 4 bloques:** con la validación empatada, el modelo más simple y barato (1,2 M de parámetros frente a 4,7 M; 8 s/época frente a 51, 6 veces menos). Las fases siguientes se abaratan en esa proporción.
  - La celda 112/5 queda como referencia: la receta final se reentrenará a 112 (D3-7) para comparar en igualdad con las arquitecturas conocidas (DEC-016).
- **Siguiente paso:** D3-4, optimización (lr y batch) sobre 32×32 con 4 bloques.

### EXP-032: Optimización de la CNN propia (entrega 3, D3-4)
- **Fecha:** 2026-10-18 · **Etapa:** CNN (entrega 3) · **Base:** `EXP-031_r32_b4` (batch 64, lr 1,8e-3) · **Estado:** ejecutado
- **Hipótesis:** la receta heredada de la ffNN (lr y batch) puede no ser la óptima para la CNN (DEC-017).
- **Cambio:** un factor cada vez, con 3 semillas: lr ×0,5 (0,9e-3) y ×2 (3,6e-3); batch 32 y 128 con la lr escalada por la regla de la raíz (1,27e-3 y 2,55e-3) [R29].
- **Resultados:**

| Variante | val mean recall | val acc | min recall | train mean recall | Brecha (pp) | s/época |
|---|---|---|---|---|---|---|
| Base: batch 64, lr 1,8e-3 | 0,784 ± 0,010 | 0,772 ± 0,006 | 0,466 | 0,916 | 13,2 | 8,2 |
| lr 0,9e-3 | 0,776 ± 0,014 | 0,761 ± 0,017 | 0,382 | 0,920 | 14,5 | 8,3 |
| lr 3,6e-3 | 0,788 ± 0,012 | 0,773 ± 0,006 | 0,425 | 0,883 | 9,3 | 8,4 |
| batch 32 | 0,785 ± 0,003 | 0,768 ± 0,002 | 0,436 | 0,948 | 17,2 | 16,6 |
| **batch 128** | **0,785 ± 0,012** | 0,775 ± 0,006 | 0,433 | 0,887 | 10,0 | **4,9** |

- **Conclusión:**
  - La receta es robusta: todas las variantes empatan dentro del ruido (0,776-0,788).
  - **Se elige batch 128 con lr 2,55e-3:** misma validación y épocas 1,7 veces más rápidas, como en la entrega 1 (EXP-010).
  - El batch pequeño sube el train (0,948 con batch 32) sin subir la validación.
- **Siguiente paso:** D3-5, el resto de la capacidad (EXP-033).

### EXP-033: El resto de la capacidad de la CNN propia (entrega 3, D3-5)
- **Fecha:** 2026-10-18 · **Etapa:** CNN (entrega 3) · **Base:** `EXP-032_bs128` (4 bloques 32 → 256, 2 convs por bloque, MaxPool, GAP; 32×32, batch 128) · **Estado:** ejecutado
- **Hipótesis:** ajustar la anchura, las convoluciones por bloque, el tipo de reducción espacial y la cabeza puede afinar la arquitectura elegida por profundidad y resolución.
- **Cambio:** un factor cada vez, con 3 semillas.
  - Anchura ×0,5 (16 → 128) y ×2 (64 → 512).
  - 1 o 3 convoluciones por bloque.
  - Reducción espacial con AveragePool o con convolución de stride 2 [R46], en lugar de MaxPool.
  - Cabeza Flatten en lugar de global average pooling [R45].
- **Resultados:**

| Variante | val mean recall | Δ (pp) | val acc | train mean recall | Brecha (pp) | Parámetros | s/época |
|---|---|---|---|---|---|---|---|
| Base | **0,785 ± 0,012** | — | 0,775 | 0,887 | 10,0 | 1 178 477 | 4,9 |
| Anchura ×0,5 | 0,753 ± 0,009 | −3,2 | 0,754 | 0,901 | 13,5 | 296 637 | 4,2 |
| Anchura ×2 | 0,778 ± 0,008 | −0,7 | 0,772 | 0,899 | 11,8 | 4 697 805 | 9,2 |
| 1 conv por bloque | 0,756 ± 0,009 | −2,9 | 0,739 | 0,926 | 17,0 | 393 197 | 3,9 |
| 3 convs por bloque | 0,778 ± 0,013 | −0,7 | 0,771 | 0,855 | 7,1 | 1 963 757 | 9,1 |
| AveragePool | 0,770 ± 0,014 | −1,5 | 0,777 | 0,825 | 3,9 | 1 178 477 | 6,4 |
| Convolución con stride 2 | 0,754 ± 0,010 | −3,1 | 0,749 | 0,889 | 12,5 | 1 178 477 | 5,7 |
| Cabeza Flatten | 0,774 ± 0,008 | −1,1 | 0,757 | 0,940 | 17,6 | 1 188 461 | 5,2 |

- **Conclusión:**
  - **Se mantiene la base:** ningún cambio la mejora.
  - Reducir la capacidad cuesta unos 3 pp (anchura ×0,5, 1 conv por bloque); aumentarla empata (anchura ×2, 3 convs).
  - **MaxPool supera a las alternativas.** La convolución con stride 2 pierde 3,1 pp. El AveragePool pierde 1,5 pp y ajusta menos el train (0,825): promediar diluye los objetos pequeños (explicación plausible, no verificada).
  - **El global average pooling supera a Flatten.** Flatten memoriza más (brecha de 17,6 pp) y generaliza peor: la regularización estructural de [R45].
- **Siguiente paso:** D3-6, el desbalance de clases (EXP-034).

### EXP-034: Desbalance de clases en la CNN propia (entrega 3, D3-6)
- **Fecha:** 2026-10-18 · **Etapa:** CNN (entrega 3; DEC-009 actualizada) · **Base:** `EXP-032_bs128` · **Estado:** ejecutado
- **Hipótesis:** como en la ffNN (EXP-014: pesos con raíz, +3,2 pp), reponderar o remuestrear sube el recall de las clases pequeñas y el mean recall. Logit adjustment está diseñado para minimizar el error balanceado, es decir, para maximizar el mean recall [R51].
- **Cambio:** un tratamiento cada vez, con 3 semillas.
  - Pesos de clase con raíz e inversos [R10, R11].
  - Sobremuestreo con α = 0,5 y α = 1 (balanceado).
  - Logit adjustment con τ = 1: entropía cruzada de softmax(log p + τ log π), solo en entrenamiento; `LogitAdjustedCE`, CHG-028.
- **Resultados:**

| Tratamiento | val mean recall | Δ (pp) | val acc | min recall | Brecha (pp) |
|---|---|---|---|---|---|
| Ninguno (base) | 0,785 ± 0,012 | — | 0,775 ± 0,006 | 0,433 | 10,0 |
| Pesos con raíz | 0,790 ± 0,007 | +0,5 | 0,765 ± 0,009 | 0,421 | 13,0 |
| Pesos inversos | 0,796 ± 0,004 | +1,1 | 0,755 ± 0,012 | 0,370 | 8,8 |
| Sobremuestreo α 0,5 | 0,797 ± 0,006 | +1,2 | 0,774 ± 0,002 | 0,416 | 8,4 |
| Sobremuestreo α 1 | 0,789 ± 0,003 | +0,4 | 0,762 ± 0,004 | 0,406 | 10,1 |
| **Logit adjustment τ 1** | **0,803 ± 0,007** | **+1,8** | 0,762 ± 0,015 | 0,406 | 10,9 |

  Recall por clase con logit adjustment, frente a la base (pp): Helipad +12, Pylon +8, Fishing vessel +5, Storage tank +5, Dump truck +4, Cargo plane +3; Building −8, Small car −4, Truck −3.
- **Conclusión:**
  - **Solo logit adjustment supera el ruido** (+1,8 pp, frente a ~1,5), a costa de 1,3 pp de accuracy: pasa predicciones de las clases grandes a las pequeñas.
  - El sobremuestreo con α 0,5 empata dentro del ruido sin perder accuracy; es la segunda opción.
  - Las ganancias son mucho menores que en la ffNN (EXP-014): elegir la época por mean recall ya compensa parte del desbalance.
  - **Se adopta logit adjustment, τ = 1**, con el criterio de mejorar la métrica guía por encima del ruido.
- **Siguiente paso:** D3-7, criba de regularizadores (EXP-035). `LogitAdjustedCE` acepta ahora también label smoothing; antes, con logit adjustment, el label smoothing se habría ignorado en silencio.

### EXP-035: Regularización revisada en la CNN propia (entrega 3, D3-7)
- **Fecha:** 2026-10-18 · **Etapa:** CNN (entrega 3) · **Base:** `EXP-034_logitadj1` (rotaciones; logit adjustment) · **Estado:** ejecutado
- **Hipótesis:** sobre un modelo con otro sesgo, la pregunta de la entrega 2 tiene otra respuesta. En particular, los desplazamientos, que rompían la ffNN (EXP-025), deberían funcionar con la convolución.
- **Cambio, en dos pasos:**
  - (1) Criba de 10 regularizadores con 1 semilla, comparados con la semilla 1 de la base (0,809):
    - Geométricos, añadidos a las rotaciones: D4, desplazamientos ±2 px, recorte con escala.
    - Fotométricos suaves: brillo, contraste, saturación y ganancia por canal.
    - De oclusión: Random Erasing [R47].
    - De mezcla: MixUp con α 0,2 [R49] y CutMix con α 1 [R50].
    - Clásicos: Dropout 0,3 en la cabeza, AdamW con weight decay 1e-4 [R32] y label smoothing 0,1.
  - (2) Las 3 mejores, con 3 semillas.
- **Resultados de la criba** (Δ en pp frente a la semilla 1 de la base: mean recall / accuracy):
  - AdamW: +0,7 / +0,2. Desplazamientos: +0,6 / +0,3. Borrado: +0,4 / +0,2. Recorte: +0,2 / −1,0. Dropout: +0,2 / −1,1.
  - D4: −0,1 / −0,8. MixUp: −0,2 / 0,0. Color: −0,5 / −0,9. Label smoothing: −0,9 / −2,4.
  - **CutMix: −1,6 / −3,5**, y su mean recall de train baja a 0,807: el parche pegado tapa objetos pequeños y centrados.
- **Confirmación** (3 semillas):

| Variante | val mean recall | Δ (pp) | val acc | min recall | train mean recall | Brecha (pp) |
|---|---|---|---|---|---|---|
| Base (rotaciones) | 0,803 ± 0,007 | — | 0,762 ± 0,015 | 0,406 | 0,901 | 10,9 |
| AdamW (wd 1e-4) | 0,810 ± 0,008 | +0,7 | 0,770 ± 0,004 | 0,421 | 0,920 | 12,6 |
| **+ desplazamientos ±2 px** | **0,821 ± 0,007** | **+1,8** | **0,779 ± 0,007** | 0,357 | 0,864 | **5,1** |
| + borrado | 0,805 ± 0,016 | +0,2 | 0,772 ± 0,006 | 0,413 | 0,939 | 14,8 |
| Las tres (EXP-036) | 0,820 ± 0,010 | +1,7 | 0,780 ± 0,013 | 0,373 | 0,877 | 6,5 |

- **Conclusión:**
  - **Los desplazamientos se confirman:** +1,8 pp de mean recall (por encima del ruido) y +1,7 de accuracy, y la brecha se reduce a la mitad.
  - Es la invarianza que la ffNN no podía aprender por los datos: en la ffNN a 16 px, 1 px de desplazamiento costaba −6 pp (EXP-025); aquí, 2 px a 32 px (el mismo 6 % del lado) suman. La convolución convierte la traslación en algo que la red puede aprovechar.
  - AdamW y el borrado quedan dentro del ruido, y combinarlos con los desplazamientos no suma (EXP-036: 0,820).
  - **Se adoptan rotaciones + desplazamientos**, la opción más simple.
  - **La CNN propia final** (D3-7) es `EXP-035_shift2`:
    - Arquitectura: 32×32, 4 bloques de 2 × [Conv3×3 → BN → LeakyReLU 0,3] (32 → 256), MaxPool y GAP.
    - Entrenamiento: logit adjustment, rotaciones + desplazamientos, Adam con batch 128 y lr 2,55e-3.
    - Resultado: **mean recall 0,821 ± 0,007 y accuracy 0,779 ± 0,007**, frente a 0,646 / 0,643 de la ffNN de la entrega 2 (+17,5 y +13,6 pp).
- **Siguiente paso:** la misma receta a 112 px con 5 bloques (EXP-037), para comparar en igualdad con las arquitecturas conocidas; después, la criba de lr de esas arquitecturas (EXP-038).

### EXP-037: La CNN propia final a 112 px (entrega 3, D3-7)
- **Fecha:** 2026-10-19 · **Etapa:** CNN (entrega 3) · **Base:** `EXP-035_shift2` (32 px) · **Estado:** ejecutado
- **Por qué:** las arquitecturas conocidas se comparan a 112 px (DEC-016), así que la CNN propia necesita también su versión a 112 para comparar en igualdad.
- **Cambio:** la receta final llevada a 112 px.
  - 5 bloques (32 → 512 filtros), que EXP-031 mostró necesarios por el campo receptivo a esa resolución.
  - Desplazamientos ±7 px: el mismo 6 % del lado que ±2 px a 32.
  - Batch 64 por memoria, con lr 1,8e-3 por la regla de la raíz; EXP-032 mostró que el batch no influye.
  - 3 semillas, cada una en su propio proceso.
- **Resultados:** mean recall 0,814 ± 0,021 (semillas 0,790 / 0,824 / 0,828), val acc 0,781 ± 0,018, min recall 0,377, brecha 7,0 pp; 4 724 845 parámetros; 52,8 s/época, frente a 4,9 a 32 px.
- **Conclusión:**
  - Empata con la versión de 32 px (0,821 ± 0,007), con más dispersión entre semillas y 10 veces más coste por época. Los desplazamientos también funcionan aquí: la celda 112/5 de EXP-031, sin desplazamientos ni logit adjustment, daba 0,788.
  - **El modelo final de la CNN propia sigue siendo el de 32 px.** Este es su punto de comparación a 112 px con las arquitecturas conocidas.

### EXP-038: Criba de lr de las arquitecturas conocidas desde cero (entrega 3, D3-8)
- **Fecha:** 2026-10-19 · **Etapa:** CNN conocidas (entrega 3, DEC-015) · **Base:** la receta de `EXP-037` a 112 px (logit adjustment, rotaciones + desplazamientos ±7 px, Adam β₁ 0,95, batch 64) · **Estado:** ejecutado
- **Por qué:** cada arquitectura tiene su lr óptima, y compararlas con una lr común castigaría a alguna. La criba solo ordena las lr: 1 semilla, como mucho 40 épocas y paciencias 10/4.
- **Cambio:** lr ∈ {3e-4, 1e-3, 3e-3} para VGG16 [R41], ResNet50 [R42] e InceptionV3 [R38]. Como el óptimo de ResNet50 e InceptionV3 quedó en el borde de la rejilla, se añadió 1e-4 para las dos.
- **Resultados** (val mean recall, semilla 1; entre paréntesis, mejor época / épocas):

| Modelo (parámetros; s/época) | 1e-4 | 3e-4 | 1e-3 | 3e-3 |
|---|---|---|---|---|
| VGG16 (14,7 M; 67 s) | — | 0,077 (1/11) | **0,761** (31/40) | 0,077 (1/11) |
| ResNet50 (23,6 M; 86 s) | 0,799 (36/40) | **0,824** (31/40) | 0,724 (40/40) | 0,565 (40/40) |
| InceptionV3 (21,8 M; 68 s) | 0,768 (37/40) | **0,781** (33/40) | 0,700 (37/40) | 0,483 (38/40) |

- **Conclusión:**
  - **VGG16, sin BatchNorm, solo entrena en una ventana estrecha de lr.**
    - Con 3e-4, la pérdida se queda plana en 2,314 desde la primera época, prácticamente la entropía de las clases (2,31): predice siempre la misma clase.
    - Con 3e-3 diverge: la pérdida salta a 13,8 y la red queda muerta.
    - Es el problema conocido de las VGG profundas sin BN, que sus autores sortearon inicializándolas desde redes menos profundas [R41], y que BatchNorm resolvió después [R6].
    - Se mantiene el VGG16 de Keras sin BN, porque es el que tendrá pesos de ImageNet en la entrega 4 (DEC-015). Si alguna de las 3 semillas colapsa a 1e-3, se añadirá la variante con BN (plan B).
  - **ResNet50 e InceptionV3 prefieren lr bajas,** con el óptimo en 3e-4 (1e-4 ya es peor). Las conexiones residuales [R42] y la BN de ambas las hacen entrenables en todo el rango, a diferencia de VGG16.
  - **ResNet50 empata ya con la CNN propia** (0,824 con 1 semilla y presupuesto corto, frente a 0,821), con 20 veces más parámetros.
  - **lr elegidas:** VGG16 1e-3, ResNet50 3e-4 e InceptionV3 3e-4.
- **Siguiente paso:** EXP-039, 3 semillas con la receta completa.

### EXP-039: Las arquitecturas conocidas desde cero con la receta completa (entrega 3, D3-8)
- **Fecha:** 2026-10-19 · **Etapa:** CNN conocidas (entrega 3, DEC-015) · **Base:** EXP-038 (lr elegidas: VGG16 1e-3, ResNet50 3e-4, InceptionV3 3e-4) · **Estado:** ejecutado
- **Por qué:** EXP-038 solo ordenaba las lr, con 1 semilla y como mucho 40 épocas. Para compararlas con la CNN propia a 112 px (EXP-037) hace falta el mismo presupuesto y varias semillas.
- **Cambio:** el único factor respecto a EXP-038 es el presupuesto completo de `D3_FINAL112`: hasta 300 épocas, con paciencias 25 / 8 en lugar de 10 / 4. Se usan 3 semillas, cada run en su propio proceso, en `sweep.py d3_known`.
- **Resultados** (validación, media ± σ entre semillas; brecha = train acc − val acc):

| Modelo | Semillas válidas | Mean recall | Val acc | Min recall | Brecha | Épocas (mejor) |
|---|---|---|---|---|---|---|
| VGG16, lr 1e-3 | 2 de 3 (ISS-014) | 0,758 ± 0,003 | 0,724 ± 0,000 | 0,358 | 10,9 pp | 53 (28), 91 (66) |
| ResNet50, lr 3e-4 | 3 | 0,818 ± 0,003 | 0,782 ± 0,009 | 0,391 | 9,6 pp | 63–89 (38–64) |
| InceptionV3, lr 3e-4 | 3 | **0,825 ± 0,005** | **0,795 ± 0,002** | **0,468** | 16,8 pp | 85–105 (60–80) |
| CNN propia a 112 px (EXP-037) | 3 | 0,814 ± 0,021 | 0,781 ± 0,018 | 0,377 | 7,0 pp | — |
| CNN propia a 32 px (EXP-035_shift2) | 3 | 0,821 ± 0,007 | 0,779 ± 0,007 | — | — | — |

  - Semillas de mean recall: VGG16 0,756 / 0,760 / colapso (0,077); ResNet50 0,822 / 0,818 / 0,815; InceptionV3 0,820 / 0,831 / 0,824.
  - Coste: ResNet50 ~84 s/época, VGG16 e InceptionV3 ~67-79 s/época. El barrido entero duró ~14 h de GPU, contando el relanzamiento de ISS-013.
  - Recall por clase (media de las semillas válidas): la clase más difícil es Truck en todas (VGG16 0,36, ResNet50 0,39, InceptionV3 0,47, CNN propia a 112 0,38), seguida de Bus y Dump truck. El min recall más alto de InceptionV3 es casi todo Truck.
- **Conclusión:**
  - **InceptionV3, ResNet50 y la CNN propia empatan en mean recall:** quedan dentro de ~1 pp, menos que el ruido entre semillas (σ hasta 2 pp en EXP-037). La CNN propia a 112 px tiene 4,7 M parámetros, unas 5 veces menos que ResNet50 (23,6 M) e InceptionV3 (21,8 M).
  - **InceptionV3 destaca en la clase difícil:** Truck sube ~8 pp frente a ResNet50 y la CNN propia. Es la red que más sobreajusta (train acc 0,96, brecha 16,8 pp), así que su ventaja no viene de ajustar peor el train.
  - **VGG16 queda claramente por detrás** (−6 pp) y es frágil: una de las 3 semillas colapsa a una sola clase con la misma lr que funciona en las otras dos (ISS-014). Era el escenario previsto en EXP-038 para el plan B, la variante con BatchNorm [R6].
  - El ranking de EXP-038 se mantiene salvo InceptionV3, que con el presupuesto completo pasa de 0,781 a 0,825: con 40 épocas estaba cortada lejos de su mejor época (60-80).
- **Siguiente paso:** decidir sobre VGG16 (relanzar la semilla 3, aplicar el plan B con BN o dejar las 2 semillas documentadas); después, MixUp y la comparación final de D3-8 para elegir el modelo del notebook.

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

### CHG-009: Fracción de unidades inactivas en cada run (2026-10-03)
`experiment_utils.inactive_units`: fracción de unidades de la primera capa oculta cuya preactivación es ≤ 0 para todas las imágenes de validación (con ReLU, unidades muertas, que no reciben gradiente [R8 §6.3.1]). `run_experiment` la guarda en `history/<id>.json` (`val.inactive_units`) desde la fase 4; no se añade al CSV para no cambiar sus columnas con la fase 3 escribiendo en él. Comprobado con `EXP-005_r16`: 0,281, igual que la medida manual de EXP-005. Las medidas de EXP-005 y EXP-006 se calculan aparte, con los modelos guardados.

### CHG-010: El notebook entregable pasa al modelo final (2026-10-03)
`ffNN_example.ipynb` mantiene su estructura (mismas secciones y celdas, en el mismo orden) y cambia lo necesario para entrenar el modelo de EXP-018, con una celda markdown breve antes de cada bloque modificado:
- Celda 1: semilla fija y `enable_op_determinism` (EXP-001). Split estratificado (EXP-001). Sin el comentario de "resampling" (ISS-004).
- Nueva celda con la caché a 16×16 (`xview_data.load_cache`, CHG-007).
- Modelo funcional `Input → Rescaling → Flatten → 4 × [Dense → BN → ReLU] → Dense(13)`; Adam lr = 3,6e-3, β₁ = 0,95.
- Callbacks sobre `val_mean_recall` (`EpochMetrics` de `experiment_utils`, que también calcula `val_loss` y `val_accuracy` en una sola pasada).
- El generador pasa a `CachedSequence` (caché + D4 + pesos de clase con raíz).
- Validación y test predichos por lotes (`predict_probs`); el test lee `cache/test_16.npy` y la lista de test filtra `*.tif` (ISS-006).
- Nueva tabla resumen del progreso por fase (sacada de `experiments.csv`).
- `EXP_ID = 'EXP-018'`.
- **Ejecutado** de arriba abajo sin interfaz (`tools/run_notebook.py`): reproduce exactamente el run `EXP-018_final` (ver EXP-018).
- `experiment_utils.py`: `set_memory_growth` se omite si TF ya está inicializado (el notebook importa el módulo después de usar la GPU; antes lanzaba `RuntimeError`).

### CHG-011: Sistema de entregas (informe + slides Beamer para Overleaf) y entrega 1 (2026-10-03)
No cambia el modelo ni los experimentos: lee `experiments/` (solo lectura) y LOGBOOK.md y genera los documentos de cada entrega del curso. Plan aportado por el usuario; decisiones de estilo en DEC-007.
- **Scripts** (`scripts/`): `build_results.py` (macros `\res{clave}` con coma decimal y tablas `booktabs` en `generated/`, más `results_index.json` con el origen de cada clave), `build_figures.py` (figuras redibujadas desde `experiments.csv` y `history/` según `figure_specs.yaml`, con `figure_log.md` de lo incluido y omitido), `plot_style.py`, `draft_sections.py` (instala los borradores de `drafts/` en `sections/` sin pisar los archivos editados a mano: hash en `.build_state.json`, y si hay edición, `.proposed.tex`), `check_bundle.py` (comprobaciones sin compilar: `\input`/`\includegraphics` existentes, `\res{}` definidas, llaves y entornos, `\ref`/`\label`, paquetes en lista blanca, caracteres no válidos en pdfLaTeX, 8–10 frames con `\note` y tiempo, estimación de páginas) y `pack_bundle.py` (zip). `Makefile`: `make results|figures|sections|check|refresh|bundle D=01`.
- **`plot_theme.py`:** el estilo de las gráficas (paleta, `rcParams`) se saca de `plot_utils.py` a un módulo común para que las figuras de las entregas y las de los experimentos sean iguales. `plot_utils.py` lo importa; sus gráficas no cambian.
- **Entrega 1** (`reports/delivery_01/`): `delivery.yaml` (qué variante representa cada fase y cada cifra), informe de ~5 páginas estimadas y 10 frames de contenido (10 min de notas), todo en español y con cifras solo a través de macros. Baseline EXP-001 (EXP-000 tiene otro split). `review_notes.md` recoge lo que hay que revisar en Overleaf, dos discrepancias con esta bitácora (ganancia de `Rescaling` +8,4 pp y no +8,3: aquí se restaron valores redondeados; aceleración del batch 256 ×16,4 con 3 semillas frente a ×15 con una) y las afirmaciones sin cifra directa en `experiments/`.
- **Plantilla** (`reports/template/`): la del usuario (extraída de su plantilla Beamer, ya compilada en Overleaf), sin cambios en `common/`, `slides.tex` ni `report.tex`: macros `\res`, `\safegraphics`, `\safetable`, `\todoslot`, `\speakernote`, `\decisionpoint`, `\decisionblock`, `\keymessage`, y secciones `slides_01..07` y `report_01..06`. Los scripts se adaptaron a ella (`meta.tex` con `\Delivery*`, tablas solo con el `tabular`, claves `best-*` y `n-experiments`; `check_bundle.py` bloquea si queda algún `\todoslot` y suma los minutos de `\speakernote`). Cambios respecto a su README, por DEC-007: figuras `figures/<id>.png` (no `<id>_<contexto>.pdf`) y pies generados (`\figcaption{id}`, que carga `results_macros.tex`). Primera versión de esta entrega sobre una plantilla reconstruida a partir de la descripción del plan, sustituida al recibir la del usuario. Comando `.claude/commands/new-delivery.md` para las entregas siguientes. Dependencia nueva en el `.venv`: `pyyaml`.
- **Verificado:** `make bundle D=01` → 0 errores; `check_bundle.py` detecta en una prueba negativa una macro sin definir, un carácter Unicode no válido, un entorno sin cerrar, una figura inexistente y un `\ref` sin `\label`; `draft_sections.py` conserva una sección editada y deja el borrador nuevo como `.proposed.tex`. No hay LaTeX local: la primera compilación será en Overleaf.

### CHG-012: Nombres del equipo (2026-10-03)
A petición del usuario: el equipo es Alex Sierra Alcalá, Roberto Inclán Alfonso y Mario Hernández Trillo. Se actualizan `\AuthorName` y `\AuthorShort` en `common/static_meta.tex` (plantilla y entrega 1) y los campos "Alumno 1/2/3" de la cabecera de `ffNN_example.ipynb` (solo esa celda markdown; el notebook no se vuelve a ejecutar). Paquete de la entrega 1 regenerado.

### CHG-013: Entrega 1 recortada a las fases 0-5 y notebook alineado con ella (2026-10-03)
Por DEC-008. Antes de recortar, el estado completo (entrega 1 con las fases 0-9 y notebook con EXP-018) se guardó en el commit `b72efda` con la etiqueta local `ffnn-full-plan`.
- **`ffNN_example.ipynb`:** vuelve al modelo de `EXP-010_bs256_sqrt_60ep` (16×16 `/255` → 4 × [`Dense(256, he_normal)` → LeakyReLU(0,3)] → `Dense(13)`; Adam AMSGrad, lr = 1,2e-3, β₁ = 0,95, clipnorm 1; batch 256; hasta 60 épocas con EarlyStopping 10 y ReduceLROnPlateau 4 sobre `val_mean_recall`), sin pesos de clase ni augmentation; `EXP_ID = 'EXP-010'`; markdown y tabla de progreso limitados a las fases 0-5. **Ejecutado** de arriba abajo: run `EXP-010_20261003-160510`, idéntico al run del barrido (mejor época 20 de 30, val acc 0,6091, mean recall 0,5736, min recall 0,2398). `model.keras` y `prediction.json` (2 365 imágenes) salen ahora de este modelo.
- **Entrega 1** (`reports/delivery_01/`): `delivery.yaml` con las fases 0-5 y final EXP-010; figuras nuevas `lr_sweep` (fase 5.1) y `stopping` (mean recall de validación por época de cada semilla del modelo final, con la mejor época y la parada); se quitan las de regularización, desbalance, augmentation y ablación. Texto reescrito para el alcance; 9 frames, ~5 páginas estimadas, `make bundle` sin errores.
- **Scripts:** `build_figures.py` con los tipos `group_sweep` y `stopping`; la tabla de ablación es opcional en `build_results.py`; `generated/tables/` y `figures/` se vacían antes de regenerarse.

### CHG-014: Diapositivas de la entrega 1 coherentes con sus figuras y pies que caben (2026-10-03)
A petición del usuario (en Overleaf, el texto de algunas diapositivas no correspondía a su figura y el pie se salía del frame).
- **Regla nueva** (también en la plantilla): las viñetas de cada frame solo afirman lo que se ve en su figura; lo demás va a la nota del orador.
- "Normalizar la entrada" tenía la figura de parada del modelo final: ahora usa una figura nueva, `normalisation` (pérdida de train por época de EXP-001 y EXP-002, escala log; solo train por ISS-007). La parada temprana pasa a su propio frame con la figura `stopping`. "Arquitectura" deja de dar mean recall y LeakyReLU (la figura es de accuracy) y "Resolución", las ReLU muertas (no se ven en la figura): van a las notas. 10 frames, 10 min.
- **Pies cortos para slides:** `caption_slides_es` en `figure_specs.yaml` → `\figcaptionshort{id}`; figuras con `height=0.62\textheight,keepaspectratio`. `check_bundle.py` estima la longitud de cada pie (con macros resueltas) y da error si pasa de ~2 líneas (80 caracteres en columna, 150 a todo el ancho), y aviso si una figura no tiene límite de altura. Probado en negativo con el pie largo anterior. Es una estimación: la comprobación real es compilar en Overleaf.
- `build_figures.py`: opciones `logy`, `show_val` y `criterion` en `learning_curves`.
- **Frame de parada rehecho** (segunda revisión del usuario: no tenía sentido). La figura de las 3 semillas solo mostraba dónde paraba cada una, no por qué. La nueva `stopping` usa la semilla 1 (el run del notebook): accuracy de train y de validación y mean recall de validación por época, con las bajadas de lr, la mejor época y la parada; sombreado lo posterior a la mejor época. Lectura, con macros `stop-*` sacadas del historial (`stopping_run` en `delivery.yaml`): hasta la época 19 suben juntas; la primera bajada de lr da la mejor época (20); después train pasa de 0,699 a 0,743 y validación se queda en 0,609-0,612 (brecha de 9,0 a 13,1 pp); EarlyStopping para en la 30. El frame deja el esquema hipótesis/evidencia/resultado, que aquí no encajaba, y la sección "Resultado final" del informe usa la misma lectura.
- **Informe** (tercera revisión del usuario: juntaba figuras sin relación y no todas las fases citaban su figura): se acaban los pares de figuras en un mismo flotante (resolución + capacidad, learning rate + parada). Ahora hay una figura por flotante (`[!htbp]`), colocada justo después del párrafo que la comenta, y cada una se cita en su texto: normalización (fase 1, figura nueva también en el informe), resolución (fase 2), capacidad (fases 3-4), learning rate (fase 5), y parada, progreso y recall por clase (resultado final). 7 figuras; ~5,5 páginas estimadas.

### CHG-015: Documentos de la entrega 2 (2026-10-03)
`reports/delivery_02/` con la plantilla del usuario: `delivery.yaml` (punto de partida EXP-010, final EXP-021; pasos: BatchNorm → rotaciones → entrenamiento más largo), `figure_specs.yaml` (curvas de train y validación entrega 1 frente a 2, curvas de validación de dropout y L2, BatchNorm, augmentation, progreso y recall por clase, con pies cortos para las slides), informe (~5 páginas estimadas) y slides (10 frames, 10 min), con las reglas de DEC-010 (mismas figuras y en el mismo orden en informe y slides; cada imagen del informe con su pie y su etiqueta). Cuadro de cambios sobre BN + rotaciones (EXP-020) generado como tabla de ablación. `make bundle D=02` sin errores.
- **Scripts** (compartidos con la entrega 1): `build_results.py` describe el modelo de partida desde su configuración (`describe_model`, `describe_training`), cabeceras genéricas en `baseline_vs_final`, macros por variante `<clave>-gap-acc` (train − val, pp) y `<clave>-l2`; `build_figures.py`, opción `logx` en las curvas de validación (`strength`).
- **Coordinación:** otra sesión de Claude (`image-recognition-10`) estaba modificando a la vez la entrega 1 y los scripts (pies cortos `\figcaptionshort`, límite de altura de figuras en slides, nueva figura de parada). La entrega 2 adopta esas convenciones; no se tocó `reports/delivery_01/`. Se le avisó de los cambios en los scripts compartidos.
- El notebook entregable sigue con el modelo de la entrega 1 (pendiente de decidir si se alinea con EXP-021). → Alineado en CHG-019.

### CHG-016: Informe y slides de la entrega 1 con las mismas figuras y en el mismo orden (2026-10-03)
Por DEC-010. El informe usaba 7 figuras y las slides 8, en distinto orden: `class_distribution` no estaba en el informe y la parada aparecía en la fase 1 de las slides pero en el resultado final del informe.
- **Informe:** `class_distribution` en "Objetivo y datos"; el análisis del punto de parada pasa del resultado final a la fase 1, como en las slides. Figuras relacionadas en la misma fila, cada una con su pie y su número (dos `minipage` con `\caption` y `\label` en un mismo flotante): normalización + parada (fase 1) y resolución + capacidad (fases 2-4). Las demás, solas. Pies del informe más cortos para esas cuatro. Orden común: class_distribution, normalisation, stopping, resolution_sweep, capacity, lr_sweep, progress, per_class_recall. ~5,6 páginas estimadas.
- **`check_bundle.py`** (funciones nuevas, sin cambiar las existentes): error si una figura está en un documento y no en el otro, o si el orden difiere; en el informe, error si un flotante tiene más de 2 imágenes o una figura no se cita con `\ref`, y aviso si dos imágenes comparten pie.
- Plantilla (`reports/template/README.md`): reglas añadidas.

### CHG-017: Unidades inactivas de la fase 2 en los historiales (2026-10-04)
`tools/backfill_inactive_units.py` añade `val.inactive_units` (la medida de CHG-009, con la misma función `experiment_utils.inactive_units`) al historial de los runs anteriores a CHG-009, calculada con el modelo guardado de cada run (`experiments/models/<id>.keras`). No cambia nada más del historial. Se aplicó a los 16 runs de EXP-005, los únicos de las fases 0-5 sin la medida (EXP-001 a EXP-004 no tienen capa oculta).
- **Comprobación:** recalculada con el script, la de `EXP-006_h256-256-256-256` da 0,3359375, igual que la guardada por `run_experiment`. La de `EXP-005_r16` (0,2812) coincide con la medida manual de EXP-005.
- **Por qué:** el informe de la entrega 1 decía "posiblemente por ReLU inactivas (no medido)" porque la medida de EXP-005 solo estaba en esta bitácora y no en `experiments/`. Ahora es evidencia, y la figura de la fase 4 la usa (CHG-018).
- **Resultado** (anotado en EXP-005): la fracción crece con la resolución y con la lr; con la mejor lr de cada tamaño queda en un 25-32 %. Fundamento: ReLU muertas [R8 §6.3.1]; paso de Adam ≈ lr por peso [R4 §2.1].

### CHG-018: Entrega 1 ampliada: más detalle de las fases 0-5, LeakyReLU con figura y slides en el orden del informe (2026-10-04)
A petición del usuario: ajustar las slides a sus cambios en el informe, explicar la elección de LeakyReLU con una gráfica del experimento y contar mucho más de lo que se hizo en las fases 0-5.
- **Ediciones del usuario adoptadas:** el usuario editó a mano las seis secciones del informe en `overleaf/sections/`. Esas versiones pasan a ser los borradores (`drafts/`) antes de ampliarlas, así que no se pierde ninguna edición. Sus cambios: la fase 1 separa en dos decisiones la normalización y los callbacks; el análisis de la parada pasa al resultado final; se marcan las decisiones tomadas con una semilla y dentro del ruido; la fase 5 indica que cambia a la vez β₁, el batch y la escala de la lr; se matiza el optimismo de las cifras.
- **Sus TODO, resueltos con datos:**
  - Pie de `class_distribution`: es el conjunto etiquetado completo, no solo train. Corregidos el pie y el eje.
  - `gain-normalisation`: es EXP-002 − EXP-001, y solo cambia `Rescaling`. Se añade `gain-phase1` para la fase completa.
  - Método de reducción: promedio por área, `tf.image.resize(method='area')`, un filtro paso bajo [R19].
  - Bajadas de la lr del modelo final: son tres, en las épocas 20, 25 y 29 (macros `stop-lr-drops`, `stop-later-lr-drops`).
  - "Truck probablemente se confunde": ahora con la media de las matrices de confusión de las 3 semillas: Small car 0,213, Bus 0,119, Dump truck 0,109.
  - "ReLU inactivas (no medido)": medido (CHG-017).
  - **Corregida** una frase del usuario en la fase 4, "no se separó el efecto de la activación del de la inicialización": la base ReLU (EXP-006) ya usaba `he_normal`, así que entre ReLU y LeakyReLU solo cambia la activación.
- **Informe ampliado:**
  - Fase 0: semilla y split.
  - Fase 1: el porqué de la pérdida desbordada (paso de Adam ≈ η·Σ|xᵢ| [R4]); la estandarización por canal (EXP-004, −3,8 pp; pérdida inicial 60,2 frente a 13,6); σ de la val acc en las 10 últimas épocas (0,067 → 0,002) y las tres métricas vigiladas.
  - Fase 2: capa oculta añadida; reducción por área; unidades inactivas (28 % → 93 %); las diez primeras épocas a 224 en la distribución a priori (pérdida 2,28 ≈ entropía de clases 2,31); la lr óptima y las inactivas con ella; 8×8 y 24×24.
  - Fase 3: rejilla de 12 arquitecturas; inactivas en todas (20-34 %).
  - Fase 4: subsección propia con la explicación de ReLU/LeakyReLU [R8 §6.3.1; R21], las 7 activaciones con su inicializador [R2; R24; R25] y la elección.
  - Fase 5: tres decisiones (optimizador y lr con momentum, Nesterov, β, RMSprop y el ruido entre semillas [R27]; batch con las reglas de la raíz y lineal [R28; R29; R30]; schedules y ajustes de Adam [R28; R31; R32]). Se añade el coste del límite de 20 épocas (−7,4 pp para la misma configuración y semilla) y el cuadro `phase5_variants` con 19 variantes.
  - Bibliografía del informe: de 5 a 19 entradas, todas de "Referencias externas".
- **Figura nueva `activation`** (fase 4, dos paneles). Izquierda: % de unidades inactivas según la resolución, una línea por lr (los 16 runs de EXP-005; rombos, la mejor lr de cada tamaño). Derecha: mean recall de validación y de train de las 7 activaciones, con el % de inactivas de cada modelo.
- **Slides** en el orden del informe (DEC-010): fases 1, 2, 3, 4 (frame nuevo de LeakyReLU con la figura `activation`) y 5, progreso, parada (pasa al bloque de resultados, como en el informe, con la decisión de los callbacks de la fase 1), resultado final y conclusiones. Viñetas y notas actualizadas con las matizaciones del usuario. Referencias de las slides: 6.
- **Scripts** (compartidos con la entrega 2; `make results`, `make figures` y `make check` de la entrega 2 dan las mismas tablas y ninguna macro desaparece):
  - `build_results.py`:
    - `Data.runs` acepta una lista de runs, para las semillas registradas con otro id (SGD con momentum 0,9).
    - Macros nuevas por variante: `-inactive`, `-train-mean-recall`, `-actual-epochs`, `-inputs`, `-width`, `-leaky-alpha`.
    - Secciones opcionales en `delivery.yaml`: `history_values` (valores del historial: época concreta, mediana de las primeras épocas, σ de las últimas, mín./máx. sobre un experimento), `confusions` (proporciones de la matriz de confusión del modelo final) y `tables` (cuadros de variantes).
    - Macro `class-entropy` y las bajadas de la lr del run de parada.
  - `build_figures.py`: tipo `activation`; el eje de `class_distribution` dice "labelled images (train + validation)".
- **`make bundle D=01`:** 0 errores y 3 avisos.
  - Informe estimado en ~9,5 páginas (la plantilla pide 4-6).
  - Slides: 11 frames con 11,5 min de notas (la plantilla pide 8-10 frames y unos 10 min).
  - Se deja así porque el usuario pidió más detalle. Recortar o no lo decide el usuario.

### CHG-019: El notebook entregable pasa al modelo de la entrega 2 (EXP-021) (2026-10-15)
A petición del usuario ("preparar todo lo relativo a la segunda entrega"; `review_notes.md` de la entrega 2 lo tenía pendiente). Por DEC-009, cada entrega parte del modelo final de la anterior, y el notebook es el código entregable.
- **`ffNN_example.ipynb`**, solo lo que separa EXP-010 de `EXP-021_final` (su `config` en `experiments.csv`): BatchNorm antes de LeakyReLU en las 4 capas ocultas, con `Dense(use_bias=False)` porque β de BN sustituye al sesgo [R6 §3.2]; lr 3,6e-3 [EXP-013]; rotaciones aleatorias de 90° en el generador de train (`augment=['rot90']`, EXP-019); paciencias 25 / 8 y hasta 300 épocas (EXP-021); `EXP_ID = 'EXP-021'`; `config.json` del run con `augment` y `batchnorm`. Markdown del modelo y del entrenamiento y tabla de progreso con los pasos de las dos entregas.
- **Ejecutado** de arriba abajo (`tools/run_notebook.py`, log en `experiments/logs/notebook_EXP-021.log`): run `EXP-021_20261015-165309`, **idéntico** al run del barrido `EXP-021_final` (semilla 1): mejor época 82 de 107, val acc 0,6421, mean recall 0,6600, min recall 0,2941. `model.keras` y `prediction.json` (2 365 imágenes) salen ahora de este modelo; `prediction.json` se copia a `reports/delivery_02/` (el de la entrega 1 sigue en `reports/delivery_01/`). La versión del notebook de la entrega 1 queda en el commit `d557b09`.
- Antes de esta tarea, el notebook tenía un cambio sin commitear que era solo de formato (VS Code guardó las celdas como listas de líneas): sin cambios de código ni de salidas.

### CHG-020: Entrega 2 ampliada al nivel de detalle de la entrega 1 (2026-10-15)
El borrador de CHG-015 (~940 palabras, 10 frames) era anterior a la ampliación de la entrega 1 (CHG-018, 11 páginas entregadas). Se amplía con el mismo esquema, sin experimentos nuevos (todo sale de EXP-013, EXP-019..EXP-021).
- **Informe** (~2 430 palabras, ~8,5 páginas estimadas):
  - Objetivo: cómo se mide la brecha (accuracy de train en modo inferencia, sin augmentation).
  - Punto de partida: la pregunta de la entrega, ¿varianza o sesgo? Se apoya en la brecha, en la capacidad (EXP-006, 1024-512-256) y en la parada.
  - Cada técnica con su fundamento:
    - Dropout [R5; R8 §7.12].
    - L2 como weight decay y MAP gaussiano [R8 §7.1.1, §5.6.1], y por qué con Adam no equivale a weight decay [R32].
    - BatchNorm antes de la activación [R6], que permite una lr mayor [R6; R35].
    - Augmentation: el grupo D4 son permutaciones exactas sobre la rejilla [R8 §7.4].
  - Combinaciones: Dropout + BN (desplazamiento de varianza [R34], dado como posible causa) y L2 delante de BN, que no restringe la función [R33].
  - Tres cuadros de variantes con train acc y brecha: `phase1_variants` (EXP-013), `augmentation_variants` (EXP-019) y `combine_variants` (EXP-020).
  - Recall por clase: Helipad en 0 con dropout 0,5. Con rotaciones, Cargo plane, Motorboat y Dump truck mejoran y Truck empeora.
  - Resultado final: análisis del punto de parada de la semilla 1 con la figura nueva `stopping`. La primera bajada de la lr llega en la época 71 (en la entrega 1, en la 20) y la mejor época es la 82.
  - Por clases: Helipad pasa de 0,303 a 0,697 (11 imágenes: muy ruidoso). Truck baja a 0,258 y se confunde con Small car, Dump truck y Bus.
  - Bibliografía: de 6 a 10 entradas.
- **Slides**: 11 frames y 11 min de notas, en el orden del informe (DEC-010), con el esquema pregunta / qué vemos / decisión de la entrega 1. Frames nuevos: el del punto de parada; se quita el frame "L2 y combinaciones", que pasa a las notas.
- **Scripts:**
  - `build_results.py`: sección opcional `class_recalls` (macros `<clave>-recall-<clase>` y `valid-count-<clase>`) y columna `gap` en los cuadros de variantes.
  - `check_bundle.py`: error si un `\res{}` cuyo valor ya lleva `$...$` va dentro de modo matemático (ISS-010). Probado en negativo. La entrega 1 sigue con 0 errores.
- **`make bundle D=02`:** 0 errores y 2 avisos de longitud, como en la entrega 1; recortar lo decide el usuario. `review_notes.md` reescrito.

### CHG-021: Las entregas pasan a un único proyecto de Overleaf, `reports/overleaf/` (2026-10-15)
Por DEC-011. No cambia el contenido de ninguna entrega.
- **Migración** (con `git mv`, para conservar la historia):
  - `common/` estaba repetido, idéntico, en la plantilla y en las dos entregas. Queda una sola copia, en `reports/overleaf/common/`.
  - `figures/`, `generated/` y `sections/` de cada entrega pasan de `reports/delivery_NN/overleaf/` a `reports/overleaf/delivery_NN/`.
  - Los `report.tex` y `slides.tex` de cada entrega pasan a `reports/overleaf/delivery_NN_report.tex` y `delivery_NN_slides.tex`, con las dos líneas de rutas de búsqueda.
  - Las fuentes de cada entrega (`delivery.yaml`, `drafts/`, `figure_specs.yaml`, notas, `.build_state.json`) siguen en `reports/delivery_NN/`.
  - El `confusion_matrix.png` sin versionar de la entrega 1 se movió con su carpeta.
  - `ablation.tex` de la entrega 2 estaba borrado desde CHG-020 y se quita del índice.
- **Scripts:**
  - Módulo nuevo `scripts/project_paths.py`: la estructura del proyecto y la generación de los documentos principales desde la plantilla, sin sobrescribir.
  - `build_results.py`, `build_figures.py` y `draft_sections.py` escriben en `reports/overleaf/delivery_NN/`. `draft_sections.py` crea además los documentos principales que falten.
  - `check_bundle.py` resuelve las rutas como LaTeX (la raíz y después el `\input@path` del documento). Da error si un documento principal no incluye su carpeta o si hay `sections/`, `generated/` o `figures/` en la raíz.
  - `pack_bundle.py` genera `reports/overleaf_project.zip` con todo el proyecto, e indica qué subir para añadir la entrega a un proyecto existente.
  - Se borran los zips por entrega; `.gitignore` ignora `reports/*.zip`.
- **Documentación:** `reports/template/README.md`, `.claude/commands/new-delivery.md`, `Makefile` y CLAUDE.md.
- **Verificado:** primera compilación real de los documentos. Los cuatro compilan desde el zip con Tectonic 0.15 (motor XeTeX, descargado solo en el scratchpad de la sesión; Overleaf usará pdfLaTeX), sin marcas de figura, tabla ni macro ausente.
  - Informe de la entrega 1: 11 páginas, igual que el PDF entregado.
  - Informe de la entrega 2: 9 páginas.
  - Slides: 14 páginas cada una. El PDF entregado de la entrega 1 tiene 13 y no tiene el frame de referencias, así que esa edición se hizo en Overleaf y no está en el repositorio.
  - `check_bundle.py`: mismos resultados que antes de migrar en las dos entregas. Generar los documentos principales de una entrega nueva (prueba con `delivery_99`, borrada después) funciona y no sobrescribe.
- **Compilar destapó ISS-011**, resuelta aquí.

### CHG-022: Label smoothing, promedio de modelos y diapositiva de la matriz de confusión (2026-10-15)
- **`experiment_utils.py`:** clave nueva `label_smoothing` (por defecto 0). Si vale 0, la pérdida sigue siendo la cadena `'categorical_crossentropy'`, así que los runs anteriores se reproducen igual. `sweep.py`: fase `d2_label_smoothing` (EXP-022).
- **`tools/ensemble_eval.py`** (nuevo): promedia las probabilidades de los modelos guardados en validación y train, con las mismas métricas que `run_experiment`, y recalcula cada modelo por separado para comprobar que coincide con `experiments.csv`. Escribe en `experiments/ensembles/<id>.json` (EXP-023).
- **Scripts de entregas:**
  - `build_results.py`: sección opcional `ensembles` (macros `<clave>-mean-recall`, `-gain`, `-best-member-mean-recall`…).
  - `build_figures.py`: la figura `confusion` acepta `select: {variant: ...}`, que dibuja la media de las matrices de las semillas normalizadas por filas.
  - `check_bundle.py`: da error si el frame anterior a "Conclusiones…" no muestra solo `figures/confusion.png` (DEC-012).
- **Plantilla:** frame de la matriz de confusión al final de `slides_05_final.tex`, la figura al final de `report_05_final.tex`, la regla 11 del README y el `figure_specs.example.yaml` con `variant`.
- **Entrega 2:**
  - Figura `confusion` (media de las 3 semillas de EXP-021): diapositiva propia, la 13 de 16, justo antes de las conclusiones, y en el informe al final del resultado final, con un párrafo sobre los dos grupos de errores (vehículos entre sí; Building recoge Storage tank en 0,20).
  - Subsección y cuadro `ls_variants` de label smoothing (EXP-022).
  - Párrafo del promedio de semillas (EXP-023) en el resultado final.
  - Frame "Dos pruebas más" y conclusiones actualizadas. Bibliografía: + [R38], [R39].
- **Verificado:** `make bundle D=02`: 0 errores y 3 avisos de longitud (~9,7 páginas estimadas, 13 frames y 13 min de notas). Compilado con Tectonic: informe de 10 páginas y slides de 16, sin desbordes ni marcas de contenido ausente. Las diapositivas nuevas caben (revisadas como imagen). La entrega 1 sigue compilando (11 páginas).

### CHG-023: Las figuras de BatchNorm y augmentation de la entrega 2 pasan de barras a dumbbell (2026-10-16)
A petición del usuario: no le gustaba cómo comparaban visualmente las dos figuras.
- **Qué fallaba en las barras:**
  - Empezaban en 0, así que diferencias de 1-4 pp de mean recall se veían planas.
  - El train era un trazo discontinuo flotando sobre cada barra, sin unirse a ella.
  - La leyenda ocupaba un tercio del ancho y las etiquetas estaban giradas.
  - La brecha train-val, que es de lo que trata la entrega, no se leía.
- **Ahora** (tipo nuevo `dumbbell` en `build_figures.py`):
  - Una fila por variante: validación como punto relleno (media de 3 semillas) más un punto pequeño por semilla, y train como punto hueco, unidos por una línea cuya longitud es la brecha.
  - Al ser puntos, el eje no necesita empezar en 0.
  - La fila de referencia, en gris, marca una vertical punteada en su validación. La variante elegida va en negrita. Etiquetas horizontales y leyenda en una fila encima.
  - Las semillas sustituyen a la barra de error: con 3 semillas, mostrarlas es más honesto que una desviación típica.
- **Sin cifras de brecha en la figura:** se miden en mean recall y el texto cita la brecha en accuracy (p. ej. 15,1 frente a 14,0 pp para el mismo modelo), así que pondrían dos definiciones a la vez. El pie dice que la brecha es en mean recall.
- **Slide de augmentation:** la viñeta pasa de la accuracy de train al mean recall de train, que es lo que muestra la figura (regla de CHG-014).
- **Verificado:** `make bundle D=02`: 0 errores. Compilado con Tectonic: informe de 11 páginas y slides de 16, sin desbordes; la slide de augmentation revisada como imagen. En esas imágenes, "¿" sale como "£": es un artefacto de Tectonic (XeTeX con `T1` e `inputenc`). En Overleaf, con pdfLaTeX, la entrega 1 lo mostraba bien con los mismos paquetes.

### CHG-024: Capacidad × regularización, desplazamientos y TTA, en el código y en la entrega 2 (2026-10-16)
- **`experiment_utils.py`:** op de augmentation `shiftK` (desplazamiento aleatorio de hasta K píxeles con relleno por reflejo; probada sobre un array pequeño) y clave `pool` (`AveragePooling2D` delante de `Flatten`). Las dos tienen valores por defecto que no cambian los runs anteriores.
- **`sweep.py`:** fases `d2_capacity_reg` (EXP-024), `d2_shift` (EXP-025) y `d2_subpixel_shift` (EXP-027).
- **`tools/ensemble_eval.py`:** opción `--tta rot90` (EXP-026).
- **Scripts de entregas:**
  - `build_figures.py`: tipo `capacity_reg`, con dos paneles: mean recall de validación y brecha según la anchura, una línea por regularizador.
  - `build_results.py`: sección `ensemble_groups`, con la media ± desviación de varias evaluaciones de `ensembles/` (la TTA por semilla).
- **Entrega 2, como continuación lógica:**
  - **Informe:**
    - La sección de técnicas abre con el orden de las preguntas: cada técnica sola, augmentation, combinaciones, label smoothing, capacidad (¿fallaba Dropout por la red pequeña?) y desplazamientos (¿se le puede enseñar la traslación como la orientación?).
    - Subsecciones nuevas «Capacidad y regularización», con la figura `capacity_reg`, y «Desplazamientos», con el cuadro `shift_variants`.
    - La TTA va junto al promedio de semillas en el resultado final.
    - Resumen y conclusiones actualizados; la bibliografía añade [R40].
  - **Slides:** frames «¿Y con una red más grande?» y «Desplazamientos», en el mismo orden que el informe; conclusiones actualizadas.
  - **Corregida una afirmación demasiado amplia del borrador:** decía que label smoothing tampoco ayudaba con redes más anchas, pero solo se probó con 4 × 256.
- **Verificado:** `make bundle D=02`: 0 errores y 3 avisos de longitud (~11,4 páginas estimadas, 15 frames, 15 min). Compilado con Tectonic: informe de 12 páginas y slides de 18, sin marcas de contenido ausente. Se arregló un frame que se salía 14 pt de alto (figura de 0,5 a 0,42 de la altura). Queda el desborde conocido de 7 pt de `phase_summary`. Las slides nuevas, revisadas como imagen.
- **Sin cambios:** el modelo final (EXP-021), el notebook y `prediction.json`.

### CHG-025: Slides de la entrega 2 recortadas a 10 frames y 10 minutos (2026-10-16)
A petición del usuario: adaptar la presentación a 10 min con los resultados más importantes. Tenía 15 frames y 15 min tras CHG-024. El informe no cambia.
- **Sin quitar ninguna figura:** las 10 siguen en las slides, en el mismo orden que en el informe (DEC-010).
- **Fusiones:**
  - Objetivo con punto de partida.
  - Dropout con L2: un frame con las dos curvas de validación.
  - Progreso con brecha: un frame con las dos figuras.
- **Frames eliminados:** «Dos pruebas más» (label smoothing y promedio de semillas) y «Desplazamientos», con sus tablas. Sus resultados pasan a una línea de las conclusiones («No funcionan»: label smoothing −2,8 pp, desplazamientos −6,0 pp con 1 px, promediar al predecir poco o nada) y a las notas del orador. Siguen completos en el informe.
- **Orden:** objetivo y punto de partida → Dropout y L2 → BatchNorm → augmentation → capacidad × regularización → progreso y brecha → parada → resultado final → matriz de confusión (sola, justo antes de las conclusiones: DEC-012) → conclusiones.
- **Notas:** 10 min en total; 1,25 min en los frames con más contenido (Dropout y L2, capacidad, conclusiones) y 0,75 en parada, resultado final y matriz.
- **Verificado:** `make bundle D=02`: 0 errores, y ya ningún aviso de las slides; queda el de longitud del informe. Compilado con Tectonic: 13 páginas (portada, 10 de contenido, referencias y portada final). Un desborde de 5 pt en «Progreso y brecha», arreglado con figuras de 0,5 de la altura. Todas las slides revisadas en una hoja de contactos.
- La versión de 15 frames nunca llegó a commitearse. Solo queda una copia temporal en el scratchpad de esta sesión; los frames eliminados se pueden reconstruir a partir de CHG-022 y CHG-024 y de las secciones del informe.

### CHG-026: Convención de métricas única y el salto del modelo final explicado, en la entrega 2 (2026-10-16)
A petición del usuario: «me quiero mantener consistente» (sin cambiar conclusiones) y «no queda claro cómo se hace el salto de 61 a 64».
- **Convención** (regla 12 de la plantilla y memoria):
  - Las variantes se comparan por mean recall, de validación y de train.
  - La brecha y las curvas por época van en accuracy, la única métrica de train registrada por época.
  - Se dice una vez en «Objetivo y datos» y en los pies de las curvas por época.
- **Cambios en la entrega 2:**
  - `build_figures.py`: el tipo `strength` acepta `options.metric`. Las curvas de Dropout y L2 pasan a mean recall: misma historia, train y validación bajan juntos.
  - Los pies de los dumbbells hablan de «distancia en mean recall» y no de «brecha».
  - Cinco pasajes del informe que describían el nivel de train de una variante pasan a mean recall (Dropout, L2, BatchNorm, augmentation y desplazamientos); macro nueva `gain-bn-train-mr`. Las brechas siguen en accuracy.
- **Salto 0,617 → 0,646 (EXP-028):**
  - **Informe:** el resultado final abre con el mecanismo (el ruido de la validación con rotaciones disparaba la bajada de la lr), las épocas de la primera bajada por semilla y la ablación (cuadro nuevo `patience_variants`).
  - El paso 3 de la tabla de pasos se llamaba «entrenamiento más largo; sin dropout ni L2»; ahora es «la learning rate baja más tarde (paciencia de ReduceLROnPlateau 6 → 8)».
  - Resumen y conclusiones corregidos.
  - **Slides:** el frame de parada pasa a «De 0,617 a 0,646: la learning rate bajaba demasiado pronto» (1 min); las conclusiones bajan a 1 min. Siguen siendo 10 frames y 10 min.
- **Scripts:** `build_results.py`, op `first_drop` en `history_values` (primera época con la lr reducida).
- **Verificado:** `make bundle D=02`: 0 errores y 1 aviso (longitud del informe, ~12,2 páginas estimadas). Compilado con Tectonic: informe de 12 páginas y slides de 13, sin desbordes nuevos ni marcas de contenido ausente.

### CHG-027: La brecha de las redes anchas, interpretada como ruido (2026-10-17)
El usuario preguntó si 4 × 1024 no era mejor modelo, al tener la misma validación y menos brecha (8,8 frente a 12,1 pp).
- **No lo es.** La brecha es un diagnóstico, no el objetivo: con la misma validación, una brecha menor solo significa que se ajusta menos el train. Además, aquí la diferencia es ruido: la de 4 × 256 se debe sobre todo a la semilla 1 (17,2 pp; las otras dos, 9,6 y 9,5 pp) y depende de la época conservada. La elección de 4 × 256 se justifica por ser el modelo más simple con la validación empatada.
- **Entrega 2:**
  - Informe: la sección de capacidad deja de decir «la red más ancha no memoriza más» y explica que la diferencia de brecha es ruido, con las brechas por semilla (variantes de una semilla `final-s1..3` en `delivery.yaml`). La decisión se argumenta por simplicidad (400 653 frente a 3 961 869 parámetros).
  - Slide de capacidad: «empatan en validación: se elige la más simple».
- **Bitácora:** corrección añadida a EXP-024.

### CHG-028: Infraestructura para las CNN de la entrega 3 (2026-10-17)
Fase 0 del plan (DEC-017). Ningún resultado anterior cambia: todas las claves nuevas tienen un valor por defecto que mantiene el comportamiento de las entregas 1-2.
- **`experiment_utils.py`:**
  - Clave `arch`: `'mlp'` (por defecto, el código anterior pasa intacto a `_build_mlp`), `'cnn'` o una arquitectura conocida.
  - Claves de la CNN propia: `conv_blocks`, `convs_per_block`, `kernel`, `downsample` (`max`, `avg` o `stride`), `head` (`gap` o `flatten`), `head_dense` y `head_dropout`. La salida es `Dense(13)` en float32.
  - Pérdida `logit_adjust` (`la_tau`): clase `LogitAdjustedCE` [R51], serializable. Con una prior uniforme da exactamente ln 13, igual que la entropía cruzada.
  - `precision` (`float32` o `mixed_float16`) y `eval_batch_size`.
  - Augmentation nueva: `rcrop` (recorte del 80-100 % del lado, reescalado), `jitter` (brillo, contraste, saturación y ganancia por canal, suaves) y `erase` (Random Erasing [R47]: p 0,5, 2-10 % del área). Las traslaciones ya existían (`shiftK`: relleno por reflejo y recorte).
  - Mezcla en el lote: `mix` = `mixup` [R49] o `cutmix` [R50] (`mix_alpha`). Mezcla imágenes, etiquetas y pesos de muestra.
- **`architectures.py`** (nuevo):
  - `custom_cnn`: bloques [Conv → BN → activación] × n + reducción espacial; cabeza GAP [R45] o Flatten.
  - `known`: VGG16, ResNet50, InceptionV3 y MobileNetV2 de `keras.applications`, con `weights=None` y `pooling='avg'` (DEC-015).
- **`sweep.py`:** `run_isolated`, que lanza cada run en un proceso nuevo para que la memoria de la GPU se libere entera entre runs; fase `d3_conv` (EXP-029).
- **Verificado:**
  - **Regresión:** con el código nuevo, la config de `EXP-021_final` (semilla 1) da un historial idéntico, época a época, en las 107 épocas (pérdida, accuracy, métricas de validación y lr).
  - Todas las arquitecturas se construyen y dan salidas (n, 13) que suman 1. Las augmentations nuevas conservan forma y rango, y MixUp y CutMix conservan la suma de las etiquetas.
- **Prueba de humo** (3 épocas; s/época de la 2.ª en adelante; pico de memoria de GPU):

| Modelo (entrada) | float32 | mixed_float16 | Pico (float32) | Observación |
|---|---|---|---|---|
| CNN propia (64) | 13,8 s | 10,7 s (×1,29) | 0,4 GB | dos runs iguales → historial idéntico |
| CNN propia (112) | 39,3 s | 33,9 s (×1,16) | 1,3 GB | |
| VGG16 (112) | 66,7 s | 38,2 s (×1,75) | 2,4 GB | en mixed **no entrena**: pérdida 2,32 ≈ entropía de clases, val en el azar |
| ResNet50 (112) | 83,9 s | 75,9 s (×1,10) | 2,6 GB | |
| InceptionV3 (112) | 70,2 s | 60,9 s (×1,15) | 2,5 GB | |
| MobileNetV2 (112) | sin memoria | 56,5 s | 3,1 GB | en mixed se queda en el azar |

- **Decisiones tomadas con estos datos:**
  - **Se trabaja en float32.** La precisión mixta solo pasa de 1,5× en VGG16, y ahí rompe el entrenamiento.
  - **El determinismo se mantiene**, aunque cuesta un ~25 %. Sin él, ResNet50 baja de 83,8 a 60,1 s/época y la CNN de 64, de 14,8 a 11,4. XLA no lo compensa: con `enable_op_determinism`, Keras desactiva `jit_compile` («Model doesn't support jit_compile=True»).
  - **La carga de datos no es el cuello de botella:** 0,4 s por época a 64 y 0,8 s a 112, con rotaciones.
  - **MobileNetV2 sale de la comparación:** era opcional, se quedó sin memoria en float32 tras 10 runs en el mismo proceso y no entrenó en mixed. El agotamiento de memoria tras muchos runs motiva `run_isolated`.
- **Presupuesto estimado** (con unas 80-90 épocas por run, en float32 determinista):
  - CNN propia a 64: unos 20 min por run.
  - CNN propia a 112: unos 55 min por run.
  - Arquitecturas conocidas: 1,5-2 h por run.
  - Se ajusta el plan: la criba de lr de las conocidas se hará con un tope de épocas corto.

---

## Incidencias (ISS)

### ISS-001: El ReLU tras `Flatten` no hace nada · resuelta (→ CHG-008)
`Activation('relu')` se aplica directamente a los píxeles, que son `uint8` en [0, 255] (verificado en 200 imágenes de train). ReLU(x) = x para x ≥ 0, así que no hay ninguna no linealidad y el modelo es lineal (EXP-000). Probablemente faltaba una `Dense` antes del ReLU. **Resolución:** IDEA-002.

### ISS-002: Los callbacks no llegan a actuar con 20 épocas · resuelta (→ EXP-003)
`EarlyStopping(patience=40)` no puede dispararse con `epochs=20`. `ReduceLROnPlateau(patience=10)` puede actuar como mucho una vez (en la época 11 o más tarde). Además, ambos vigilan `val_accuracy`, que es una señal ruidosa y escalonada. Vigilar `val_loss` es más suave. **Confirmada en EXP-000:** la lr fue 1e-3 las 20 épocas. La mejor val acc llegó en la época 17, así que ReduceLROnPlateau no tuvo margen para actuar, y EarlyStopping no podía. **Resolución:** IDEA-004.

### ISS-003: No hay GPU · resuelta (→ CHG-006: WSL2 + TF 2.21 con CUDA)
`tf.config.list_physical_devices('GPU')` devuelve `[]`. TensorFlow 2.10 fue la última versión con soporte de GPU en Windows nativo [R13], y el kernel usa TF 2.20, así que todo se entrena en CPU aunque CUDA 12.5 esté instalado. Esto condiciona el coste de cada experimento. **Opciones:** Google Colab (el notebook viene de Colab según sus metadatos), WSL2 con `tensorflow[and-cuda]`, o reducir el coste por época (IDEA-006). **Resolución:** Keras 3 sobre PyTorch con CUDA (CHG-003, DEC-004). El `torch 2.3.1+cu118` del Python global sí veía la GPU, pero falla al importar con su `numpy 2.2.6` ("A module that was compiled using NumPy 1.x cannot be run in NumPy 2.2.6"), por eso se usa un entorno aparte. **Reabierta** porque CHG-004 restaura la celda 1 original, que ya no activa el backend torch. Para volver a la GPU hay que fijar `os.environ['KERAS_BACKEND'] = 'torch'` antes del primer `import keras` y usar el kernel del `.venv`.

### ISS-004: Comentario engañoso sobre "resampling" · resuelta (→ CHG-010)
En la celda que construye `anns` aparece `# Resampling strategy to reduce training time`, pero no hay ningún remuestreo implementado. Hay que eliminarlo o implementar lo que promete (relacionado con IDEA-005).

### ISS-005: La carga de imágenes domina el tiempo de entrenamiento · resuelta (→ CHG-007)
El generador abre con rasterio cada TIF de 224×224 en cada batch y en cada época, en el mismo hilo que el entrenamiento. Medido (CHG-003): 63 ms por batch de 16 con la caché de disco caliente y 232 ms en frío, frente a 13,8 ms del paso de entrenamiento en GPU. Por la ley de Amdahl [R16], acelerar solo el cálculo da como mucho (63 + 26,4) / (63 + 13,8) ≈ ×1,16 por paso con la caché caliente, aunque la GPU haga el cálculo ×1,9 más rápido. En la prueba de CHG-003, con lecturas en frío, el paso completo fue de unos 240 ms. Mientras no se resuelva, la GPU apenas acorta las épocas de esta ffNN. Con modelos más pesados (un MLP grande o una CNN) el cálculo pesará más y la GPU se notará más, pero la carga seguirá costando unos 63 ms por paso. **Medido en EXP-000** (CPU, TF 2.20, 1,5 GB de RAM libre): de 105 a 422 ms por paso según la época, y 70 min en total. Las épocas varían mucho (111–446 s) sin ningún cambio en el cálculo, lo que delata que el tiempo lo marca la E/S y la caché de disco. **Resolución:** IDEA-009.

### ISS-006: Archivos `:Zone.Identifier` en la copia del dataset · resuelta (→ CHG-007)
Al copiar el dataset desde Windows a WSL, cada TIF llegó acompañado de un `<nombre>.tif:Zone.Identifier` (el flujo alternativo "Mark of the Web" de NTFS convertido en archivo): 21 112 archivos de más. La celda de test del notebook recorre `xview_test/` con `os.walk` sin filtrar la extensión, así que habría contado 4 730 imágenes de test y fallado al abrir los `Zone.Identifier` con rasterio. **Resolución:** el usuario pidió borrarlos (`find xview_recognition -name '*:Zone.Identifier' -delete`; quedan 18 746 + 2 365 TIF y el JSON) y, además, `xview_data.list_test_files()` filtra por `*.tif`. El mismo filtro se aplica en la celda de test del notebook desde CHG-010.

---

### ISS-007: La `val_loss` de `EpochMetrics` no es comparable con la pérdida de train cuando la softmax se satura · abierta (menor)
`experiment_utils.classification_metrics` recorta las probabilidades a 1e-7 antes del logaritmo, así que la `val_loss` de cada época no puede superar −ln(1e-7) ≈ 16,1. La pérdida de train la calcula Keras sin ese tope (desde los logits). En EXP-001 la pérdida de train es ~3 300 y la de validación ~10: la diferencia es del cálculo, no del modelo. Con la softmax no saturada (desde EXP-002) el efecto es pequeño, y ninguna decisión usó `val_loss` salvo EXP-003_val_loss (vigilar la pérdida), cuyo efecto no se puede separar. **Detectado** al dibujar la figura de normalización de la entrega 1, que por eso muestra solo la pérdida de train. **Resolución propuesta:** calcular la `val_loss` con `keras.losses.categorical_crossentropy` sobre las mismas predicciones (afecta solo a runs futuros).

### ISS-008: EXP-005 describía como permanente el colapso a 224×224 · resuelta (→ corrección en EXP-005)
EXP-005 decía que a 224×224 "la pérdida de train se queda en ≈ 2,28 desde la época 2 y la accuracy en ≈ 0,21". El historial (`history/EXP-005_r224.json`) muestra que eso dura unas 11 épocas: después la pérdida baja a 2,03 y la train acc llega a 0,315 (el valor de `experiments.csv`). **Detectado** al redactar la fase 2 de la entrega 1 (CHG-018), que ahora habla de "las diez primeras épocas" con la mediana de esas épocas. La conclusión (problema de optimización, no de varianza) no cambia.

### ISS-009: La tabla de combinaciones de la entrega 2 decía "Quitando" · resuelta (→ CHG-020)
La tabla de EXP-020 en la entrega 2 se generaba con `write_ablation`, pensada para la ablación de la entrega 1. Su cabecera de columna es "Quitando", pero la mayoría de sus filas *añaden* técnicas ("+ Dropout 0,1"). Además, no mostraba la accuracy de train ni la brecha, que son lo que se discute en esta entrega. Se sustituye por el cuadro de variantes `combine_variants`. `write_ablation` no cambia.

### ISS-010: `check_bundle.py` no detectaba `\res{}` con `$` dentro de modo matemático · resuelta (→ CHG-020)
Algunas macros ya traen su propio modo matemático, como la lr (`$3{,}6\cdot 10^{-3}$`) o el factor L2. Escribirlas dentro de `$...$` anida los `$` y rompe la compilación en pdfLaTeX, y la comprobación no lo veía. Apareció al redactar la entrega 2 (`$\lambda = \res{l2-1e-5-l2}$`). También se evitan los números con coma decimal dentro de `$...$` (`$p = 0,5$`), porque en modo matemático la coma añade un espacio.

### ISS-011: Las tablas de variantes de la entrega 2 se salían del margen · resuelta (→ CHG-021)
Se vio al compilar por primera vez. `phase1_variants`, `augmentation_variants` y `combine_variants` tienen 7 columnas y se salían del margen derecho entre 55 y 71 pt, incluso con `\scriptsize`; `baseline_vs_final` se salía 27 pt. Se escalan al ancho del texto con `\resizebox{\linewidth}{!}{...}` en los borradores. Queda un desborde de 7 pt en `phase_summary`, inapreciable. `check_bundle.py` no mide anchos: solo se ven compilando.

### ISS-012: EXP-021 cambiaba dos factores a la vez y su ganancia se atribuía a «entrenar más» · resuelta (→ EXP-028, CHG-026)
EXP-021 cambió las paciencias de EarlyStopping (15 → 25) y de ReduceLROnPlateau (6 → 8) a la vez. Eso rompe la regla de un factor por experimento, y la bitácora, el informe y las slides atribuían la ganancia a un entrenamiento más largo. El usuario preguntó de dónde salía el salto. La ablación (EXP-028) lo atribuye entero a la paciencia de ReduceLROnPlateau.

### ISS-013: Un reinicio de WSL cortó `EXP-039_inception_v3_seed3` a mitad de entrenamiento · resuelta (→ relanzado, EXP-039)
- **Fecha:** 2026-10-19. El run empezó a las 16:51 y su último checkpoint es de las 17:32. WSL se reinició hacia las 17:50 (los procesos del sistema arrancan a las 17:51), por causa desconocida, y el proceso de `sweep.py d3_known` murió sin escribir su fila en `experiments.csv` ni su history.
- **Impacto:** solo ese run. Como `run_isolated` lanza cada run en su propio proceso y `sweep.py` se salta los ids que ya están en el CSV, los 8 runs anteriores quedaron intactos.
- **Solución:** relanzar `sweep.py d3_known` (log en `experiments/logs/d3_known_resume.log`). Se saltó los 8 runs hechos y entrenó el que faltaba desde cero, sobrescribiendo el checkpoint a medias; terminó a las 20:15 (105 épocas, 8 375 s). No se reanuda desde el checkpoint a mitad de run: el estado del optimizador y de los callbacks no se guarda, y reanudar cambiaría el run respecto a sus semillas hermanas.

### ISS-014: `EXP-039_vgg16_seed3` colapsa a una sola clase con lr 1e-3 · abierta
- **Fecha:** 2026-10-19. Con la misma lr y receta que las semillas 1 y 2 (0,756 / 0,760), la semilla 3 no aprende: la pérdida de train se queda plana en 2,314 desde la época 2, prácticamente la entropía de las clases (2,31), y la red predice siempre Excavator (val acc 0,042, mean recall 0,077 = 1/13). El early stopping la cortó en la época 26 con la mejor en la 1.
- **Causa probable:** el mismo colapso que EXP-038 vio con lr 3e-4 y 3e-3: VGG16 sin BatchNorm solo entrena en una ventana estrecha de lr, y 1e-3 está en el borde de esa ventana, de modo que depende de la inicialización. Es el problema de entrenar VGG profundas desde cero sin normalización [R41], que BN resuelve [R6].
- **Impacto:** VGG16 queda con 2 semillas válidas en EXP-039. Se cumple la condición que EXP-038 fijó para el plan B (añadir la variante con BN).
- **Pendiente de decisión del usuario:** relanzar la semilla 3 (~1-1,5 h; no arregla la fragilidad), aplicar el plan B con BN, o dejar VGG16 con 2 semillas y el colapso documentado.

---

## Ideas pendientes (backlog)

Prioridad = impacto esperado / coste. Al probar una idea se crea su EXP y la idea pasa a *probada → EXP-###*.

### IDEA-001: Normalizar la entrada · probada → EXP-002, EXP-004
- **Qué:** escalar los píxeles a [0, 1], o estandarizar por canal con la media y la desviación típica de train.
- **Por qué:** (1) si todas las entradas son positivas, los gradientes de los pesos que llegan a una neurona tienen todos el mismo signo y el descenso avanza en zigzag; centrar en cero mejora el condicionamiento [R1 §4.3]. Adam reescala cada parámetro (precondicionador diagonal), lo que compensa la escala pero no la correlación que introduce una media no nula. (2) Con `glorot_uniform` y 150 528 entradas, la desviación típica inicial de los logits es ≈ 110 con píxeles crudos frente a ≈ 1,4 con entradas estandarizadas (estimación con las estadísticas reales: media ≈ 62, desviación ≈ 47,5). Con logits ≈ 10² la softmax está saturada desde el principio.
- **Éxito:** la pérdida inicial ≈ ln 13 ≈ 2,56 y una convergencia más rápida que en EXP-000.
- **Evidencia de EXP-000:** pérdida de ~3 000 durante las 20 épocas, val acc oscilando ±5 pp y dos clases frecuentes que nunca se predicen. Es lo esperable con logits saturados (análisis en la conclusión de EXP-000). Es el cambio de mayor impacto esperado. Además de lo anterior, comprobar que la val acc deja de oscilar (σ entre épocas ≪ 0,048).

### IDEA-002: Añadir capas ocultas reales (MLP) e inicialización He · probada → EXP-005, EXP-006
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

### IDEA-005: Tratar el desbalance de clases · probada → EXP-014
- **Qué:** `class_weight` inversamente proporcional a la frecuencia (o según el "effective number" [R11]), u oversampling de las clases minoritarias en el generador.
- **Por qué:** el desbalance es de 32:1 (Building 3 594 frente a Helipad 111). Sin corregirlo, la pérdida está dominada por las clases frecuentes. En CNNs, el oversampling suele ser la estrategia más robusta [R10]. La métrica guía debe ser el mean recall (balanced accuracy), no la accuracy.
- **Evidencia de EXP-000:** Helipad (minoritaria) tiene recall 0, pero también Truck, que es la 3.ª clase más frecuente. No todo es desbalance: aplicar esta idea después de IDEA-001 para no confundir los dos efectos.

### IDEA-006: Reducir la resolución de entrada para la ffNN · probada → EXP-005
- **Qué:** redimensionar a 64×64 o 32×32 antes de `Flatten`.
- **Por qué:** divide la dimensión de entrada por ≈ 12 o ≈ 49, y con ella los parámetros de la primera capa: menos varianza (mejor relación datos/parámetros) y menos cálculo por paso. Esto último importa poco ahora que hay GPU (CHG-003) y que la carga de imágenes domina el tiempo por paso (ISS-005), porque el redimensionado se hace después de leer cada imagen. **Coste:** se pierde detalle en objetos pequeños (Small car, Pylon).

### IDEA-007: Regularización (dropout, L2, BatchNorm) · probada → EXP-013
- **Por qué:** cuando el MLP tenga capacidad de sobreajustar (hueco entre `train_accuracy` y `val_accuracy` en `runs.csv`). Dropout [R5], penalización L2 [R8 §7.1], BatchNorm para estabilizar y acelerar el entrenamiento [R6].

### IDEA-008: Data augmentation con rotaciones de 90° y flips · probada → EXP-015
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

### IDEA-011: Entrenar el modelo final con batch pequeño · probada → EXP-016 (refutada)
- **Qué:** reentrenar la configuración final con batch 16 (lr reescalada por la regla de la raíz) y 3 semillas, y compararla con batch 256.
- **Por qué:** EXP-007 con batch 16 da 0,587 ± 0,004 frente a 0,578 ± 0,007 con batch 256 (EXP-012). Puede ser el efecto regularizador del ruido de los batches pequeños [R30]. Con regularización explícita (fase 6) o augmentation (fase 8) la diferencia podría desaparecer.
- **Coste:** ~8 min por run con batch 16. **Éxito:** > 1 pp por encima del batch 256 con 3 semillas.

### IDEA-012: Test-time augmentation con las 4 rotaciones de 90° · probada → EXP-026 (sin mejora)
- **Qué:** promediar las predicciones de cada imagen y de sus 3 rotaciones, con el modelo de EXP-021 y, combinado con EXP-023, con los 3 modelos.
- **Por qué:** el modelo se entrenó con rotaciones, y promediar sobre ellas impone también en la predicción la invarianza que se le enseñó. Krizhevsky et al. 2012 promediaban la predicción de 10 recortes de cada imagen de test [R40].
- **Coste/riesgo:** no hay que entrenar; unos segundos de inferencia. **Éxito:** más de +1 pp de mean recall sobre el mismo modelo sin TTA.

### IDEA-013: Traslaciones aleatorias de ±1-2 píxeles como augmentation · probada → EXP-025, EXP-027 (refutada)
- **Qué:** una transformación nueva en `augment_batch` que desplaza la imagen de 16×16 (con reflejo en el borde), sobre EXP-021, con 3 semillas.
- **Por qué:** es la invarianza que más le falta a una red densa [R8 §9.2], y es la única propuesta que ataca su sesgo y no su varianza. Data augmentation: [R8 §7.4].
- **Coste/riesgo:** ~2 min por run en GPU. Los objetos están centrados en el recorte, así que un desplazamiento grande puede sacar parte del objeto. **Éxito:** más de +1,5 pp de mean recall con 3 semillas.

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

### DEC-007: Estilo de las figuras de las entregas (2026-10-03)
**Decisión del usuario:** las figuras del informe y de las slides usan el **mismo estilo que las de los experimentos** (`plot_theme.py`), con **textos en inglés** y en **PNG** (200 dpi), y se insertan así en LaTeX. Sustituye a lo que pedía el plan de entregas (etiquetas en español, estilo propio por contexto informe/slides y PDF vectorial), que se llegó a implementar y se descartó. Consecuencias: un solo PNG por figura para los dos documentos (cambia solo el ancho de inclusión); los nombres de clase se dejan en inglés también en tablas y texto, para que coincidan con las figuras; los pies de figura y el texto siguen en español.

### DEC-008: Alcance de la entrega 1: fases 0-5 (2026-10-03)
**Decisión del usuario.** El enunciado de la primera entrega pide diseñar y entrenar una ffNN eligiendo arquitectura, algoritmo de optimización y parámetros de entrenamiento, con un conjunto de validación para ajustar los hiperparámetros y decidir el punto de parada. Entran las fases 0-5 (baseline; normalización, callbacks y parada; resolución; arquitectura; activación; optimizador y batch). Quedan fuera las fases 6 (regularización), 7 (desbalance), 8 (augmentation) y 9 (ablación y EXP-018): sus experimentos y entradas de la bitácora no se tocan y servirán para una entrega posterior. El modelo final de la entrega es `EXP-010_bs256_sqrt_60ep` (0,578 ± 0,007 de mean recall); EXP-007 con batch 16 (0,587 ± 0,004) se cita como referencia del coste del batch grande. El notebook entregable se alinea con este modelo (CHG-013).
**Alternativas descartadas:** fases 0-6 (incluir la regularización y BatchNorm, final EXP-013, 0,590 ± 0,009) y mantener todo con las fases 7-9 como extras.

### DEC-009: Calendario de entregas del curso y reparto del trabajo ya hecho (2026-10-03)
El usuario aporta el enunciado de las cinco entregas: (1) ffNN: arquitectura, optimizador y parámetros de entrenamiento, con la validación para ajustar hiperparámetros y decidir la parada; (2) regularización de esa ffNN (penalizaciones de norma, Dropout, BatchNorm, data augmentation; sin CNN ni transfer learning); (3) CNN propias desde cero frente a arquitecturas conocidas (GoogLeNet, VGG, ResNet…) desde cero; (4) transfer learning con esas arquitecturas frente a la entrega 3; (5) experimentos pendientes y modelo final refinado.
**Reparto del plan de 9 fases (DEC-006):** fases 0-5 → entrega 1 (DEC-008); fase 6 (dropout, L2, BatchNorm) y fase 8 (augmentation) → entrega 2, con la ablación de la fase 9 como apoyo; fase 7 (desbalance de clases) e IDEA-010 → entrega 5.
**Entrega 2 (decidido por el usuario siguiendo la recomendación):** EXP-015 (augmentation) se midió sobre el modelo con pesos de clase (EXP-014); para presentar la augmentation solo como regularización se repite sobre el ganador de la fase 6 (EXP-013, sin pesos de clase): EXP-019 en adelante. **El desbalance de clases (fase 7) pasa a la entrega 5.**
**Consecuencia para la entrega 3:** la entrada de 16×16 (EXP-005) es una decisión propia de la ffNN; las CNN eligen su resolución de nuevo (la caché de 224×224 ya existe).
**Actualización (2026-10-17, el usuario):** el tratamiento del desbalance de clases no está reservado a la entrega 5: «se puede usar normal». Pasa a ser una fase de la entrega 3, sobre la CNN (D3-5 del plan de DEC-017). Ese mismo día, una sesión paralela lo había probado sobre la ffNN de la entrega 2 (EXP-029/DEC-014, sin commitear); el usuario pidió deshacer esos cambios y se borraron con sus runs, así que sus resultados no se citan y los IDs se reutilizan.

### DEC-010: El informe y las slides cuentan la misma historia (2026-10-03)
**Decisión del usuario.** Las slides se presentan guiándose por el informe: los dos documentos usan **las mismas figuras, en el mismo orden**. En el informe, para optimizar el espacio, dos figuras **relacionadas** pueden compartir fila (máximo 2, cada una con su pie y su número); figuras sin relación nunca van juntas. Cada figura se cita en el texto. `check_bundle.py` lo comprueba (CHG-016).
*Actualización (CHG-018, por las ediciones del usuario en el informe):* el análisis de la parada va en el resultado final, no en la fase 1, así que la normalización va sola. Las filas compartidas quedan en resolución + capacidad (fases 2-3). Orden común: class_distribution, normalisation, resolution_sweep, capacity, activation, lr_sweep, progress, stopping, per_class_recall.

### DEC-011: Un solo proyecto de Overleaf para todas las entregas (2026-10-15)
**Decisión del usuario.** No quiere crear un proyecto de Overleaf por entrega, sino tener todo en uno, organizado y con cada entrega separada.
- **Alternativas:**
  - (a) Cada documento principal dentro de la carpeta de su entrega. Overleaf no lo admite: el documento principal tiene que estar en la raíz [R36].
  - (b) Un documento principal en la raíz que cargue el de la subcarpeta con el paquete `import` [R36].
  - (c) **Elegida:** los documentos principales en la raíz (`delivery_NN_report.tex`, `delivery_NN_slides.tex`), y cada uno añade `delivery_NN/` a las rutas de búsqueda de LaTeX: `\input@path` para `\input` e `\IfFileExists`, y `\graphicspath` para las imágenes.
- **Por qué (c):** las secciones no cambian (siguen usando `figures/...` y `generated/...`), `common/` queda en una sola copia y cada documento solo carga las macros `\res{}` de su entrega, así que las claves repetidas entre entregas no chocan. `\graphicspath` hace falta porque en Beamer `\includegraphics` no encontraba las figuras solo con `\input@path`; se vio al compilar.
- **Restricción:** no puede haber carpetas `sections/`, `generated/` ni `figures/` en la raíz del proyecto, porque LaTeX busca primero allí.
- **Subir una entrega nueva:** con el botón Upload del proyecto existente se sube la carpeta `delivery_NN/` y los dos `.tex` principales. Overleaf conserva la estructura de las carpetas subidas [R37].

### DEC-012: La matriz de confusión del modelo final, siempre en su propia diapositiva antes de las conclusiones (2026-10-15)
**Regla del usuario, para todas las entregas.** En las slides, la matriz de confusión del modelo final va sola en el frame que precede a "Conclusiones…". Por DEC-010, la misma figura va también en el informe, al final del resultado final. Se usa la media de las matrices de las semillas normalizadas por filas: es la que cita el texto (`final-conf-*`). `check_bundle.py` da error si falta, y la plantilla ya trae el frame (CHG-022).

### DEC-013: Prueba de dos factores, capacidad × regularización, sobre el modelo de la entrega 2 (2026-10-16)
**Decisión del usuario**, tras revisar el método de un factor cada vez: probar la regularización y el Dropout variando la arquitectura, tomar la decisión más adecuada y seguir experimentando.
- **Motivo:** en la entrega 2, Dropout y L2 solo se juzgaron sobre 4 × 256. Esa red se eligió en la entrega 1 (fase 3) en parte porque las más grandes memorizaban, justo lo que la regularización debe corregir. Dropout está pensado para redes con capacidad de sobra [R5], así que la conclusión «Dropout no ayuda» solo valía para esa red.
- **Por qué dos factores:** lo que se mide es la interacción. Cambiando un solo factor cada vez, una técnica que solo funciona combinada con otra se infravalora; ya pasó con BatchNorm (+1,3 pp solo, −4,8 pp al quitarlo con augmentation: EXP-013, EXP-020). Es una excepción deliberada a la regla de un factor por experimento. Cada celda de la rejilla sigue comparándose con su base de la misma anchura y sin regularización.
- **Diseño (EXP-024):** sobre `EXP-021_final`, anchura de las 4 capas ocultas ∈ {256, 512, 1024} × {sin regularización, Dropout 0,2, Dropout 0,5, L2 1e-4}, con 3 semillas. Se mantiene todo lo demás: BN + rotaciones de 90°, lr 3,6e-3 y paciencias 25/8.
- **Criterio:** se adopta una celda si supera a EXP-021 en más que el ruido entre semillas (~1,5 pp de mean recall) sin perder accuracy. Si no, se mantiene 4 × 256 y la conclusión de la entrega se refuerza: la regularización tampoco ayuda con más capacidad.

### DEC-014: El notebook pasa a llamarse `xview_recognition.ipynb` (2026-10-17)
**Decisión del usuario** al empezar la entrega 3: «no sé si ffNN_example sea ya el mejor nombre… quiero que se vea la continuidad del trabajo y que luego todo sea reproducible».
- **Elegida:** `git mv ffNN_example.ipynb xview_recognition.ipynb`. Un solo notebook que evoluciona entrega a entrega; `git log --follow` conserva su historia, y cada entrega deja su versión exacta en una etiqueta (`delivery-01`, `delivery-02` y, al cerrar, `delivery-03`). Su tabla de progreso cubre todas las entregas.
- **Descartadas:** un notebook nuevo para las CNN, que dejaría `ffNN_example.ipynb` congelado, con la continuidad repartida entre dos archivos; y mantener el nombre, que ya no describe el contenido.
- Se actualizan las referencias que describen el estado actual (CLAUDE.md, la cabecera de esta bitácora, `tools/run_notebook.py` y el criterio de `build_figures.py`). Las entradas antiguas de la bitácora y las notas de las entregas 1 y 2 conservan el nombre antiguo, porque así se llamaba entonces.

### DEC-015: Las arquitecturas conocidas de la entrega 3 salen de `keras.applications`, desde cero (2026-10-17)
- **Elegidas:** VGG16 [R41], ResNet50 [R42] e InceptionV3 [R38], con `weights=None`. MobileNetV2 [R44] entra como referencia ligera.
- **Por qué estas:** son las mismas que en la entrega 4 tendrán pesos de ImageNet, así que «desde cero frente a preentrenada» será una comparación directa, con la misma arquitectura.
- **GoogLeNet:** su versión original, Inception v1 [R43], no existe en Keras. La familia la representa InceptionV3 y se dice así.
- **Cabeza común:** `include_top=False`, `pooling='avg'` → `Dense(13)`. El clasificador original de VGG16 (Flatten → 4096 → 4096, 50 M de parámetros a 112 px) está pensado para las 1 000 clases de ImageNet, y con una cabeza GAP la comparación entre arquitecturas solo cambia el extractor de características [R45].
- **Descartadas:** escribir a mano ResNet18 y GoogLeNet v1, más pequeñas y más cercanas a los artículos, pero sin equivalente preentrenado en Keras para la entrega 4.
- **Plan B:** si VGG16 sin BatchNorm no entrena desde cero, se documenta y se añade una variante con BN.

### DEC-016: Resolución de entrada de las CNN (2026-10-17)
- La entrada de 16×16 se eligió para la ffNN (EXP-005). Una CNN comparte pesos entre posiciones, así que más píxeles no le multiplican los parámetros como a la red densa, y debería aprovechar mejor el detalle.
- **Las CNN propias** barren {16, 32, 64, 112}; a 16 se compara a igual entrada con la ffNN.
- **Las arquitecturas conocidas** se entrenan a 112: InceptionV3 necesita al menos 75 px, la caché de 112 existe (705 MB en RAM) y cabe en la GPU de 6 GB.
- **224 queda fuera** por coste: unas 4 veces más cómputo, y su caché (2,8 GB, leída con memmap) no cabe holgada en los ~4 GB de RAM libres.
- La CNN propia final también se mide a 112, para comparar en igualdad de condiciones.

### DEC-017: Plan de la entrega 3 (2026-10-17)
**Plan aprobado por el usuario**, con un presupuesto de varios días de GPU y 3 semillas en todo lo que decide.
- **Método:** el de las entregas anteriores. Cada fase parte del ganador de la anterior y cambia un solo factor. Las decisiones del orden del ruido se toman con 3 semillas. Las interacciones esperadas se miden con rejillas pequeñas de dos factores (DEC-013).
- **Fases de la CNN propia:**
  - D3-1: la convolución a igual entrada (16×16) frente a la ffNN EXP-021.
  - D3-2: resolución.
  - D3-3: optimización (lr y batch).
  - D3-4: capacidad y reducción espacial (profundidad × anchura; MaxPool, AveragePool o stride 2 [R46]; cabeza GAP [R45] o Flatten).
  - D3-5: desbalance (pesos con raíz e inversos, sobremuestreo, logit adjustment [R51]).
  - D3-6: regularización revisada, con una criba de 1 semilla y después las mejores con 3: rotaciones, D4, desplazamientos y recorte con escala; brillo, contraste, saturación y ganancia por canal; Random Erasing [R47]; Cutout [R48]; MixUp [R49]; CutMix [R50]; Dropout en la cabeza; weight decay.
  - D3-7: CNN propia final.
- **Arquitecturas conocidas** (DEC-015): criba de lr con 1 semilla y después 3 semillas, con la misma receta que la CNN propia. MixUp y CutMix se prueban también sobre ellas.
- **D3-8: comparación final.** Se elige el modelo que entrena `xview_recognition.ipynb`.
- **Reproducibilidad:** se mantiene `enable_op_determinism`. La precisión mixta se adopta solo si acelera al menos 1,5× y conserva el determinismo (prueba de humo, CHG-028).

### DEC-018: La rejilla profundidad × resolución se adelanta a la optimización (2026-10-17)
- **Motivo:** EXP-030 muestra que, con 3 bloques, la resolución alta da subajuste, y el campo receptivo explica por qué: resolución y profundidad interactúan. Elegir la resolución con una sola profundidad fijaría 32 px por una limitación de la red, no de los datos. Además, la comparación en igualdad con las arquitecturas conocidas (a 112, DEC-016) necesita una CNN propia que funcione a 112.
- **Cambio en el plan de DEC-017:**
  - La fase D3-3 pasa a ser la rejilla de profundidad {3, 4, 5} bloques (filtros 32 → … → 512, duplicándose) × resolución {32, 64, 112}, con 3 semillas (EXP-031), al estilo de DEC-013.
  - La optimización (lr y batch) va después, sobre la celda ganadora.
  - Anchura, tipo de reducción espacial y cabeza siguen en la fase de capacidad.
- **Coste:** unas 11 h de GPU (18 runs nuevos, cada uno en su propio proceso).

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
- **[R21]** Maas, Hannun, Ng (2013). *Rectifier Nonlinearities Improve Neural Network Acoustic Models*. ICML Workshop on Deep Learning for Audio, Speech and Language Processing. (LeakyReLU)
- **[R22]** Clevert, Unterthiner, Hochreiter (2016). *Fast and Accurate Deep Network Learning by Exponential Linear Units (ELUs)*. ICLR.
- **[R23]** Hendrycks, Gimpel (2016). *Gaussian Error Linear Units (GELUs)*. arXiv:1606.08415.
- **[R24]** Klambauer, Unterthiner, Mayr, Hochreiter (2017). *Self-Normalizing Neural Networks*. NeurIPS. (SELU, `lecun_normal`, AlphaDropout)
- **[R25]** Glorot, Bengio (2010). *Understanding the difficulty of training deep feedforward neural networks*. AISTATS.
- **[R26]** Bengio (2012). *Practical Recommendations for Gradient-Based Training of Deep Architectures*. En *Neural Networks: Tricks of the Trade*, 2.ª ed., Springer. (§3.1.1: hiperparámetros de la red; la lr inicial como el hiperparámetro más importante, buscada en escala logarítmica)
- **[R27]** Sutskever, Martens, Dahl, Hinton (2013). *On the importance of initialization and momentum in deep learning*. ICML. (momentum clásico y de Nesterov)
- **[R28]** Goyal et al. (2017). *Accurate, Large Minibatch SGD: Training ImageNet in 1 Hour*. arXiv:1706.02677. (regla lineal de la lr con el batch y warmup)
- **[R29]** Malladi, Lyu, Panigrahi, Arora (2022). *On the SDEs and Scaling Rules for Adaptive Gradient Algorithms*. NeurIPS. (regla de la raíz cuadrada para Adam/RMSprop)
- **[R30]** Keskar et al. (2017). *On Large-Batch Training for Deep Learning: Generalization Gap and Sharp Minima*. ICLR.
- **[R31]** Loshchilov, Hutter (2017). *SGDR: Stochastic Gradient Descent with Warm Restarts*. ICLR. (schedule coseno; citada desde EXP-011, añadida a la lista en CHG-018)
- **[R32]** Loshchilov, Hutter (2019). *Decoupled Weight Decay Regularization*. ICLR. (AdamW; citada desde EXP-012, añadida a la lista en CHG-018)
- **[R33]** van Laarhoven (2017). *L2 Regularization versus Batch and Weight Normalization*. arXiv:1706.05350. (delante de BN, L2 no restringe la función y solo cambia la lr efectiva; citada en CHG-020)
- **[R34]** Li, Chen, Hu, Yang (2019). *Understanding the Disharmony between Dropout and Batch Normalization by Variance Shift*. CVPR. (citada en CHG-020)
- **[R35]** Santurkar, Tsipras, Ilyas, Madry (2018). *How Does Batch Normalization Help Optimization?* NeurIPS. (citada en CHG-020)
- **[R36]** Overleaf. *My main file is in a subfolder, but this is not allowed on Overleaf* (documentación de ayuda). https://www.overleaf.com/help/243-my-main-file-is-in-a-subfolder-but-this-is-not-allowed-on-overleaf-but-i-dont-want-to-change-my-folder-structure-and-slash-input-paths
- **[R37]** Overleaf. *Uploading files to a project* (documentación). https://docs.overleaf.com/managing-projects-and-files/adding-files-to-a-project/uploading-files-to-a-project
- **[R38]** Szegedy, Vanhoucke, Ioffe, Shlens, Wojna (2016). *Rethinking the Inception Architecture for Computer Vision*. CVPR. (§7: label smoothing)
- **[R39]** Müller, Kornblith, Hinton (2019). *When Does Label Smoothing Help?* NeurIPS.
- **[R40]** Krizhevsky, Sutskever, Hinton (2012). *ImageNet Classification with Deep Convolutional Neural Networks*. NeurIPS. (§4.1: promedio de la predicción sobre 10 recortes de cada imagen de test)
- **[R41]** Simonyan, Zisserman (2015). *Very Deep Convolutional Networks for Large-Scale Image Recognition*. ICLR. (VGG)
- **[R42]** He, Zhang, Ren, Sun (2016). *Deep Residual Learning for Image Recognition*. CVPR. (ResNet)
- **[R43]** Szegedy et al. (2015). *Going Deeper with Convolutions*. CVPR. (GoogLeNet / Inception v1)
- **[R44]** Sandler, Howard, Zhu, Zhmoginov, Chen (2018). *MobileNetV2: Inverted Residuals and Linear Bottlenecks*. CVPR.
- **[R45]** Lin, Chen, Yan (2014). *Network In Network*. ICLR. (global average pooling como cabeza)
- **[R46]** Springenberg, Dosovitskiy, Brox, Riedmiller (2015). *Striving for Simplicity: The All Convolutional Net*. ICLR (workshop). (convoluciones con stride en lugar de pooling)
- **[R47]** Zhong, Zheng, Kang, Li, Yang (2020). *Random Erasing Data Augmentation*. AAAI.
- **[R48]** DeVries, Taylor (2017). *Improved Regularization of Convolutional Neural Networks with Cutout*. arXiv:1708.04552.
- **[R49]** Zhang, Cisse, Dauphin, Lopez-Paz (2018). *mixup: Beyond Empirical Risk Minimization*. ICLR.
- **[R50]** Yun, Han, Oh, Chun, Choe, Yoo (2019). *CutMix: Regularization Strategy to Train Strong Classifiers with Localizable Features*. ICCV.
- **[R51]** Menon, Jayasumana, Rawat, Jain, Veit, Kumar (2021). *Long-tail learning via logit adjustment*. ICLR.
- **[R52]** Luo, Li, Urtasun, Zemel (2016). *Understanding the Effective Receptive Field in Deep Convolutional Neural Networks*. NeurIPS. (el campo receptivo efectivo es mucho menor que el teórico)

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
