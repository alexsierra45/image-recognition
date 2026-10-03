---
name: log-experiment
description: Registra en LOGBOOK.md un experimento, cambio, incidencia, idea o decisión del proyecto xview_recognition, con fundamento teórico citado y resultados sacados de experiments/. Usar tras modificar el notebook o el código, tras un entrenamiento, al detectar un problema y antes de cualquier commit.
argument-hint: "[qué se hizo, opcional]"
---

# Registrar en la bitácora

Argumento del usuario (puede estar vacío): $ARGUMENTS

## 1. Reunir la evidencia (no preguntar lo que se puede leer)
- `git status` y `git diff -- '*.ipynb' '*.py'`. Gracias al textconv, el diff de los notebooks muestra solo el código de las celdas. Si aún no hay commits, leer el notebook directamente.
- `experiments/runs.csv`: filas nuevas desde la última entrada de la bitácora. `run_id` es la clave; si hay filas duplicadas, cuenta la última.
- Por cada run nuevo, en `experiments/runs/<run_id>/`:
  - `history.csv`: curvas. ¿Hay underfitting, overfitting o una divergencia? ¿En qué época está el mejor resultado? ¿Actuó ReduceLROnPlateau (columna `learning_rate`)?
  - `confusion_matrix.csv`: recall por clase y los pares de clases que más se confunden.
  - `config.json` y `model_summary.txt`: qué configuración se usó realmente.
- Las salidas del notebook (logs de `fit`, métricas impresas) si no hay run guardado.
- LOGBOOK.md: último ID de cada tipo, estado actual, ideas e incidencias abiertas.

## 2. Clasificar
- **EXP**: cambia el modelo, los datos, el preprocesado o el entrenamiento. Tiene, o tendrá, métricas.
- **CHG**: cambio de código sin efecto en el modelo (refactor, instrumentación, utilidades).
- **ISS**: bug, inconsistencia o problema del entorno.
- **IDEA**: mejora todavía no probada.
- **DEC**: elección entre alternativas.

Un mismo cambio puede generar varias entradas: por ejemplo, un EXP que resuelve una ISS y cierra una IDEA.

## 3. Fundamentar
- Buscar primero en `references/` (índice en `references/README.md`): leer la parte relevante del PDF y citarla como `[C#, p. X]`.
- Si no está ahí, usar la bibliografía canónica (`[R#]`, añadiéndola a "Referencias externas" si es nueva). Verificar la referencia si hay alguna duda; no inventar citas.
- Si la base es solo empírica o heurística, decirlo.
- Preferir argumentos cuantitativos con los datos reales: número de parámetros, frecuencias de clase, varianza de las estimaciones, escalas de las activaciones.

## 4. Preguntar solo lo imprescindible
La hipótesis o motivación del usuario, si no es obvia por el código. Nunca rellenar resultados de memoria: si un run no se ha ejecutado, la entrada queda *pendiente*.

## 5. Escribir
- Usar las plantillas que hay al final de LOGBOOK.md. Para un EXP, indicar la base y el único factor que cambia. Comparar con la base usando las mismas métricas (`val_accuracy`, `val_mean_recall`, `val_min_recall`, hueco train-val).
- Actualizar "Estado actual" (el mejor experimento según mean recall) y el "Índice de experimentos".
- Actualizar el estado de las ISS y las IDEA afectadas (resuelta → CHG/EXP, probada → EXP), sin borrarlas.
- Si el `EXP_ID` del notebook no coincide con la entrada, avisar al usuario.

## 6. Cerrar
Resumir al usuario en 2 o 3 líneas qué IDs se crearon o actualizaron y cuál es el siguiente paso recomendado.
