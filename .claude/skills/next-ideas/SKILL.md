---
name: next-ideas
description: Analiza el estado del proyecto xview_recognition (LOGBOOK.md, runs, curvas, matriz de confusión, notebook, references/) y propone los siguientes experimentos priorizados, cada uno con hipótesis, fundamento teórico citado, coste estimado en CPU y criterio de éxito. Usar cuando el usuario pregunte qué probar a continuación o empiece un tema nuevo del curso.
argument-hint: "[tema del curso actual, opcional]"
---

# Proponer los siguientes experimentos

Tema o foco indicado por el usuario (puede estar vacío): $ARGUMENTS

## 1. Diagnóstico con evidencia
- LOGBOOK.md: estado actual, índice, ideas e incidencias abiertas, y conclusiones de los últimos EXP.
- Mejor run de `experiments/runs.csv` y su carpeta:
  - Curvas (`history.csv`): ¿underfitting (train y val bajos), overfitting (hueco train-val creciente) u optimización inestable?
  - `confusion_matrix.csv`: clases con peor recall y los pares que más se confunden. ¿Son confusiones visualmente razonables (Truck/Dump truck, Bus/Small car) o un sesgo hacia las clases mayoritarias?
- El notebook actual, por si hay incidencias nuevas.
- `references/`: si hay documentos nuevos del curso (el tema que se está viendo), priorizar ideas que lo pongan en práctica. Si no consta en qué tema va el usuario, preguntarlo.

## 2. Proponer de 3 a 5 experimentos
Ordenados por impacto esperado / coste. Para cada uno:
- **Hipótesis** y qué síntoma del diagnóstico ataca.
- **Cambio concreto**: un solo factor respecto al mejor EXP.
- **Fundamento** con cita (`[C#, p. X]` de `references/` o `[R#]`). Si es heurística, decirlo.
- **Coste estimado** en CPU (parámetros, tiempo por época relativo al run base) y riesgo.
- **Criterio de éxito** medible (p. ej. +X pp de mean recall, o reducir el hueco train-val).

Antes de proponer, revisar si alguna IDEA del backlog ya cubre la propuesta y reutilizarla.

## 3. Cerrar
Preguntar cuáles añadir al backlog como IDEA-### (o repriorizar las existentes) y registrarlas en LOGBOOK.md. No modificar el notebook sin confirmación del usuario.
