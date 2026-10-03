#!/usr/bin/env bash
# PostToolUse (Edit|Write|NotebookEdit): after Claude edits a notebook or .py file, remind it to log the change.
path=$(cat | grep -oE '"(file_path|notebook_path)" *: *"[^"]*"' | head -1 | sed -E 's/.*: *"(.*)"/\1/')
case "$path" in
    *.ipynb|*.py) ;;
    *) exit 0 ;;
esac
name=$(basename "$(printf '%s' "$path" | tr '\\' '/')")
printf '{"hookSpecificOutput":{"hookEventName":"PostToolUse","additionalContext":"Se modificó %s. Antes de terminar la tarea, registrar el cambio en LOGBOOK.md (CHG/EXP/ISS/IDEA) con su justificación teórica."}}\n' "$name"
exit 0
