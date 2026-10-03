#!/usr/bin/env bash
# PreToolUse (Bash): blocks a `git commit` from Claude when code/notebooks changed but LOGBOOK.md did not.
# Exit code 2 = block, stderr is fed back to Claude. The .githooks/pre-commit hook is the safety net for manual commits.
cat | grep -qE '"command" *: *"[^"]*\bgit\b[^"]*\bcommit\b' || exit 0
cd "$(dirname "$0")/../.." || exit 0
# Staged + unstaged, so `git commit -a` is covered too
changed=$( { git diff --name-only; git diff --cached --name-only; } 2>/dev/null | sort -u )
code=$(printf '%s\n' "$changed" | grep -E '\.(ipynb|py)$')
if [ -n "$code" ] && ! printf '%s\n' "$changed" | grep -qx 'LOGBOOK.md'; then
    echo "Commit bloqueado: hay cambios en código/notebooks y LOGBOOK.md no está actualizado:" >&2
    printf '%s\n' "$code" | sed 's/^/  - /' >&2
    echo "Registra los cambios en LOGBOOK.md (skill log-experiment), añádelo al commit y vuelve a intentarlo." >&2
    exit 2
fi
exit 0
