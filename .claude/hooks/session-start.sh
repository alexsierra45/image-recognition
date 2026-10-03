#!/usr/bin/env bash
# SessionStart: tells Claude about code/notebook changes not reflected in LOGBOOK.md,
# and checks the local git config the project relies on. Stdout is added to Claude's context.
cd "$(dirname "$0")/../.." || exit 0
git rev-parse --git-dir >/dev/null 2>&1 || { echo "Aviso: image_recognition no es un repositorio git; ejecutar 'git init' y 'bash tools/setup.sh'."; exit 0; }

[ "$(git config --get core.hooksPath)" = ".githooks" ] || echo "Aviso: el hook pre-commit no está activo; ejecutar 'bash tools/setup.sh'."
[ -n "$(git config --get diff.ipynb.textconv)" ] || echo "Aviso: diff de notebooks sin configurar; ejecutar 'bash tools/setup.sh'."

if git rev-parse --verify -q HEAD >/dev/null; then
    changed=$( { git diff --name-only HEAD; git ls-files --others --exclude-standard; } 2>/dev/null | sort -u )
else
    changed=$( { git diff --cached --name-only; git ls-files --others --exclude-standard; } 2>/dev/null | sort -u )
fi
logbook_changed=$(printf '%s\n' "$changed" | grep -x 'LOGBOOK.md')

unlogged=""
while IFS= read -r f; do
    [ -n "$f" ] || continue
    # Unlogged if LOGBOOK.md was not touched at all, or the file was saved after the last LOGBOOK.md update
    if [ -z "$logbook_changed" ] || [ "$f" -nt LOGBOOK.md ]; then
        unlogged="$unlogged\n  - $f"
    fi
done <<EOF
$(printf '%s\n' "$changed" | grep -E '\.(ipynb|py)$')
EOF
if [ -n "$unlogged" ]; then
    printf 'Bitácora: estos archivos tienen cambios sin commitear que LOGBOOK.md quizá no refleja:%b\n' "$unlogged"
    echo "Revisar con 'git diff' (y experiments/runs.csv si hubo entrenamientos) y proponer al usuario registrarlos con /log-experiment."
fi
exit 0
