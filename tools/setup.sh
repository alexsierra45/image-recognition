#!/usr/bin/env bash
# Local git config this project relies on (git does not version it): run once after `git init` or `git clone`.
cd "$(dirname "$0")/.." || exit 1
git config core.hooksPath .githooks
git config diff.ipynb.textconv "python tools/nb_textconv.py"
echo "OK: pre-commit hook activo y diff de notebooks configurado."
