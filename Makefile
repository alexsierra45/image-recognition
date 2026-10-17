# Deliveries (report + Beamer slides) for Overleaf. Usage: make bundle D=01
# Every delivery lives in ONE Overleaf project, reports/overleaf/ (DEC-011); bundle zips it to reports/overleaf_project.zip.
# Read-only on experiments/. Only Python from the project .venv is needed (no LaTeX: compile in Overleaf).
PY  ?= .venv/bin/python
D   ?= 01
DIR := reports/delivery_$(D)

.PHONY: results figures sections status check bundle refresh

results:
	$(PY) scripts/build_results.py $(DIR)

figures:
	$(PY) scripts/build_figures.py $(DIR)

sections:          ## install drafts/ into reports/overleaf/delivery_NN/sections/ (no overwrite of edits) + main documents
	$(PY) scripts/draft_sections.py $(DIR)

status:            ## which sections were edited by hand
	$(PY) scripts/draft_sections.py $(DIR) --status

check:
	$(PY) scripts/check_bundle.py $(DIR)

refresh: results figures   ## regenerate only generated/ and figures/ (never touches sections/)

bundle: results figures sections check
	$(PY) scripts/pack_bundle.py $(DIR)
