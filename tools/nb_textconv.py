"""Git textconv for notebooks: prints only cell sources, so `git diff` shows code changes instead of outputs/metadata."""
import json
import sys

sys.stdout.reconfigure(encoding='utf-8')
with open(sys.argv[1], encoding='utf-8') as ifs:
    raw = ifs.read()
try:
    nb = json.loads(raw)
except json.JSONDecodeError:
    print(raw)
    sys.exit(0)
for cell in nb.get('cells', []):
    print('# ---- %s ----' % cell['cell_type'])
    print(''.join(cell['source']))
    print()
