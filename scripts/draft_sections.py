"""Installs the section drafts (<delivery>/drafts/*.tex, written by Claude) into <delivery>/overleaf/sections/
without overwriting the user's edits.

A section is (re)written only if it does not exist or if it is unchanged since the system last wrote it
(sha256 kept in <delivery>/.build_state.json). If the user edited it, the new draft is saved as
sections/<name>.proposed.tex and reported; the edited file is left untouched.

Usage: python scripts/draft_sections.py reports/delivery_01 [--status]
"""
import hashlib
import json
import os
import shutil
import sys


def sha(path):
    with open(path, 'rb') as ifs:
        return hashlib.sha256(ifs.read()).hexdigest()


def main(delivery_dir, status_only=False):
    drafts = os.path.join(delivery_dir, 'drafts')
    sections = os.path.join(delivery_dir, 'overleaf', 'sections')
    state_path = os.path.join(delivery_dir, '.build_state.json')
    state = json.load(open(state_path)) if os.path.exists(state_path) else {}
    os.makedirs(sections, exist_ok=True)
    for name in sorted(os.listdir(drafts)):
        if not name.endswith('.tex'):
            continue
        src, dst = os.path.join(drafts, name), os.path.join(sections, name)
        if not os.path.exists(dst):
            action = 'new'
        elif state.get(name) == sha(dst):
            action = 'unchanged' if sha(src) == sha(dst) else 'update'
        else:
            action = 'user-edited'
        if status_only:
            print('%-12s %s' % (action, name))
            continue
        if action in ('new', 'update'):
            shutil.copyfile(src, dst)
            state[name] = sha(dst)
            print('%-12s %s' % ('written', name))
        elif action == 'user-edited':
            proposed = dst[:-4] + '.proposed.tex'
            if sha(src) != sha(dst):
                shutil.copyfile(src, proposed)
                print('%-12s %s -> kept your version; new draft in %s' % ('EDITED', name, os.path.basename(proposed)))
            else:
                state[name] = sha(dst)
                print('%-12s %s (your edit equals the draft)' % ('in sync', name))
        else:
            print('%-12s %s' % ('unchanged', name))
    if not status_only:
        with open(state_path, 'w') as ofs:
            json.dump(state, ofs, indent=1, sort_keys=True)


if __name__ == '__main__':
    main(sys.argv[1], '--status' in sys.argv)
