"""Run a notebook end to end without a UI, saving outputs into it.

Usage (needs nbclient + nbformat in the interpreter that runs this script, e.g. the project .venv):
    .venv/Scripts/python.exe tools/run_notebook.py ffNN_example.ipynb --python <kernel interpreter>

The kernel runs with --python (it needs ipykernel), so the notebook can use a different environment than this
script: today the notebook needs the global Python 3.12 (TensorFlow, cell 1). Cells run in order from the
notebook's folder. After each cell a progress copy is saved to the temp dir. Keras progress-bar redraws
(\\r, \\b) are collapsed to what a viewer would see. The notebook is overwritten only if it did not change during
the run; otherwise the executed copy is left in the temp dir.
"""
import argparse
import asyncio
import hashlib
import json
import os
import sys
import tempfile
import time


def log(msg):
    print(time.strftime('%H:%M:%S ') + msg, flush=True)


def sha(path):
    with open(path, 'rb') as ifs:
        return hashlib.sha256(ifs.read()).hexdigest()


def dump(nb, path):
    with open(path, 'w', encoding='utf-8', newline='\n') as ofs:
        json.dump(nb, ofs, indent=1, ensure_ascii=False)
        ofs.write('\n')


def clean_streams(outputs):
    """Merge consecutive stream outputs and resolve \\r/\\b progress-bar redraws."""
    merged = []
    for out in outputs:
        if out.get('output_type') == 'stream' and merged and merged[-1].get('output_type') == 'stream' and merged[-1]['name'] == out['name']:
            merged[-1]['text'] += out['text']
        else:
            merged.append(out)
    for out in merged:
        if out.get('output_type') == 'stream':
            lines = []
            for line in out['text'].replace('\x08', '').split('\n'):
                segs = [s for s in line.split('\r') if s.strip()]
                lines.append(segs[-1] if segs else '')
            out['text'] = '\n'.join(lines)
    return merged


def main():
    parser = argparse.ArgumentParser(description=__doc__.split('\n')[0])
    parser.add_argument('notebook')
    parser.add_argument('--python', default=sys.executable, help='interpreter for the kernel (default: this one)')
    args = parser.parse_args()

    nb_path = os.path.abspath(args.notebook)
    name = os.path.splitext(os.path.basename(nb_path))[0]
    tmp = tempfile.mkdtemp(prefix='run_notebook_')
    # Throwaway kernelspec, so the user's Jupyter kernels are not touched
    spec_dir = os.path.join(tmp, 'kernels', 'run-notebook')
    os.makedirs(spec_dir)
    with open(os.path.join(spec_dir, 'kernel.json'), 'w', encoding='utf-8') as ofs:
        json.dump({'argv': [args.python, '-m', 'ipykernel_launcher', '-f', '{connection_file}'], 'display_name': 'run_notebook', 'language': 'python'}, ofs)
    os.environ['JUPYTER_PATH'] = tmp + os.pathsep + os.environ.get('JUPYTER_PATH', '')

    if sys.platform == 'win32':
        # zmq needs add_reader, which the default Proactor loop on Windows lacks
        asyncio.set_event_loop_policy(asyncio.WindowsSelectorEventLoopPolicy())
    import nbformat
    from nbclient import NotebookClient

    original_hash = sha(nb_path)
    nb = nbformat.read(nb_path, as_version=4)
    kernelspec = dict(nb.metadata.get('kernelspec', {}))
    progress = os.path.join(tmp, name + '.running.ipynb')
    client = NotebookClient(nb, kernel_name='run-notebook', timeout=None, resources={'metadata': {'path': os.path.dirname(nb_path)}})
    log('START %s (%d cells) kernel=%s progress=%s' % (nb_path, len(nb.cells), args.python, progress))
    failed = False
    with client.setup_kernel():
        for idx, cell in enumerate(nb.cells):
            if cell.cell_type != 'code':
                continue
            log('cell %d start: %s' % (idx, cell.source.split('\n')[0][:70]))
            t0 = time.time()
            try:
                client.execute_cell(cell, idx)
            except Exception as exc:
                failed = True
                log('cell %d FAILED after %.0fs: %s' % (idx, time.time() - t0, str(exc)[-1500:]))
            cell.outputs = clean_streams(cell.outputs)
            dump(nb, progress)
            if failed:
                break
            log('cell %d done in %.0fs' % (idx, time.time() - t0))

    nb.metadata['kernelspec'] = kernelspec
    if failed:
        log('ABORTED: notebook untouched; partial outputs in ' + progress)
        sys.exit(1)
    if sha(nb_path) == original_hash:
        dump(nb, nb_path)
        log('DONE: outputs written to ' + nb_path)
    else:
        executed = os.path.join(tmp, name + '.executed.ipynb')
        dump(nb, executed)
        log('DONE, but the notebook changed during the run: NOT overwritten; executed copy at ' + executed)


if __name__ == '__main__':
    main()
