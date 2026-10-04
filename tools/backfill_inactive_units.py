"""Adds val.inactive_units (experiment_utils.inactive_units, CHG-009) to the history of runs logged before it existed,
measured on the saved model of each run (experiments/models/<id>.keras). Nothing else in the history changes.

Usage: python tools/backfill_inactive_units.py EXP-005 [more id prefixes] [--check ID]
  --check ID  recompute a run that already has the value and compare (sanity check of the method).
"""
import glob
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import experiment_utils as eu  # noqa: E402
import keras  # noqa: E402

EXP = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'experiments')


def measure(run_id):
    path = os.path.join(EXP, 'history', run_id + '.json')
    h = json.load(open(path))
    data = eu.get_data(h['config']['res'], h['config']['stratify'])
    model = keras.models.load_model(os.path.join(EXP, 'models', run_id + '.keras'))
    value = eu.inactive_units(model, data['x'], data['valid'])
    keras.backend.clear_session()
    return path, h, value


def main(args):
    if '--check' in args:
        run_id = args[args.index('--check') + 1]
        _, h, value = measure(run_id)
        print('%s: logged %s, recomputed %s' % (run_id, h['val'].get('inactive_units'), value))
        return
    for prefix in args:
        for path in sorted(glob.glob(os.path.join(EXP, 'history', prefix + '*.json'))):
            run_id = os.path.basename(path)[:-5]
            if json.load(open(path))['val'].get('inactive_units') is not None:
                print('%-28s already logged' % run_id)
                continue
            path, h, value = measure(run_id)
            h['val']['inactive_units'] = value
            with open(path, 'w') as ofs:
                json.dump(h, ofs, indent=1)
            print('%-28s res %3d  inactive %.4f' % (run_id, h['config']['res'], value))


if __name__ == '__main__':
    main(sys.argv[1:])
