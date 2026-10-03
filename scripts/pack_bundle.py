"""Zips <delivery>/overleaf/ into <delivery>/delivery_NN_overleaf.zip (files at the zip root, *.proposed.tex excluded).

Usage: python scripts/pack_bundle.py reports/delivery_01
"""
import os
import sys
import zipfile


def main(delivery_dir):
    root = os.path.join(delivery_dir, 'overleaf')
    name = os.path.basename(os.path.normpath(delivery_dir))
    out = os.path.join(delivery_dir, '%s_overleaf.zip' % name)
    n = 0
    with zipfile.ZipFile(out, 'w', zipfile.ZIP_DEFLATED) as z:
        for d, _, files in os.walk(root):
            for f in sorted(files):
                if f.endswith('.proposed.tex'):
                    continue
                p = os.path.join(d, f)
                z.write(p, os.path.relpath(p, root))
                n += 1
    print('%d files -> %s (Overleaf: New Project > Upload Project; main document report.tex or slides.tex; pdfLaTeX)' % (n, out))


if __name__ == '__main__':
    main(sys.argv[1])
