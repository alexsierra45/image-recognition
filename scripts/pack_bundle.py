"""Zips the single Overleaf project (reports/overleaf/, every delivery; DEC-011) into reports/overleaf_project.zip
(files at the zip root, *.proposed.tex excluded) and says what to upload for the given delivery.

- First time: Overleaf > New Project > Upload Project with the zip.
- Later deliveries, into the same project: upload the folder delivery_NN/ and the two main documents
  delivery_NN_report.tex / delivery_NN_slides.tex (Upload button of the file tree; folders keep their structure).
  common/ only if it changed.

Usage: python scripts/pack_bundle.py reports/delivery_02
"""
import os
import sys
import zipfile

import project_paths


def main(delivery_dir):
    root = project_paths.PROJECT
    out = os.path.join(project_paths.REPORTS, 'overleaf_project.zip')
    n = 0
    with zipfile.ZipFile(out, 'w', zipfile.ZIP_DEFLATED) as z:
        for d, dirs, files in os.walk(root):
            dirs.sort()
            for f in sorted(files):
                if f.endswith('.proposed.tex'):
                    continue
                p = os.path.join(d, f)
                z.write(p, os.path.relpath(p, root))
                n += 1
    deliveries = sorted(x for x in os.listdir(root) if os.path.isdir(os.path.join(root, x)) and x != 'common')
    name = project_paths.name(delivery_dir)
    print('%d files (%s) -> %s' % (n, ', '.join(deliveries), out))
    print('Overleaf, first time: New Project > Upload Project with the zip.')
    print('Overleaf, existing project: upload the folder %s/ and %s, %s (common/ only if it changed).'
          % (name, project_paths.main_file(delivery_dir, 'report'), project_paths.main_file(delivery_dir, 'slides')))
    print('Compile: open %s or %s and Recompile (pdfLaTeX).'
          % (project_paths.main_file(delivery_dir, 'report'), project_paths.main_file(delivery_dir, 'slides')))


if __name__ == '__main__':
    main(sys.argv[1])
