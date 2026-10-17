"""Layout of the single Overleaf project that holds every delivery (DEC-011).

    reports/overleaf/                    <- the Overleaf project (one for the whole course)
        common/                          <- shared style, one copy
        delivery_NN_report.tex           <- main documents, at the root (Overleaf does not allow them in subfolders)
        delivery_NN_slides.tex
        delivery_NN/figures|generated|sections/

Each main document adds its own folder to LaTeX's search path (\\input@path, and \\graphicspath for the images), so
the sections keep their relative paths (figures/..., generated/..., sections/...).
The sources of each delivery (delivery.yaml, drafts/, figure_specs.yaml, review notes) stay in reports/delivery_NN/.

Usage: python scripts/project_paths.py reports/delivery_02     (writes the two main documents if they are missing)
"""
import os
import re
import sys

REPORTS = os.path.normpath(os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'reports'))
PROJECT = os.path.join(REPORTS, 'overleaf')
TEMPLATE = os.path.join(REPORTS, 'template', 'overleaf')
KINDS = ('report', 'slides')
# \\input@path: \\input and \\IfFileExists; \\graphicspath: \\includegraphics (Beamer does not use \\input@path for it)
SEARCH_PATH_LINE = ('\\makeatletter\\def\\input@path{{%(d)s/}}\\makeatother   %% this delivery\'s files: %(d)s/{sections,generated,figures}\n'
                    '\\AtBeginDocument{\\graphicspath{{%(d)s/}}}\n')


def name(delivery_dir):
    """reports/delivery_02 -> delivery_02"""
    return os.path.basename(os.path.normpath(delivery_dir))


def delivery_root(delivery_dir):
    """Folder of this delivery inside the project: reports/overleaf/delivery_NN."""
    return os.path.join(PROJECT, name(delivery_dir))


def main_file(delivery_dir, kind):
    """Main document of this delivery, relative to the project root: delivery_NN_report.tex / delivery_NN_slides.tex."""
    return '%s_%s.tex' % (name(delivery_dir), kind)


def search_path(root, main):
    """Folders where LaTeX looks for files after the project root (\\input@path of the main document)."""
    with open(os.path.join(root, main), encoding='utf-8') as ifs:
        m = re.search(r'\\def\\input@path\{((?:\{[^}]*\})*)\}', ifs.read())
    return [os.path.join(root, p) for p in re.findall(r'\{([^}]*)\}', m.group(1))] if m else []


def write_main(delivery_dir, kind, src=None):
    """Main document from the template (or from `src`), with this delivery's search path. Never overwrites."""
    dst = os.path.join(PROJECT, main_file(delivery_dir, kind))
    if os.path.exists(dst):
        return None
    with open(src or os.path.join(TEMPLATE, kind + '.tex'), encoding='utf-8') as ifs:
        text = ifs.read()
    i = text.index('\n', text.index('\\documentclass')) + 1
    text = text[:i] + SEARCH_PATH_LINE % {'d': name(delivery_dir)} + text[i:]
    with open(dst, 'w', encoding='utf-8') as ofs:
        ofs.write(text)
    return dst


if __name__ == '__main__':
    for k in KINDS:
        p = write_main(sys.argv[1], k)
        print('%-9s %s' % ('written' if p else 'exists', main_file(sys.argv[1], k)))
