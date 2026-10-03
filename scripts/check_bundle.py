"""Checks an Overleaf bundle without compiling it (no LaTeX installed locally).

Errors block the bundle; warnings are reported. Usage: python scripts/check_bundle.py reports/delivery_01
"""
import os
import re
import shutil
import subprocess
import sys

ALLOWED_PACKAGES = {
    'inputenc', 'fontenc', 'babel', 'amsmath', 'amssymb', 'graphicx', 'booktabs', 'xcolor', 'geometry', 'hyperref',
    'caption', 'subcaption', 'array', 'tabularx', 'microtype', 'lmodern', 'enumitem', 'float', 'url', 'siunitx',
    'multirow', 'makecell', 'parskip', 'titlesec', 'fancyhdr', 'csquotes', 'textcomp', 'tikz', 'pgfpages',
}
# Characters pdfLaTeX (inputenc utf8 + T1) handles in text: ASCII, Latin-1 letters and common punctuation.
SAFE_EXTRA = set('áéíóúÁÉÍÓÚñÑüÜ¿¡«»“”‘’–—…ºª°·')
FRAME_MIN, FRAME_MAX = 8, 10
WORDS_PER_PAGE, FIG_PAGES, TABLE_PAGES = 500, 0.3, 0.2


class Report:
    def __init__(self):
        self.errors, self.warnings = [], []

    def err(self, msg):
        self.errors.append(msg)

    def warn(self, msg):
        self.warnings.append(msg)


def strip_comments(text):
    return '\n'.join(re.sub(r'(?<!\\)%.*$', '', line) for line in text.split('\n'))


def read(path):
    with open(path, encoding='utf-8') as ifs:
        return ifs.read()


def resolve(root, name, exts):
    for e in [''] + exts:
        p = os.path.join(root, name + e)
        if os.path.isfile(p):
            return p
    return None


def collect(root, main, rep, seen=None):
    """Main file + every \\input/\\include reachable from it (recursively)."""
    seen = seen if seen is not None else []
    path = os.path.join(root, main)
    if path in seen:
        return seen
    seen.append(path)
    for m in re.finditer(r'\\(?:input|include)\{([^}]+)\}', strip_comments(read(path))):
        if '#' in m.group(1):  # macro parameter inside a definition (e.g. \\safetable)
            continue
        target = resolve(root, m.group(1), ['.tex'])
        if target is None:
            rep.err('%s: \\input{%s} does not exist' % (os.path.relpath(path, root), m.group(1)))
        else:
            collect(root, os.path.relpath(target, root), rep, seen)
    return seen


def defined_keys(path, prefix):
    if not os.path.exists(path):
        return set()
    return set(re.findall(r'\\csname %s@([^\\]+)\\endcsname' % prefix, read(path)))


def check_text(root, path, rep, res_keys, fig_keys, labels, refs):
    rel = os.path.relpath(path, root)
    raw = read(path)
    text = strip_comments(raw)
    for m in re.finditer(r'\\(includegraphics|safegraphics)(?:\[[^\]]*\])?\{([^}]+)\}', text):
        if '#' not in m.group(2) and resolve(root, m.group(2), ['.pdf', '.png', '.jpg', '.jpeg']) is None:
            rep.err('%s: \\%s{%s} does not exist' % (rel, m.group(1), m.group(2)))
    for m in re.finditer(r'\\safetable\{([^}]+)\}', text):
        if '#' not in m.group(1) and resolve(root, m.group(1), ['.tex']) is None:
            rep.err('%s: \\safetable{%s} does not exist' % (rel, m.group(1)))
    for m in re.finditer(r'\\todoslot\{', text):
        if 'newcommand' not in text[max(0, m.start() - 20):m.start()]:
            rep.err('%s:%d: \\todoslot left (pending text)' % (rel, text[:m.start()].count('\n') + 1))
    for k in re.findall(r'\\res\{([^}]+)\}', text):
        if k not in res_keys:
            rep.err('%s: \\res{%s} is not defined in generated/results_macros.tex' % (rel, k))
    for k in re.findall(r'\\figcaption\{([^}]+)\}', text):
        if k not in fig_keys:
            rep.err('%s: \\figcaption{%s} is not defined in generated/figure_captions.tex' % (rel, k))
    labels.update(re.findall(r'\\label\{([^}]+)\}', text))
    refs.update(re.findall(r'\\(?:ref|autoref|eqref)\{([^}]+)\}', text))
    # braces (ignoring escaped ones)
    depth = 0
    for i, ch in enumerate(text):
        if ch in '{}' and i > 0 and text[i - 1] == '\\':
            continue
        depth += 1 if ch == '{' else -1 if ch == '}' else 0
        if depth < 0:
            rep.err('%s: unbalanced "}" near line %d' % (rel, text[:i].count('\n') + 1))
            depth = 0
    if depth:
        rep.err('%s: %d unclosed "{"' % (rel, depth))
    # environments
    stack = []
    for m in re.finditer(r'\\(begin|end)\{([^}]+)\}', text):
        line = text[:m.start()].count('\n') + 1
        if m.group(1) == 'begin':
            stack.append((m.group(2), line))
        elif not stack or stack[-1][0] != m.group(2):
            rep.err('%s:%d: \\end{%s} without matching \\begin' % (rel, line, m.group(2)))
        else:
            stack.pop()
    for env, line in stack:
        rep.err('%s:%d: \\begin{%s} never closed' % (rel, line, env))
    # characters pdfLaTeX cannot typeset in text mode (outside $...$)
    no_math = re.sub(r'\$[^$]*\$', '', text)
    for ch in sorted(set(no_math)):
        if ord(ch) > 127 and ch not in SAFE_EXTRA:
            rep.err('%s: character %r (U+%04X) is not pdfLaTeX-safe in text mode' % (rel, ch, ord(ch)))
    # unescaped specials in generated files
    if '%sgenerated%s' % (os.sep, os.sep) in path:
        for n, line in enumerate(text.split('\n'), 1):
            body = re.sub(r'\$[^$]*\$', '', line)
            body = re.sub(r'\\(?:InputIfFileExists|input|detokenize)\{[^}]*\}|#\d', '', body)
            body = re.sub(r'\\(csname|expandafter|def|endcsname|newcommand|providecommand|ifcsname|else|fi)\b', '', body)
            if re.search(r'(?<!\\)[_#]', body) and 'csname' not in line:
                rep.err('%s:%d: unescaped "_" or "#"' % (rel, n))
    # packages
    for m in re.finditer(r'\\usepackage(?:\[[^\]]*\])?\{([^}]+)\}', text):
        for pkg in m.group(1).split(','):
            if pkg.strip() not in ALLOWED_PACKAGES:
                rep.err('%s: package %s is not in the whitelist' % (rel, pkg.strip()))
    if 'shell-escape' in text or '\\write18' in text:
        rep.err('%s: requires shell-escape' % rel)
    return text


def check_slides(files_text, rep):
    frames = re.findall(r'\\begin\{frame\}(.*?)\\end\{frame\}', files_text, flags=re.S)
    content = [f for f in frames if '\\titlepage' not in f and 'Referencias' not in f[:200]]
    if not FRAME_MIN <= len(content) <= FRAME_MAX:
        rep.warn('slides: %d content frames (expected %d-%d)' % (len(content), FRAME_MIN, FRAME_MAX))
    total = 0.0
    for i, f in enumerate(content, 1):
        title = re.search(r'^\s*\{([^}]*)\}', f)
        name = title.group(1) if title else '#%d' % i
        m = re.search(r'\\speakernote\{(\d+(?:[.,]\d+)?)\}', f) or \
            re.search(r'\\note\{[^}]*?(\d+(?:[.,]\d+)?)\s*min', f, flags=re.S)
        if not m:
            rep.err('slides: frame "%s" has no \\speakernote{minutes}{...}' % name)
        else:
            total += float(m.group(1).replace(',', '.'))
    if not 8 <= total <= 11:
        rep.warn('slides: speaker notes add up to %.1f min (target ~10)' % total)
    return len(content), total


def check_report_length(files_text, rep):
    body = re.sub(r'\\(res|input|include|label|ref|cite|includegraphics|safegraphics|safetable|figcaption|bibitem)(\[[^\]]*\])?\{[^}]*\}', ' x ', files_text)
    body = re.sub(r'\\[a-zA-Z]+\*?(\[[^\]]*\])?', ' ', body)
    words = len(re.findall(r'[A-Za-záéíóúñÁÉÍÓÚÑü]{2,}', body))
    figs = len(re.findall(r'\\(?:includegraphics|safegraphics)', files_text))
    tables = len(re.findall(r'\\begin\{table\}', files_text))
    pages = words / WORDS_PER_PAGE + figs * FIG_PAGES + tables * TABLE_PAGES
    msg = 'report: ~%d words, %d figures, %d tables -> ~%.1f pages (estimate)' % (words, figs, tables, pages)
    if not 4 <= pages <= 6.5:
        rep.warn(msg + ' -- outside 4-6 pages')
    return msg


def main(delivery_dir):
    root = os.path.join(delivery_dir, 'overleaf')
    rep = Report()
    gen = os.path.join(root, 'generated')
    res_keys = defined_keys(os.path.join(gen, 'results_macros.tex'), 'res')
    fig_keys = defined_keys(os.path.join(gen, 'figure_captions.tex'), 'figcap')
    summary = []
    for main_file, kind in [('report.tex', 'report'), ('slides.tex', 'slides')]:
        if not os.path.exists(os.path.join(root, main_file)):
            rep.err('%s is missing' % main_file)
            continue
        labels, refs, texts = set(), set(), []
        for path in collect(root, main_file, rep):
            texts.append(check_text(root, path, rep, res_keys, fig_keys, labels, refs))
        for r in sorted(refs - labels):
            rep.err('%s: \\ref{%s} has no \\label' % (main_file, r))
        allt = '\n'.join(texts)
        body_text = '\n'.join(t for p, t in zip(collect(root, main_file, Report()), texts)
                              if os.sep + 'sections' + os.sep in p)
        if kind == 'slides':
            n, total = check_slides(allt, rep)
            summary.append('slides: %d content frames, speaker notes %.1f min' % (n, total))
        else:
            summary.append(check_report_length(body_text, rep))
    for tool in ['tectonic', 'latexmk']:
        if shutil.which(tool):
            for main_file in ['report.tex', 'slides.tex']:
                cmd = [tool, main_file] if tool == 'tectonic' else [tool, '-pdf', '-interaction=nonstopmode', main_file]
                r = subprocess.run(cmd, cwd=root, capture_output=True, text=True)
                (rep.err if r.returncode else summary.append)('%s %s: %s' % (tool, main_file, 'failed' if r.returncode else 'ok'))
            break
    else:
        summary.append('no LaTeX compiler available: compilation not checked (compile in Overleaf with pdfLaTeX)')
    print('\n'.join(summary))
    for w in rep.warnings:
        print('WARNING:', w)
    for e in rep.errors:
        print('ERROR:', e)
    print('%d errors, %d warnings' % (len(rep.errors), len(rep.warnings)))
    return 1 if rep.errors else 0


if __name__ == '__main__':
    sys.exit(main(sys.argv[1]))
