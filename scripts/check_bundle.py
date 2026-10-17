"""Checks an Overleaf bundle without compiling it (no LaTeX installed locally).

Errors block the bundle; warnings are reported. Usage: python scripts/check_bundle.py reports/delivery_01 [project_root]
"""
import os
import re
import shutil
import subprocess
import sys

import project_paths

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


SEARCH_PATH = []   # folders LaTeX searches after the project root (\\input@path of the main document, DEC-011)


def resolve(root, name, exts):
    for d in [root] + SEARCH_PATH:
        for e in [''] + exts:
            p = os.path.join(d, name + e)
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


def math_keys(path, prefix):
    """Keys whose value already carries $...$ (e.g. lr = $3{,}6\\cdot 10^{-3}$): they cannot go inside math mode."""
    if not os.path.exists(path):
        return set()
    return {k for k, v in re.findall(r'\\csname %s@([^\\]+)\\endcsname\{(.*)\}' % prefix, read(path)) if '$' in v}


def check_text(root, path, rep, res_keys, fig_keys, labels, refs, res_math=frozenset()):
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
    # \res{} with a $...$ value inside inline math -> nested $ (breaks pdfLaTeX)
    for i, seg in enumerate(re.split(r'(?<!\\)\$', text)):
        if i % 2:
            for k in re.findall(r'\\res\{([^}]+)\}', seg):
                if k in res_math:
                    rep.err('%s: \\res{%s} is inside math mode but its value already has $...$' % (rel, k))
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


CAPTION_CHARS = {'column': 80, 'full': 150}   # ~2 lines of \\footnotesize in a 0.55 column / in the full text width


def slide_captions(root, frame):
    """Plain-text length of each \\caption in a frame, with \\figcaptionshort / \\figcaption / \\res resolved."""
    defs = {}
    for name in ('generated/figure_captions.tex', 'generated/results_macros.tex'):
        name = os.path.relpath(resolve(root, name, []) or os.path.join(root, name), root)
        path = os.path.join(root, name)
        if os.path.exists(path):
            for m in re.finditer(r'\\csname (figcapshort|figcap|res)@([^\\]+)\\endcsname\{(.*)\}\s*$', read(path), flags=re.M):
                defs[(m.group(1), m.group(2))] = m.group(3)
    out = []
    for m in re.finditer(r'\\caption\{', frame):
        i, depth = m.end(), 1
        while i < len(frame) and depth:
            depth += {'{': 1, '}': -1}.get(frame[i], 0)
            i += 1
        cap = frame[m.end():i - 1]
        cap = re.sub(r'\\figcaptionshort\{([^}]*)\}', lambda k: defs.get(('figcapshort', k.group(1)), defs.get(('figcap', k.group(1)), '')), cap)
        cap = re.sub(r'\\figcaption\{([^}]*)\}', lambda k: defs.get(('figcap', k.group(1)), ''), cap)
        cap = re.sub(r'\\res\{([^}]*)\}', lambda k: defs.get(('res', k.group(1)), 'xxxxx'), cap)
        cap = re.sub(r'\\[a-zA-Z]+|[{}$]', '', cap)
        out.append(len('Figura 1: ' + ' '.join(cap.split())))
    return out


def check_slides(files_text, rep, root=None):
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
        if root:
            limit = CAPTION_CHARS['column' if '\\begin{column}' in f else 'full']
            for n in slide_captions(root, f):
                if n > limit:
                    rep.err('slides: frame "%s": caption of ~%d characters, more than ~%d fit in 2 lines; use a shorter '
                            'caption_slides_es' % (name, n, limit))
            if re.search(r'\\(safegraphics|includegraphics)\[[^\]]*\]', f) and not re.search(r'height=[^,\]]*\\textheight', f):
                rep.warn('slides: frame "%s": figure without a height limit (height=...\\textheight, keepaspectratio)' % name)
        if not m:
            rep.err('slides: frame "%s" has no \\speakernote{minutes}{...}' % name)
        else:
            total += float(m.group(1).replace(',', '.'))
    if not 8 <= total <= 11:
        rep.warn('slides: speaker notes add up to %.1f min (target ~10)' % total)
    return len(content), total


def check_confusion_slide(files_text, rep):
    """User rule (DEC-012): the confusion matrix always gets a slide of its own, right before the conclusions."""
    frames = [f for f in re.findall(r'\\begin\{frame\}(.*?)\\end\{frame\}', files_text, flags=re.S) if '\\titlepage' not in f]
    titles = [(re.search(r'^\s*\{([^}]*)\}', f) or re.search(r'()', '')).group(1) for f in frames]
    concl = [i for i, t in enumerate(titles) if t.lower().startswith('conclusiones')]
    if not concl:
        rep.err('slides: no frame titled "Conclusiones..." (the confusion matrix slide goes right before it)')
        return
    i = concl[0]
    prev = frames[i - 1] if i > 0 else ''
    figs = FIG_RE.findall(prev)
    if figs != ['confusion']:
        rep.err('slides: the frame right before "%s" must show only figures/confusion.png (found: %s)'
                % (titles[i], ', '.join(figs) or 'no figure'))


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


FIG_RE = re.compile(r'\\(?:safegraphics|includegraphics)(?:\[[^\]]*\])?\{figures/([^}.]+)')


def figure_sequence(root, main_file):
    """Figure ids in reading order (sections in \\input order), first appearance only."""
    seq = []
    for path in collect(root, main_file, Report()):
        if os.sep + 'sections' + os.sep in path:
            for m in FIG_RE.finditer(strip_comments(read(path))):
                if m.group(1) not in seq:
                    seq.append(m.group(1))
    return seq


def check_parity(root, rep, mains):
    """Report and slides tell the same story: the same figures, in the same order (DEC-010)."""
    rpt, sld = figure_sequence(root, mains['report']), figure_sequence(root, mains['slides'])
    for f in sld:
        if f not in rpt:
            rep.err('figure "%s" is in the slides but not in the report' % f)
    for f in rpt:
        if f not in sld:
            rep.err('figure "%s" is in the report but not in the slides' % f)
    common_r = [f for f in rpt if f in sld]
    common_s = [f for f in sld if f in rpt]
    if common_r != common_s:
        rep.err('report and slides show the figures in a different order:\n    report: %s\n    slides: %s'
                % (', '.join(common_r), ', '.join(common_s)))
    return 'figures (same order in both): %s' % ', '.join(common_s)


def check_report_figures(root, rep, main='report.tex'):
    """Report: every figure float holds at most 2 images (one row) and every figure label is cited with \\ref."""
    text = '\n'.join(strip_comments(read(p)) for p in collect(root, main, Report()))
    for fl in re.findall(r'\\begin\{figure\}.*?\\end\{figure\}', text, flags=re.S):
        n = len(FIG_RE.findall(fl))
        if n > 2:
            rep.err('report: a figure float holds %d images (max. 2 per row)' % n)
        if n == 2 and len(re.findall(r'\\caption\{', fl)) < 2:
            rep.warn('report: two images share one caption; give each its own \\caption and \\label')
    labels = set(re.findall(r'\\label\{(fig:[^}]+)\}', text))
    refs = set(re.findall(r'\\ref\{(fig:[^}]+)\}', text))
    for l in sorted(labels - refs):
        rep.err('report: figure \\label{%s} is never cited with \\ref in the text' % l)


def check_project(root, rep):
    """The single Overleaf project (DEC-011): no folder at its root that would shadow a delivery's folders."""
    for d in ('sections', 'generated', 'figures'):
        if os.path.isdir(os.path.join(root, d)):
            rep.err('%s/ at the project root would shadow the folders of every delivery (\\input@path searches the root first)' % d)


def main(delivery_dir, root=project_paths.PROJECT):
    rep = Report()
    mains = {k: project_paths.main_file(delivery_dir, k) for k in project_paths.KINDS}
    gen = os.path.join(root, project_paths.name(delivery_dir), 'generated')
    check_project(root, rep)
    for kind, main_file in mains.items():
        if os.path.exists(os.path.join(root, main_file)):
            SEARCH_PATH[:] = project_paths.search_path(root, main_file)
            if os.path.join(root, project_paths.name(delivery_dir) + '/') not in SEARCH_PATH:
                rep.err('%s does not add %s/ to \\input@path' % (main_file, project_paths.name(delivery_dir)))
    res_keys = defined_keys(os.path.join(gen, 'results_macros.tex'), 'res')
    fig_keys = defined_keys(os.path.join(gen, 'figure_captions.tex'), 'figcap')
    res_math = math_keys(os.path.join(gen, 'results_macros.tex'), 'res')
    summary = []
    for kind, main_file in mains.items():
        if not os.path.exists(os.path.join(root, main_file)):
            rep.err('%s is missing' % main_file)
            continue
        labels, refs, texts = set(), set(), []
        for path in collect(root, main_file, rep):
            texts.append(check_text(root, path, rep, res_keys, fig_keys, labels, refs, res_math))
        for r in sorted(refs - labels):
            rep.err('%s: \\ref{%s} has no \\label' % (main_file, r))
        allt = '\n'.join(texts)
        body_text = '\n'.join(t for p, t in zip(collect(root, main_file, Report()), texts)
                              if os.sep + 'sections' + os.sep in p)
        if kind == 'slides':
            n, total = check_slides(allt, rep, root)
            check_confusion_slide(allt, rep)
            summary.append('slides: %d content frames, speaker notes %.1f min' % (n, total))
        else:
            summary.append(check_report_length(body_text, rep))
    if all(os.path.exists(os.path.join(root, m)) for m in mains.values()):
        summary.append(check_parity(root, rep, mains))
        check_report_figures(root, rep, mains['report'])
    for tool in ['tectonic', 'latexmk']:
        if shutil.which(tool):
            for main_file in mains.values():
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
    sys.exit(main(sys.argv[1], *sys.argv[2:3]))   # optional 2nd argument: another project root (tests)
