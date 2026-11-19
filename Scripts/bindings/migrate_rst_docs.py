#!/usr/bin/env python3
"""
Migrates element documentation from hand-written reStructuredText pages (e.g. the
usage guide) into '//!' comments in the C++ builder sources, and replaces it in the
page by an `.. xml-element::` directive, which renders it from there again (see
Documentation/gh-pages/source/_ext/xml_bindings.py).

The content of a section up to its first subsection is migrated if it contains exactly
one XML example (`.. code-block:: xml`) whose root element is a known binding and matches
the section heading (or is confirmed with --assign), and no structural directives such as
toctree. Its content is moved as follows:

    - the XML example                    -> @example
    - images                             -> @image (path relative to the repository root)
    - "The parameters are:" + "- name: text" list
                                         -> descriptions of the attributes
    - "An XML example is as follows:"    -> dropped (the directive shows the example)
    - all other text                     -> element description (reStructuredText)

The section heading stays, its content is replaced by the directive. Everything that
cannot be migrated is reported.

Typical use:

    extract.py <source dirs> -o bindings.json
    migrate_rst_docs.py bindings.json <rst files>                  # dry run: plan and problems (-v: details)
    migrate_rst_docs.py bindings.json <rst files> --apply --format
"""

import os
import re
import sys
import argparse
import collections

import bindings_model
from migrate_xml_docs import Plan, applyInsertions

RE_UNDERLINE = re.compile(r'^([=\-~^"\'`#*+.:_])\1{2,}\s*$')
RE_PARAM = re.compile(r'^- ``?(\w+)``?: (.*)$|^- (\w+): (.*)$')
RE_EXAMPLE_INTRO = re.compile(r'^(An|The) XML example is (as follows|below|shown below):?$', re.I)
RE_PARAMS_INTRO = re.compile(r'^The (parameters|attributes) are:$', re.I)


class Section:
    def __init__(self, title, level, start, body_start):
        self.title = title
        self.level = level
        self.start = start            # index of the title line
        self.body_start = body_start  # index after the underline
        self.end = None               # index of the next section's title line (or end of file)
        self.has_subsections = False


def parseSections(lines):
    levels, sections = [], []
    for i in range(1, len(lines)):
        m = RE_UNDERLINE.match(lines[i])
        title = lines[i - 1].strip()
        if not m or not title or RE_UNDERLINE.match(lines[i - 1]) or len(lines[i].strip()) < len(title) - 2:
            continue
        if m.group(1) not in levels:
            levels.append(m.group(1))
        sections.append(Section(title, levels.index(m.group(1)), i - 1, i + 1))
    for k, s in enumerate(sections):
        s.end = sections[k + 1].start if k + 1 < len(sections) else len(lines)
        s.has_subsections = k + 1 < len(sections) and sections[k + 1].level > s.level
    return sections


def blocks(lines):
    """Splits section content into blocks: directives with their indented content, or paragraphs/lists."""
    out, i = [], 0
    while i < len(lines):
        if not lines[i].strip():
            i += 1
            continue
        j = i + 1
        if lines[i].startswith('.. '):
            while j < len(lines) and (not lines[j].strip() or lines[j].startswith((' ', '\t'))):
                j += 1
        else:
            while j < len(lines) and lines[j].strip() and not lines[j].startswith('.. '):
                j += 1
        block = lines[i:j]
        while block and not block[-1].strip():
            block.pop()
        out.append(block)
        i = j
    return out


def normalize(text):
    return re.sub(r'[^a-z0-9]', '', text.lower())


def elementsByTag(bindings):
    by_tag = collections.defaultdict(dict)
    for element in bindings.elements.values():
        for tag, info in element.child_info.items():
            by_tag[tag][info.type] = bindings.elements[info.type]
    return by_tag


def migrateSection(section, lines, path, bindings, by_tag, plan, problems, repo_dir, assignments):
    """Migrates the content of a section up to its first subsection (subsections stay)."""
    location = f"{path}:{section.start + 1}"
    where = f"{location} ({section.title})"
    assigned = assignments.get(location)
    body = blocks(lines[section.body_start:section.end])
    structural = [b[0].split('::')[0][3:] for b in body
                  if re.match(r'\.\. (toctree|include|contents|only|raw)::', b[0].strip())]
    if structural:
        problems['contains structural directives (not migrated)'].append(f"{where}: {', '.join(structural)}")
        return None
    examples = [b for b in body if b[0].strip() in ('.. code-block:: xml', '.. code:: xml')]
    if len(examples) != 1:
        if examples:
            problems['several XML examples'].append(where)
        return None
    code = [l[4:] if l.startswith('    ') else l.lstrip() for l in examples[0][1:] if not l.strip().startswith(':')]
    example = '\n'.join(code).strip('\n')
    m = re.search(r'<\s*([\w:.-]+)', example)
    if not m or m.group(1) not in by_tag:
        problems['unknown example element'].append(f"{where}: <{m.group(1) if m else '?'}>")
        return None
    tag = m.group(1)
    if len(by_tag[tag]) > 1:
        problems['ambiguous tag (several element types)'].append(f"{where}: <{tag}>")
        return None
    element = next(iter(by_tag[tag].values()))
    heading, name = normalize(re.sub(r'\(.*?\)', '', section.title)), normalize(tag)
    if assigned and assigned != tag:
        problems['assigned element does not match the example'].append(f"{where}: example is <{tag}>, assigned <{assigned}>")
        return None
    if not assigned and not name.startswith(heading) and not heading.startswith(name):
        problems['heading does not match the example element'].append(
            f"{where}: example is <{tag}> (confirm with --assign {location}={tag})")
        return None
    if element.description or element.examples or element.images:
        problems['already documented in C++'].append(f"{where}: <{tag}> ({element.name})")
        return None

    def isMeta(block):
        first = block[0].strip()
        return len(block) == 1 and (RE_EXAMPLE_INTRO.match(first) or RE_PARAMS_INTRO.match(first)) \
            or (all(l.startswith(('- ', '  ')) for l in block) and any(RE_PARAM.match(l) for l in block))
    # the example becomes @example if only meta blocks (example intro, parameter list) follow it,
    # otherwise it stays in the description at its position
    after = body[body.index(examples[0]) + 1:]
    inline_example = any(not isMeta(b) for b in after)

    description, attributes, unmatched_params = [], {}, []
    params_next = False
    for block in body:
        first = block[0].strip()
        if block is examples[0]:
            if inline_example:
                description += block + ['']
            continue
        if first.startswith('.. image::') or first.startswith('.. figure::'):
            rel = first.split('::', 1)[1].strip()
            img = os.path.normpath(os.path.join(os.path.dirname(path), rel.lstrip('/')))
            img = os.path.relpath(img, repo_dir)
            width = next((l.split(':width:', 1)[1].strip() for l in block[1:] if ':width:' in l), None)
            description += [f"@image {img}" + (f" {width}" if width else ''), '']
            continue
        if len(block) == 1 and RE_EXAMPLE_INTRO.match(first) and not inline_example:
            continue
        if len(block) == 1 and RE_PARAMS_INTRO.match(first):
            params_next = True
            continue
        if params_next and all(l.startswith('- ') or l.startswith('  ') for l in block):
            params_next = False
            items = []
            for l in block:
                if l.startswith('- '):
                    items.append(l)
                else:
                    items[-1] += ' ' + l.strip()
            for item in items:
                p = RE_PARAM.match(item)
                pname, text = (p.group(1) or p.group(3), p.group(2) or p.group(4)) if p else (None, None)
                if pname and pname in element.attributes:
                    attributes[pname] = text.strip()
                else:
                    unmatched_params.append(item)
            continue
        params_next = False
        # strip a trailing "An XML example is below:" sentence from a paragraph
        if not inline_example and not first.startswith('.. ') and RE_EXAMPLE_INTRO.search(block[-1].strip().split('. ')[-1]):
            last = block[-1].rstrip()
            cut = re.split(r'(?<=[.!?:])\s+(?=(An|The) XML example)', last)[0] if '. ' in last else ''
            block = block[:-1] + ([cut] if cut and not RE_EXAMPLE_INTRO.match(cut.strip()) else [])
            if not block:
                continue
        description += block + ['']
    if unmatched_params:
        description += ['The parameters are:', ''] + unmatched_params + ['']
        problems['parameters kept in the description (not attributes of the element)'].append(
            f"{where}: " + '; '.join(unmatched_params))

    while description and not description[-1].strip():
        description.pop()
    plan.element_docs[element.name] = {'description': '\n'.join(description) or None,
                                       'description_lines': description,
                                       'examples': [] if inline_example else [example],
                                       'images': [], 'source': where}
    for pname, text in attributes.items():
        plan.addRelation(plan.attribute_docs, 'attribute', (element.name, pname), text, where)
    spec = tag
    return spec


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('bindings', metavar='JSON', help='bindings JSON written by extract.py (before migrating)')
    parser.add_argument('rst_files', metavar='RST', nargs='+', help='reStructuredText pages to migrate')
    parser.add_argument('--base-dir', default='.', help='repository root (paths in the JSON are relative to it)')
    parser.add_argument('--assign', metavar='FILE:LINE=TAG', action='append', default=[],
                        help='migrate the section at FILE:LINE (its title line) although its heading does not '
                             'match the example element TAG')
    parser.add_argument('--apply', action='store_true', help='modify the C++ files and the pages')
    parser.add_argument('--format', metavar='CLANG_FORMAT', nargs='?', const='clang-format',
                        help='with --apply: run clang-format on the modified C++ lines')
    parser.add_argument('-v', '--verbose', action='store_true', help='show the planned comments')
    args = parser.parse_args()

    assignments = dict(a.rpartition('=')[::2] for a in args.assign)
    bindings = bindings_model.load(args.bindings)
    by_tag = elementsByTag(bindings)
    plan = Plan(bindings)
    problems = plan.problems
    page_edits = {}
    for path in args.rst_files:
        with open(path, 'rb') as f:
            data = f.read().decode('utf-8')
        newline = '\r\n' if '\r\n' in data else '\n'
        lines = data.split(newline)
        edits = []
        for section in parseSections(lines):
            spec = migrateSection(section, lines, path, bindings, by_tag, plan, problems, args.base_dir, assignments)
            if spec:
                edits.append((section, spec))
        page_edits[path] = (lines, newline, edits)

    comments, splits = plan.insertions(args.base_dir)
    migrated = sum(len(e[2]) for e in page_edits.values())
    print(f"{migrated} sections migrated: {len(plan.element_docs)} element descriptions, "
          f"{len(plan.attribute_docs)} attribute descriptions")
    for path, (_, _, edits) in page_edits.items():
        for section, spec in edits:
            print(f"  {path}:{section.start + 1} ({section.title}) -> <{spec}>")
    if args.verbose:
        for path in sorted(set(comments) | set(splits)):
            for line in sorted(set(comments.get(path, {})) | set(splits.get(path, {}))):
                print(f"\n{path}:{line}")
                for l in comments.get(path, {}).get(line, []):
                    print('    //! ' + l)
                for text, doc in splits.get(path, {}).get(line, []):
                    for l in doc:
                        print('    //! ' + l)
                    print('    ' + text.strip())

    if args.apply:
        for path, (lines, newline, edits) in page_edits.items():
            for section, spec in sorted(edits, key=lambda e: -e[0].start):
                lines[section.body_start:section.end] = ['', f'.. xml-element:: {spec}', '']
            with open(path, 'wb') as f:
                f.write(newline.join(lines).encode('utf-8'))
        changed_ranges = {}
        changed = applyInsertions(comments, splits, args.base_dir, problems, changed_ranges)
        print(f"inserted {changed} comment blocks")
        if args.format:
            import subprocess
            for path, ranges in sorted(changed_ranges.items()):
                subprocess.run([args.format, '-i', '--style=file'] + [f'--lines={a}:{b}' for a, b in ranges]
                               + [os.path.join(args.base_dir, path)], check=True)

    for category, entries in problems.items():
        print(f"\n{category} ({len(entries)}):")
        for e in entries:
            print('  ' + e)


if __name__ == '__main__':
    main()
