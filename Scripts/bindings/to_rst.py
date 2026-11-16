#!/usr/bin/env python3
"""
Renders the bindings JSON as reStructuredText reference pages for Sphinx:
one page per top-level XML section (e.g. <geometry>, <kemfield>, <kassiopeia>),
listing what can go into it, and one page per element (in elements/), showing
its description, child elements, attributes, examples and the elements it can be
used in. The pages form a navigation tree following the XML nesting.

Every element type is documented once, below the first element (breadth-first,
sections in the order given by --page) from which it can be reached.
"""

import os
import re
import argparse
import collections

import bindings_model

# heading underlines (page title, element sections)


def escape(text):
    """Escapes characters with a meaning in reStructuredText inline markup."""
    return re.sub(r'([\\*`_|\[\]])', r'\\\1', text)


def label(element):
    return 'binding-' + re.sub(r'[^A-Za-z0-9]+', '-', element.name).strip('-').lower()


def ref(element, text):
    return f':ref:`{text.replace("`", "")} <{label(element)}>`'


def literal(text):
    return '``' + text + '``' if text else ''


RE_KEY_LINE = re.compile(r'^[\w ()|*.-]+:\s')


def renderText(text, indent=''):
    """Renders a plain-text description: paragraphs are separated by empty lines; within a
    paragraph, lines are joined unless they look like a list of 'key: value' lines (or follow a
    line ending with ':'), in which case line breaks are kept (line block)."""
    out = []
    for paragraph in re.split(r'\n\s*\n', text.strip()):
        lines = [l.strip() for l in paragraph.split('\n') if l.strip()]
        if not lines:
            continue
        keep_breaks = any(RE_KEY_LINE.match(l) for l in lines[1:]) or any(l.endswith(':') for l in lines[:-1])
        if keep_breaks:
            out += [indent + '| ' + escape(l) for l in lines]
        else:
            out.append(indent + escape(' '.join(lines)))
        out.append('')
    return out[:-1] if out else []


def summary(text):
    """First sentence of a description."""
    if not text:
        return ''
    first = re.split(r'(?<=[.!?])\s|\n\s*\n', text.strip(), maxsplit=1)[0]
    return ' '.join(first.split())


def sourceLink(location, base_url):
    path, _, line = location.rpartition(':')
    return f'`{os.path.basename(path)}:{line} <{base_url}{path}#L{line}>`__'


class Page:
    def __init__(self, name, title, roots):
        self.name = name
        self.title = title
        self.roots = roots        # (tag, Element) of top-level elements on this page


class Tree:
    """Navigation tree: every element type gets its own document, placed below the element from
    which it is first reached (breadth-first from the top-level elements of each page). Elements
    that can be used in several places are linked from all of them, but listed in the navigation
    only once."""
    def __init__(self):
        self.pages = []
        self.page_of = {}         # C++ type -> Page
        self.children = {}        # C++ type or page name -> [(tag, Element)] in the navigation tree
        self.titles = {}          # C++ type -> title (tag, disambiguated if needed)
        self.single_root = {}     # page name -> root Element documented on the page itself


def buildTree(bindings, page_specs):
    root = bindings.elements[bindings_model.MASTER_ROOT]
    tree = Tree()
    remaining = dict(root.children)
    for name, title, tags in page_specs:
        roots = [(t, remaining.pop(t)) for t in tags if t in remaining]
        if roots:
            tree.pages.append(Page(name, title, roots))
    if remaining:
        tree.pages.append(Page('other', 'Other top-level elements', sorted(remaining.items())))

    reached_via = {}
    for page in tree.pages:
        queue = collections.deque((tag, element, page.name, None) for tag, element in page.roots)
        while queue:
            tag, element, parent_key, parent_tag = queue.popleft()
            if element.name in tree.page_of:
                continue
            tree.page_of[element.name] = page
            tree.children.setdefault(parent_key, []).append((tag, element))
            reached_via[element.name] = (tag, parent_tag)
            for child_tag, child in sorted(element.children.items()):
                if child.name not in tree.page_of:
                    queue.append((child_tag, child, element.name, tag))
        if len(page.roots) == 1:
            tree.single_root[page.name] = page.roots[0][1]

    # the same tag can be bound to different types depending on the parent element (e.g.
    # zonal_harmonic_field_solver for electric and magnetic fields): add the parent to such titles
    count = collections.Counter(tag for tag, _ in reached_via.values())
    for name, (tag, parent_tag) in reached_via.items():
        tree.titles[name] = f'{tag} (in {parent_tag})' if count[tag] > 1 and parent_tag else tag
    count = collections.Counter(tree.titles.values())
    for name, title in tree.titles.items():
        if count[title] > 1:
            tree.titles[name] = f'{title} ({name})'
    for key in tree.children:
        tree.children[key].sort(key=lambda e: tree.titles[e[1].name])
    return tree


def docName(element):
    return 'elements/' + label(element)[len('binding-'):]


def titleMarkup(title):
    """'tag (in parent)' -> '``tag`` (in ``parent``)'"""
    m = re.match(r'^(\w+)(?: \(in (\w+)\))?(?: \((.+)\))?$', title)
    if not m:
        return literal(title)
    out = literal(m.group(1))
    if m.group(2):
        out += f' (in {literal(m.group(2))})'
    if m.group(3):
        out += f' ({literal(m.group(3))})'
    return out


def cellLines(cell):
    return ['      - ' + cell[0]] + [('        ' + l if l else '') for l in cell[1:]] if cell else ['      -']


def toctree(entries, prefix):
    if not entries:
        return []
    return ['.. toctree::', '    :hidden:', ''] + [f'    {prefix}{docName(e)}' for _, e in entries] + ['']


def renderElement(tag, element, bindings, base_url, tree, title=None, intro=None):
    """Renders the document of one element. `title` replaces the heading (used for the section pages)."""
    heading = title or titleMarkup(tree.titles.get(element.name, tag))
    out = [f'.. _{label(element)}:', '', heading, '=' * len(heading), '']
    out += intro or []

    if element.deprecated is not None:
        out += ['.. warning::', '', '    **Deprecated.** ' + escape(' '.join(element.deprecated.split())), '']

    if element.description:
        out += renderText(element.description) + ['']
    elif not intro:
        out += ['*No description available.*', '']

    facts = [f'XML element: {literal("<" + tag + ">")}', f'C++ type: {literal(element.name)}']
    other_tags = [t for t in element.tags if t != tag]
    if other_tags:
        facts.append('Also available as: ' + ', '.join(literal(t) for t in other_tags))
    if element.defined_at:
        facts.append('Defined in: ' + sourceLink(element.defined_at, base_url))
    out += ['| ' + f for f in facts] + ['']

    if element.child_info:
        out += ['.. list-table:: Child elements', '    :header-rows: 1', '    :widths: 30 70', '',
                '    * - Element', '      - Description']
        for child_tag, info in sorted(element.child_info.items()):
            child = bindings.elements[info.type]
            out.append(f'    * - {ref(child, child_tag)}')
            cell = []
            if info.deprecated is not None or child.deprecated is not None:
                text = info.deprecated if info.deprecated is not None else child.deprecated
                cell += ['**Deprecated:** ' + escape(' '.join(text.split())), '']
            if info.description:
                cell += renderText(info.description)
            elif child.description:
                cell += [escape(summary(child.description))]
            out += cellLines(cell)
        out.append('')

    if element.attribute_info:
        out += ['.. list-table:: Attributes', '    :header-rows: 1', '    :widths: 20 15 65', '',
                '    * - Attribute', '      - Type', '      - Description']
        for name, info in sorted(element.attribute_info.items()):
            out += [f'    * - {literal(name)}', f'      - {literal(info.type)}']
            cell = []
            if info.deprecated is not None:
                cell += ['**Deprecated:** ' + escape(' '.join(info.deprecated.split())), '']
            cell += renderText(info.description) if info.description else []
            out += cellLines(cell)
        out.append('')

    for i, example in enumerate(element.examples):
        out += ['**Example:**' if len(element.examples) == 1 else f'**Example {i + 1}:**', '',
                '.. code-block:: xml', '']
        out += ['    ' + l if l.strip() else '' for l in example.split('\n')]
        out.append('')

    # where the element can be used (grouped by parent element)
    usages = []
    for parent in sorted(element.parents, key=lambda p: (p.tags[0] if p.tags else '', p.name)):
        tags = []
        for child_tag, info in sorted(parent.child_info.items()):
            if info.type != element.name:
                continue
            text = literal('<' + child_tag + '>')
            if info.deprecated is not None:
                text += ' (**deprecated:** ' + escape(' '.join(info.deprecated.split())) + ')'
            tags.append(text)
        if not tags:
            continue
        if parent.name == bindings_model.MASTER_ROOT:
            where = 'at the top level'
        else:
            where = 'in ' + ref(parent, f'<{parent.tags[0] if parent.tags else parent.name}>')
        usages.append(f"{', '.join(tags)} {where}")
    if usages:
        out += ['**Used as:**', ''] + ['- ' + u for u in usages] + ['']
    return out


def writeDocs(tree, bindings, base_url, output_dir):
    os.makedirs(os.path.join(output_dir, 'elements'), exist_ok=True)

    def write(name, lines):
        with open(os.path.join(output_dir, name + '.rst'), 'w') as f:
            f.write('\n'.join(lines).rstrip() + '\n')

    for page in tree.pages:
        root = tree.single_root.get(page.name)
        if root is not None:
            # the page documents its root element directly and lists what can go into it
            tag = page.roots[0][0]
            intro = [f'This page documents the {literal("<" + tag + ">")} section of the configuration. '
                     'Each child element links to its own page, which in turn lists what can go into it.', '']
            lines = renderElement(tag, root, bindings, base_url, tree, title=page.title, intro=intro)
            lines += toctree(tree.children.get(root.name, []), '')
        else:
            lines = [f'.. _bindings-{page.name}:', '', page.title, '=' * len(page.title), '',
                     '.. list-table:: Elements', '    :header-rows: 1', '    :widths: 30 70', '',
                     '    * - Element', '      - Description']
            for tag, element in page.roots:
                lines += [f'    * - {ref(element, tag)}'] + cellLines([escape(summary(element.description))]
                                                                   if element.description else [])
            lines += [''] + toctree(tree.children.get(page.name, []), '')
        write(page.name, lines)

    for name, page in tree.page_of.items():
        element = bindings.elements[name]
        if tree.single_root.get(page.name) is element:
            continue
        tag = tree.titles[name].split(' ')[0]
        lines = renderElement(tag, element, bindings, base_url, tree)
        lines += toctree(tree.children.get(name, []), '/bindings_reference/')
        write(docName(element), lines)


DEFAULT_PAGES = [
    ('geometry', 'Geometry (KGeoBag)', ['geometry', 'geometry_printer']),
    ('kemfield', 'Fields (KEMField)', ['kemfield']),
    ('kassiopeia', 'Simulation (Kassiopeia)', ['kassiopeia']),
]


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('input', metavar='JSON', type=str,
                        help="bindings JSON file written by extract.py ('-' for standard input)")
    parser.add_argument('-o', '--output-dir', metavar='DIR', required=True,
                        help='directory to write one .rst file per page into')
    parser.add_argument('-b', '--base-url', metavar='URL', type=str, default="",
                        help='base url for links to source files (e.g. GitHub tree)')
    parser.add_argument('--page', metavar='NAME:TITLE:TAG[,TAG...]', action='append',
                        help='a page for the given top-level tags (repeatable; default: geometry, kemfield, kassiopeia); '
                             'remaining top-level elements go to an additional page')
    args = parser.parse_args()

    specs = DEFAULT_PAGES
    if args.page:
        specs = [(n, t, tags.split(',')) for n, t, tags in (p.split(':', 2) for p in args.page)]

    bindings = bindings_model.load(args.input)
    tree = buildTree(bindings, specs)
    writeDocs(tree, bindings, args.base_url, args.output_dir)
    with open(os.path.join(args.output_dir, 'pages.txt'), 'w') as f:
        f.write('\n'.join(page.name for page in tree.pages) + '\n')
