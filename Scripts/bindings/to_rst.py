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
import sys
import argparse
import textwrap
import collections

import bindings_model

# image paths in the documentation comments are relative to the repository root; images must be
# inside the Sphinx source directory to be shown
DOCS_SOURCE = 'Documentation/gh-pages/source'


def label(element):
    return 'binding-' + re.sub(r'[^A-Za-z0-9]+', '-', element.name).strip('-').lower()


def ref(element, text):
    return f':ref:`{text.replace("`", "")} <{label(element)}>`'


def literal(text):
    return '``' + text + '``' if text else ''


def renderText(text, indent=''):
    """Documentation texts are reStructuredText: they are passed through, indented as needed."""
    return [(indent + l) if l.strip() else '' for l in textwrap.dedent(text.strip('\n')).split('\n')]


def summary(text):
    """First sentence of a description (as a single line)."""
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


def imagePath(path, docs_source=DOCS_SOURCE):
    """Path of an image for the Sphinx image directive (absolute paths are relative to the source dir)."""
    rel = os.path.relpath(path, docs_source)
    if rel.startswith('..'):
        print(f"warning: image {path} is outside of {docs_source} and cannot be shown", file=sys.stderr)
        return None
    return '/' + rel.replace(os.sep, '/')


def deprecationText(text):
    return ' '.join(text.split())


# references to more element types than this are only named (e.g. KSComponent, the base of all
# Kassiopeia objects), not listed
MAX_LISTED_REFERENCES = 40

GEOMETRY_TARGETS = {'surfaces': 'KGeoBag surfaces', 'spaces': 'KGeoBag spaces',
                    'surface': 'a KGeoBag surface', 'space': 'a KGeoBag space'}


def className(type_name):
    return re.sub(r'<.*', '', type_name).split('::')[-1]


def primaryTag(element, bindings):
    """The tag under which an element is usually defined: one registered directly in a top-level
    section (e.g. <ksgen_value_set> in <kassiopeia> rather than <energy_set> in a composite)."""
    root = bindings.elements.get(bindings_model.MASTER_ROOT)
    sections = {e.name for e in root.children.values()} if root else set()
    for parent in element.parents:
        if parent.name in sections:
            for tag, info in sorted(parent.child_info.items()):
                if info.type == element.name:
                    return tag
    return element.tags[0] if element.tags else element.name


def reachableElements(bindings):
    """Elements that can be used in a configuration (reachable from the top level), i.e. those with a page."""
    if '_reachable' not in bindings.__dict__:
        reachable, todo = set(), [bindings.elements[bindings_model.MASTER_ROOT]]
        while todo:
            element = todo.pop()
            for child in element.children.values():
                if child.name not in reachable:
                    reachable.add(child.name)
                    todo.append(child)
        bindings.__dict__['_reachable'] = reachable
    return bindings.__dict__['_reachable']


def referableElements(bindings, type_name):
    """Elements creating objects of the given C++ type (or a derived one), which can therefore be
    referenced by name where an object of that type is expected."""
    cache = bindings.__dict__.setdefault('_referable', {})
    if type_name not in cache:
        cls = className(type_name)
        reachable = reachableElements(bindings)
        cache[type_name] = sorted((e for n, e in bindings.elements.items()
                                   if n in reachable and (className(n) == cls or cls in e.bases)),
                                  key=lambda e: primaryTag(e, bindings))
    return cache[type_name]


def referenceText(reference, bindings):
    """reStructuredText describing what a reference attribute names."""
    if 'geometry' in reference:
        target = GEOMETRY_TARGETS.get(reference['geometry'], reference['geometry'])
        return f'**Refers to** {target} (:ref:`path <kgeobag-geometry-paths>` in the geometry).'
    elements = referableElements(bindings, reference['type'])
    if not elements:
        return f'**Refers to** the name of a {literal(reference["type"])} object.'
    if len(elements) > MAX_LISTED_REFERENCES:
        return f'**Refers to** the name of any {literal(reference["type"])} object.'
    links = ', '.join(ref(e, '<' + primaryTag(e, bindings) + '>') for e in elements)
    return f'**Refers to** the name of one of: {links}.'


def referencingAttributes(bindings, element):
    """(attribute name, referencing element) for attributes that can refer to this element by name."""
    index = bindings.__dict__.get('_referenced_by')
    if index is None:
        index = bindings.__dict__.setdefault('_referenced_by', {})
        reachable = reachableElements(bindings)
        for owner in bindings.elements.values():
            if owner.name not in reachable:
                continue
            for name, info in owner.attribute_info.items():
                if info.reference and 'type' in info.reference:
                    targets = referableElements(bindings, info.reference['type'])
                    if len(targets) <= MAX_LISTED_REFERENCES:
                        for target in targets:
                            index.setdefault(target.name, []).append((name, owner))
    return sorted(index.get(element.name, []), key=lambda r: (primaryTag(r[1], bindings), r[0]))


def renderElementBody(tag, element, bindings, base_url, show_description=True, usage_links=False, more=None):
    """Renders the documentation of one element without title: description, images, facts, child
    elements, attributes, examples and where it can be used. Used by the reference pages and by the
    xml-element directive of the Sphinx extension (Documentation/gh-pages/source/_ext/xml_bindings.py)."""
    out = []
    if element.deprecated is not None:
        out += ['.. warning::', '', '    **Deprecated.** ' + deprecationText(element.deprecated), '']

    if show_description:
        description = renderText(element.description) if element.description else []
        # images are shown at their position in the description
        for image in sorted(element.images, key=lambda i: -i.get('line', len(description))):
            path = imagePath(image['path'])
            if path:
                at = min(image.get('line', len(description)), len(description))
                description[at:at] = ['', f'.. image:: {path}'] + \
                    ([f'    :width: {image["width"]}'] if image.get('width') else []) + ['']
        out += description + ['']

    facts = [f'XML element: {literal("<" + tag + ">")}', f'C++ type: {literal(element.name)}']
    other_tags = [t for t in element.tags if t != tag]
    if other_tags:
        facts.append('Also available as: ' + ', '.join(literal(t) for t in other_tags))
    if element.defined_at:
        facts.append('Defined in: ' + sourceLink(element.defined_at, base_url))
    out += ['| ' + f for f in facts] + ['']

    if element.child_info:
        out += ['.. list-table:: Child elements', '    :header-rows: 1', '    :widths: 30 70', '    :class: xml-bindings-table', '',
                '    * - Element', '      - Description']
        for child_tag, info in sorted(element.child_info.items()):
            child = bindings.elements[info.type]
            out.append(f'    * - {ref(child, child_tag)}')
            cell = []
            if info.deprecated is not None or child.deprecated is not None:
                text = info.deprecated if info.deprecated is not None else child.deprecated
                cell += ['**Deprecated:** ' + deprecationText(text), '']
            if info.description:
                cell += renderText(info.description)
            elif child.description:
                cell += [summary(child.description)]
            out += cellLines(cell)
        out.append('')

    if element.attribute_info:
        out += ['.. list-table:: Attributes', '    :header-rows: 1', '    :widths: 20 15 65', '    :class: xml-bindings-table', '',
                '    * - Attribute', '      - Type', '      - Description']
        for name, info in sorted(element.attribute_info.items()):
            out += [f'    * - {literal(name)}', f'      - {literal(info.type)}']
            cell = []
            if info.deprecated is not None:
                cell += ['**Deprecated:** ' + deprecationText(info.deprecated), '']
            cell += renderText(info.description) if info.description else []
            if info.reference:
                cell += ([''] if cell else []) + [referenceText(info.reference, bindings)]
            if info.values:
                values = ', '.join(literal(v) for v in info.values['values'])
                cell += ([''] if cell else []) + [f'**Values:** {values}' + (', or others.' if info.values.get('open') else '.')]
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
                text += ' (**deprecated:** ' + deprecationText(info.deprecated) + ')'
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

    # attributes of other elements naming objects of this element
    referenced = referencingAttributes(bindings, element)
    if referenced:
        out += ['**Can be referenced by:**', '']
        out += [f'- {literal(name)} of {ref(owner, "<" + primaryTag(owner, bindings) + ">")}' for name, owner in referenced]
        out.append('')

    # links to further documentation and examples (with usage_links, the Sphinx extension adds the
    # heading itself if there are any)
    links = list(more or [])
    if links:
        out += ['.. rubric:: More information', ''] + ['| ' + l for l in links] + ['']
    if usage_links:
        out += [f'.. xml-element-usages:: {element.name}', '']
    return out


def renderElement(tag, element, bindings, base_url, tree, title=None, intro=None, usage_links=True):
    """Renders the reference document of one element. `title` replaces the heading (used for the section pages)."""
    heading = title or titleMarkup(tree.titles.get(element.name, tag))
    out = [f'.. _{label(element)}:', '', heading, '=' * len(heading), '']
    out += intro or []
    return out + renderElementBody(tag, element, bindings, base_url, usage_links=usage_links)


def writeDocs(tree, bindings, base_url, output_dir, usage_links=True):
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
            lines = renderElement(tag, root, bindings, base_url, tree, title=page.title, intro=intro, usage_links=usage_links)
            lines += toctree(tree.children.get(root.name, []), '')
        else:
            lines = [f'.. _bindings-{page.name}:', '', page.title, '=' * len(page.title), '',
                     '.. list-table:: Elements', '    :header-rows: 1', '    :widths: 30 70', '    :class: xml-bindings-table', '',
                     '    * - Element', '      - Description']
            for tag, element in page.roots:
                lines += [f'    * - {ref(element, tag)}'] + cellLines([summary(element.description)]
                                                                   if element.description else [])
            lines += [''] + toctree(tree.children.get(page.name, []), '')
        write(page.name, lines)

    for name, page in tree.page_of.items():
        element = bindings.elements[name]
        if tree.single_root.get(page.name) is element:
            continue
        tag = tree.titles[name].split(' ')[0]
        lines = renderElement(tag, element, bindings, base_url, tree, usage_links=usage_links)
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
    parser.add_argument('--no-usage-links', action='store_true',
                        help='do not emit xml-element-usages directives (only for use without the Sphinx extension)')
    parser.add_argument('--page', metavar='NAME:TITLE:TAG[,TAG...]', action='append',
                        help='a page for the given top-level tags (repeatable; default: geometry, kemfield, kassiopeia); '
                             'remaining top-level elements go to an additional page')
    args = parser.parse_args()

    specs = DEFAULT_PAGES
    if args.page:
        specs = [(n, t, tags.split(',')) for n, t, tags in (p.split(':', 2) for p in args.page)]

    bindings = bindings_model.load(args.input)
    tree = buildTree(bindings, specs)
    writeDocs(tree, bindings, args.base_url, args.output_dir, not args.no_usage_links)
    with open(os.path.join(args.output_dir, 'pages.txt'), 'w') as f:
        f.write('\n'.join(page.name for page in tree.pages) + '\n')
