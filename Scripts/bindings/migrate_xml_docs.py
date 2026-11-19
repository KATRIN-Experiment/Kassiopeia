#!/usr/bin/env python3
"""
Migrates documentation from annotated XML example files ("Complete XMLs") into
'//!' comments in the C++ builder sources, where extract.py picks them up.

The XML files contain example elements, each followed by a comment like

    <ksgen_value_set name="value_set" value_start="1." value_stop="2." value_count="11"/>
    <!--
        description:
            generates a set of constant values ...

        parameters:
            value_start:
                the beginning of the interval.

        elements:
            some_child:
                what the child element does.

                parameters:
                    attribute_of_child:
                        ...

        example:
            <more xml/>
    -->

Each comment is attached to the preceding element or one nested inside it
(whichever matches the documented parameters/elements best), the element's
C++ type is determined from its position in the binding tree, and the texts
are inserted as '//!' comments above the builder typedef (element description,
the example element as @example) and above the attribute/child registrations
(see bindings_model.py for the comment format).

Typical use:

    extract.py <source dirs> -o bindings.json
    migrate_xml_docs.py bindings.json <xml files>            # dry run: shows the plan and all problems
    migrate_xml_docs.py bindings.json <xml files> --apply [--format]   # inserts the comments (and runs clang-format)
    extract.py <source dirs> -o bindings_new.json
    migrate_xml_docs.py bindings.json <xml files> --check bindings_new.json

Paths in the bindings JSON are relative to the directory extract.py was run
in; run this script from the same directory (or use --base-dir).
"""

import os
import re
import sys
import json
import argparse
import textwrap
import difflib
import collections

import bindings_model
from extract import REGEX_PATTERN

# Preprocessor elements of KXMLInitializer: children of "transparent" ones belong to the enclosing element,
# "ignored" ones are skipped entirely.
DEFAULT_TRANSPARENT = ['tag', 'if', 'else', 'loop']
DEFAULT_IGNORED = ['define', 'undefine', 'redefine', 'append', 'prepend', 'include', 'print', 'warning', 'error', 'assert',
                   'global_define', 'global_undefine', 'global_redefine', 'global_append', 'global_prepend',
                   'external_define', 'external_undefine', 'external_redefine', 'external_append', 'external_prepend',
                   'local_define']

SECTIONS = {'description': 'description', 'parameters': 'parameters', 'elements': 'elements',
            'example': 'example', 'examples': 'example'}


###############################################################################
# XML dialect parsing (mirrors KXMLTokenizer: comments may be nested)

class Node:
    def __init__(self, kind, name='', line=0, start=0):
        self.kind = kind            # 'element' or 'comment'
        self.name = name
        self.line = line
        self.start = start          # offset of '<' in the file
        self.end = start            # offset after the element / comment
        self.attrs = {}
        self.children = []
        self.text = ''
        self.parent = None
        self.type = None            # bindings_model.Element, if resolved

    def elements(self):
        return [c for c in self.children if c.kind == 'element']


RE_TAG_NAME = re.compile(r'<\s*([\w:.-]+)')
RE_ATTRIBUTE = re.compile(r'\s*([\w:.-]+)\s*=\s*("[^"]*"|\'[^\']*\')')
RE_TAG_END = re.compile(r'\s*(/?)>')


def parseXML(path):
    with open(path, errors='replace') as f:
        text = f.read()
    root = Node('element', '#document')
    root.end = len(text)
    current = root
    i, line, n = 0, 1, len(text)
    while i < n:
        if text.startswith('<!--', i):
            depth, j = 1, i + 4
            while j < n and depth:
                if text.startswith('<!--', j):
                    depth, j = depth + 1, j + 4
                elif text.startswith('-->', j):
                    depth, j = depth - 1, j + 3
                else:
                    j += 1
            node = Node('comment', line=line, start=i)
            node.text, node.end, node.parent = text[i+4:j-3], j, current
            current.children.append(node)
        elif text.startswith('</', i):
            j = text.index('>', i) + 1
            name = text[i+2:j-1].strip()
            if current.name != name:
                print(f"{path}:{line}: warning: </{name}> closes <{current.name}>", file=sys.stderr)
            current.end = j
            current = current.parent or root
        elif text.startswith('<?', i) or text.startswith('<!', i):
            j = text.index('>', i) + 1
        elif text[i] == '<':
            m = RE_TAG_NAME.match(text, i)
            j, attrs = m.end(), {}
            while True:
                a = RE_ATTRIBUTE.match(text, j)
                if not a:
                    break
                attrs[a.group(1)] = a.group(2)[1:-1]
                j = a.end()
            e = RE_TAG_END.match(text, j)
            if not e:
                raise ValueError(f"{path}:{line}: cannot parse tag {text[i:i+60]!r}")
            j = e.end()
            node = Node('element', m.group(1), line, i)
            node.attrs, node.parent, node.end = attrs, current, j
            current.children.append(node)
            if not e.group(1):
                current = node
        else:
            j = text.find('<', i)
            j = n if j < 0 else j
        line += text.count('\n', i, j)
        i = j
    return root, text


def resolveTypes(node, context, transparent, ignored, problems, path):
    """Assigns bindings_model.Element types along the XML tree."""
    for child in node.elements():
        if child.name in transparent:
            resolveTypes(child, context, transparent, ignored, problems, path)
            continue
        if child.name in ignored:
            continue
        child.type = context.children.get(child.name) if context else None
        if child.type is None and context is not None:
            problems['unknown elements'].append(f"{path}:{child.line}: <{child.name}> is not a registered child of {context.name}")
        resolveTypes(child, child.type, transparent, ignored, problems, path)


def descendants(node, transparent):
    yield node
    for child in node.elements():
        yield from descendants(child, transparent)


###############################################################################
# documentation comment parsing

RE_KEY = re.compile(r'^([\w()|]+):\s*(.*)$')


def dedent(lines):
    while lines and not lines[0].strip():
        lines = lines[1:]
    while lines and not lines[-1].strip():
        lines = lines[:-1]
    return textwrap.dedent('\n'.join(lines)).strip('\n')


def parseKeyedBlock(lines):
    """Parses lines of 'key:' entries (keys at the smallest indentation) into {key: [lines]}.
    Text directly after 'key:' on the same line is kept."""
    entries = collections.OrderedDict()
    key_indent, current = None, None
    for line in lines:
        indent = len(line) - len(line.lstrip())
        m = RE_KEY.match(line.strip())
        if m and line.strip() and (key_indent is None or indent <= key_indent):
            key_indent = indent
            current = entries.setdefault(m.group(1), [])
            if m.group(2):
                current.append(' ' * (indent + 4) + m.group(2))
        elif current is not None:
            current.append(line)
    return entries


def parseDocComment(text):
    """Returns {'description', 'parameters': {name: text}, 'elements': {name: {'description', 'parameters'}},
    'examples': [text]} or None if the comment is not a documentation comment."""
    lines = [l.expandtabs(4).rstrip() for l in text.split('\n')]
    sections = collections.defaultdict(list)
    current = None
    found = False
    for line in lines:
        m = RE_KEY.match(line.strip())
        if m and m.group(1) in SECTIONS:
            indent = len(line) - len(line.lstrip())
            # a nested "parameters:" inside an elements entry belongs to that entry
            if current == 'elements' and m.group(1) == 'parameters' and sections['elements'] \
                    and indent > min(len(l) - len(l.lstrip()) for l in sections['elements'] if l.strip()):
                sections[current].append(line)
                continue
            current = SECTIONS[m.group(1)]
            found = found or current == 'description'
            if m.group(2) and m.group(2).strip() != 'none':
                sections[current].append(' ' * (indent + 4) + m.group(2))
            continue
        if current:
            sections[current].append(line)
    if not found:
        return None

    parameters = {k: dedent(v) for k, v in parseKeyedBlock(sections['parameters']).items()}
    elements = {}
    for name, entry_lines in parseKeyedBlock(sections['elements']).items():
        description, nested = [], []
        target = description
        for line in entry_lines:
            m = RE_KEY.match(line.strip())
            if m and m.group(1) == 'parameters':
                target = nested
                continue
            target.append(line)
        elements[name] = {'description': dedent(description),
                          'parameters': {k: dedent(v) for k, v in parseKeyedBlock(nested).items()}}
    examples = [dedent(sections['example'])] if dedent(sections['example']) else []
    return {'description': dedent(sections['description']), 'parameters': parameters,
            'elements': elements, 'examples': examples}


def namePattern(name):
    """Documented names may be patterns like '(x|y|z)_(fix|set)'."""
    if '(' in name or '|' in name:
        return re.compile('^' + name + '$')
    return re.compile('^' + re.escape(name) + '$')


def matchNames(name, available):
    pattern = namePattern(name)
    return [a for a in available if pattern.match(a)]


###############################################################################
# attaching comments to elements

class DocComment:
    def __init__(self, path, node, previous, doc):
        self.path = path
        self.node = node            # the comment node
        self.previous = previous    # preceding XML element
        self.doc = doc
        self.target = None          # XML element node the comment documents
        self.how = None
        self.candidates = []

    @property
    def location(self):
        return f"{self.path}:{self.node.line}"


def score(candidate, doc):
    attributes = set(candidate.type.attributes)
    children = set(candidate.type.children)
    s = sum(1 for p in doc['parameters'] if matchNames(p, attributes))
    s += sum(1 for e in doc['elements'] if matchNames(e, children))
    return s


def attachComments(comments, assignments, transparent):
    """Decides the documented element for each comment (in several passes, see below)."""
    for c in comments:
        c.candidates = [d for d in descendants(c.previous, transparent) if d.type is not None]
        hint = assignments.get(c.location) or next((v for k, v in assignments.items() if c.location.endswith(k)), None)
        if hint:
            matching = [d for d in c.candidates if d.name == hint]
            if matching:
                c.target, c.how = matching[0], 'assigned'
            else:
                c.how = f"assigned tag <{hint}> not found"

    # pass 1: unique best match; later passes: prefer candidates whose type no other comment documents
    documented = {c.target.type.name for c in comments if c.target is not None}
    changed, first = True, True
    while changed:
        changed = False
        for c in comments:
            if c.target is not None or (c.how and c.how.startswith('assigned')):
                continue
            candidates = c.candidates
            if not first:
                undocumented = [d for d in candidates if d.type.name not in documented]
                candidates = undocumented or candidates
            if not candidates:
                continue
            documented_names = bool(c.doc['parameters'] or c.doc['elements'])
            scored = sorted(((score(d, c.doc), -i, d) for i, d in enumerate(candidates)), key=lambda t: t[:2], reverse=True)
            best = scored[0]
            unique = len(scored) == 1 or scored[1][0] < best[0]
            if documented_names and best[0] == 0:
                continue
            if unique or (not first and documented_names and len([s for s in scored if s[0] == best[0]]) == 1):
                c.target, c.how = best[2], 'matched' if first else 'inferred'
            elif not first and len(candidates) < len(c.candidates) and documented_names:
                # several equally good undocumented candidates: take the outermost
                c.target, c.how = best[2], 'inferred (outermost)'
            else:
                continue
            documented.add(c.target.type.name)
            changed = True
        if first:
            first, changed = False, True


###############################################################################
# planning the C++ comments

class Plan:
    def __init__(self, bindings):
        self.bindings = bindings
        self.element_docs = collections.OrderedDict()    # type -> {'description', 'examples', 'source'}
        self.attribute_docs = collections.OrderedDict()  # (type, name) -> (text, source)
        self.child_docs = collections.OrderedDict()      # (parent type, tag) -> (text, source)
        self.problems = collections.defaultdict(list)

    @staticmethod
    def same(a, b):
        return re.sub(r'\s+', ' ', a or '').strip() == re.sub(r'\s+', ' ', b or '').strip()

    def addElement(self, element, description, examples, source):
        if not description and not examples:
            return
        existing = self.element_docs.get(element.name)
        if existing:
            if description and not self.same(existing['description'], description):
                # e.g. several forms of the same element, each documented with its own example
                existing['description'] = (existing['description'] + '\n\n' + description).strip()
                self.problems['merged descriptions (please review)'].append(
                    f"{source}: appended to the description of {element.name} from {existing['source']}")
            for e in examples:
                if all(not self.same(e, x) for x in existing['examples']):
                    existing['examples'].append(e)
            return
        self.element_docs[element.name] = {'description': description, 'examples': list(examples), 'source': source}

    def addRelation(self, store, kind, key, text, source):
        if not text:
            return
        if key in store:
            if not self.same(store[key][0], text):
                self.problems['conflicting descriptions'].append(
                    f"{source}: {kind} '{key[1]}' of {key[0]} is already described at {store[key][1]}; this description is not migrated")
            return
        store[key] = (text, source)

    def addComment(self, c, xml_text):
        element = c.target.type
        source = c.location
        example = exampleSnippet(c.target, xml_text)
        description = c.doc['description']
        # entries of "elements:" that are not child tags (e.g. categories like "integrators") go into the description
        unmatched = [(name, entry) for name, entry in c.doc['elements'].items() if not matchNames(name, element.children)]
        if unmatched:
            description = (description + '\n\nchild elements:\n' + '\n'.join(
                f"{name}: {entry['description']}" for name, entry in unmatched)).strip()
            self.problems['folded into the description (please review)'].append(
                f"{source}: {', '.join(n for n, _ in unmatched)} are not child tags of <{c.target.name}> ({element.name})")
        self.addElement(element, description, [example] + c.doc['examples'], source)
        for name, text in c.doc['parameters'].items():
            names = matchNames(name, element.attributes)
            if not names:
                self.problems['unknown attributes'].append(f"{source}: '{name}' is not an attribute of <{c.target.name}> ({element.name})")
            for n in names:
                self.addRelation(self.attribute_docs, 'attribute', (element.name, n), text, source)
        for name, entry in c.doc['elements'].items():
            for tag in matchNames(name, element.children):
                self.addRelation(self.child_docs, 'child', (element.name, tag), entry['description'], source)
                child = element.children[tag]
                for pname, text in entry['parameters'].items():
                    pnames = matchNames(pname, child.attributes)
                    if not pnames:
                        self.problems['unknown attributes'].append(f"{source}: '{pname}' is not an attribute of <{tag}> ({child.name})")
                    for n in pnames:
                        self.addRelation(self.attribute_docs, 'attribute', (child.name, n), text, source)

    def insertions(self, base_dir):
        """Returns ({path: {line: [comment lines]}}, {path: {line: [(registration text, [comment lines])]}}):
        comments to insert above a line, and lines with several registrations to split into one
        registration per line (each with its comment). Records problems for docs that cannot be placed."""
        locations = collections.Counter()
        for element in self.bindings.elements.values():
            for info in list(element.attribute_info.values()) + list(element.child_info.values()):
                locations[info.registered_at] += 1

        by_location = collections.defaultdict(dict)    # location -> {(kind, name): lines}
        for (type_name, name), (text, source) in self.attribute_docs.items():
            info = self.bindings.elements[type_name].attribute_info[name]
            by_location[info.registered_at][('Attribute', type_name, name)] = docLines('', text)
        for (type_name, tag), (text, source) in self.child_docs.items():
            info = self.bindings.elements[type_name].child_info[tag]
            by_location[info.registered_at][('Element', type_name, tag)] = docLines('', text)

        comments = collections.defaultdict(lambda: collections.defaultdict(list))
        splits = collections.defaultdict(dict)
        element_lines = collections.defaultdict(list)  # type -> @attribute/@child lines for the element comment
        for location, docs in by_location.items():
            path, line = location.rsplit(':', 1)
            line = int(line)
            if locations[location] == 1:
                comments[path][line] += next(iter(docs.values()))
                continue
            segments = splitRegistrations(os.path.join(base_dir, path), line, locations[location])
            if segments is not None:
                splits[path][line] = [(text, docs.get((kind, None, name), []) or
                                       next((v for (k, t, n), v in docs.items() if k == kind and n == name), []))
                                      for text, kind, name in segments]
                continue
            # registrations generated by a macro: document them in the element comment
            for (kind, type_name, name), lines in docs.items():
                tag = '@attribute' if kind == 'Attribute' else '@child'
                element_lines[type_name] += [f"{tag} {name}: {lines[0]}"] + lines[1:]

        for type_name in list(self.element_docs) + [t for t in element_lines if t not in self.element_docs]:
            element = self.bindings.elements[type_name]
            doc = self.element_docs.get(type_name, {'description': None, 'examples': [], 'source': None})
            if doc.get('description_lines') is not None:
                lines = list(doc['description_lines'])  # pre-formatted (reStructuredText, indentation kept)
            else:
                lines = docLines('', doc['description']) if doc['description'] else []
            for image in doc.get('images', []):
                lines.append(f"@image {image['path']}" + (f" {image['width']}" if image.get('width') else ''))
            for example in doc['examples']:
                lines += docLines('@example ', example, keep_indent=True)
            lines += element_lines.get(type_name, [])
            if not element.defined_at:
                self.problems['cannot be placed'].append(
                    f"{doc['source'] or type_name}: {type_name} has no builder typedef/using to document")
                continue
            path, line = element.defined_at.rsplit(':', 1)
            comments[path][int(line)] += lines
        return comments, splits


def splitRegistrations(path, line, expected):
    """If the given source line contains `expected` registrations literally (not via a macro),
    returns [(text, kind, name)] with one entry per registration, else None."""
    with open(path, 'rb') as f:
        text = f.read().decode('utf-8').replace('\r\n', '\n').split('\n')[line - 1]
    matches = list(re.finditer(REGEX_PATTERN, text))
    if len(matches) != expected:
        return None
    segments = []
    for i, m in enumerate(matches):
        start = m.start()
        end = matches[i + 1].start() if i + 1 < len(matches) else len(text)
        kind = 'Attribute' if m.group(3) == 'Attribute' else 'Element'
        segments.append((text[start:end].rstrip(), kind, m.group(5)))
    return segments


def docLines(prefix, text, keep_indent=False):
    lines = text.split('\n') if keep_indent else [l.strip() for l in text.split('\n')]
    out = [prefix + lines[0]] if not keep_indent or len(lines) == 1 else ([prefix.rstrip()] + lines)
    if not keep_indent or len(lines) == 1:
        out += lines[1:]
    return [l.rstrip() for l in out]


def exampleSnippet(node, text):
    line_start = text.rfind('\n', 0, node.start) + 1
    snippet = text[line_start:node.end].expandtabs(4)
    return textwrap.dedent(snippet).strip('\n')


###############################################################################
# applying

def applyInsertions(comments, splits, base_dir, problems, changed_ranges):
    """Inserts the comments. A statement prefix like 'STATICINT sX =' in front of a documented
    registration gets its own line (the layout clang-format produces once a comment is inside the
    statement), so that each comment stays directly above its registration."""
    changed = 0
    for path in sorted(set(comments) | set(splits)):
        full = os.path.join(base_dir, path)
        with open(full, 'rb') as f:
            data = f.read().decode('utf-8')
        newline = '\r\n' if '\r\n' in data else '\n'
        lines = data.split(newline)
        targets = sorted(set(comments.get(path, {})) | set(splits.get(path, {})), reverse=True)
        for line in targets:
            target = lines[line - 1]
            if line >= 2 and lines[line - 2].strip().startswith('//!'):
                problems['already documented'].append(f"{path}:{line}: already has a '//!' comment, not changed")
                continue
            indent = target[:len(target) - len(target.lstrip())]
            comment = lambda doc, ind: [ind + ('//! ' + l if l else '//!') for l in doc]
            above = comments.get(path, {}).get(line, [])

            first = re.search(REGEX_PATTERN, target)
            prefix = target[:first.start()].rstrip() if first else ''
            statement_continues = not target.rstrip().endswith(';')
            if line in splits.get(path, {}):
                segments = splits[path][line]
            elif first and prefix.strip() and statement_continues:
                segments = [(target[first.start():].rstrip(), above)]
                above = []
            else:
                lines[line - 1:line] = comment(above, indent) + [target]
                changed += 1
                continue

            new = comment(above, indent)
            inner = indent
            if prefix.strip():
                new.append(prefix)
                inner = indent + '    '
            for text, doc in segments:
                new += comment(doc, inner) + [inner + text.strip()]
            lines[line - 1:line] = new
            changed += 1
        with open(full, 'wb') as f:
            f.write(newline.join(lines).encode('utf-8'))
        original = data.split(newline)
        matcher = difflib.SequenceMatcher(a=original, b=lines, autojunk=False)
        changed_ranges[path] = [(j1 + 1, max(j1 + 1, j2)) for op, i1, i2, j1, j2 in matcher.get_opcodes() if op != 'equal']
    return changed


def check(plan, new_json):
    """Compares the planned descriptions with a JSON extracted after applying."""
    new = bindings_model.load(new_json)
    missing = []
    for type_name, doc in plan.element_docs.items():
        element = new.elements.get(type_name)
        if doc['description'] and (element is None or not Plan.same(element.description, doc['description'])):
            missing.append(f"element {type_name} (from {doc['source']})")
    for (type_name, name), (text, source) in plan.attribute_docs.items():
        info = new.elements[type_name].attribute_info.get(name) if type_name in new.elements else None
        if info is None or not Plan.same(info.description, text):
            missing.append(f"attribute {name} of {type_name} (from {source})")
    for (type_name, tag), (text, source) in plan.child_docs.items():
        info = new.elements[type_name].child_info.get(tag) if type_name in new.elements else None
        if info is None or not Plan.same(info.description, text):
            missing.append(f"child {tag} of {type_name} (from {source})")
    return missing


###############################################################################

def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('bindings', metavar='JSON', help='bindings JSON written by extract.py (before migrating)')
    parser.add_argument('xml_files', metavar='XML', nargs='+', help='annotated XML files to migrate')
    parser.add_argument('-r', '--root', default=bindings_model.MASTER_ROOT, help='type of the document root element (default: %(default)s)')
    parser.add_argument('--transparent', default=','.join(DEFAULT_TRANSPARENT),
                        help='comma-separated preprocessor elements whose children belong to the enclosing element (default: %(default)s)')
    parser.add_argument('--ignored', default=','.join(DEFAULT_IGNORED),
                        help='comma-separated preprocessor elements to skip')
    parser.add_argument('--assign', metavar='FILE:LINE=TAG', action='append', default=[],
                        help='resolve an ambiguous comment: the comment at FILE:LINE documents the nested element TAG '
                             '(FILE may be a path suffix)')
    parser.add_argument('--base-dir', default='.', help='directory the paths in the bindings JSON are relative to')
    parser.add_argument('--apply', action='store_true', help='insert the comments into the C++ files')
    parser.add_argument('--format', metavar='CLANG_FORMAT', nargs='?', const='clang-format',
                        help='with --apply: run clang-format (using the project .clang-format) on the modified lines')
    parser.add_argument('--check', metavar='NEW_JSON', help='verify that a JSON extracted after --apply contains all descriptions')
    parser.add_argument('-v', '--verbose', action='store_true', help='also list every planned comment')
    args = parser.parse_args()

    bindings = bindings_model.load(args.bindings)
    if args.root not in bindings.elements:
        sys.exit(f"Root element {args.root} not found in {args.bindings}")
    transparent = set(filter(None, args.transparent.split(',')))
    ignored = set(filter(None, args.ignored.split(',')))
    assignments = {}
    for a in args.assign:
        location, _, tag = a.rpartition('=')
        assignments[location] = tag

    plan = Plan(bindings)
    comments, texts = [], {}
    for path in args.xml_files:
        root, texts[path] = parseXML(path)
        resolveTypes(root, bindings.elements[args.root], transparent, ignored, plan.problems, path)
        for node in descendants(root, transparent):
            previous = None
            for child in node.children:
                if child.kind == 'element' and child.name not in ignored:
                    previous = child
                elif child.kind == 'comment':
                    doc = parseDocComment(child.text)
                    if doc is None:
                        continue
                    if previous is None or previous.name in transparent:
                        plan.problems['not attached'].append(f"{path}:{child.line}: documentation comment without a preceding element")
                        continue
                    comments.append(DocComment(path, child, previous, doc))

    attachComments(comments, assignments, transparent)
    for c in comments:
        if c.target is None:
            names = ', '.join(sorted({f"<{d.name}>" for d in c.candidates})) or 'none'
            reason = c.how or ('no candidate matches the documented parameters/elements'
                               if c.doc['parameters'] or c.doc['elements'] else 'several candidates fit')
            plan.problems['not attached'].append(
                f"{c.location}: {reason} (candidates: {names}); resolve with --assign {c.location}=<tag>")
            continue
        plan.addComment(c, texts[c.path])

    comments_plan, splits = plan.insertions(args.base_dir)

    if args.check:
        missing = check(plan, args.check)
        for m in missing:
            print(f"missing after migration: {m}")
        print(f"{len(missing)} planned descriptions missing in {args.check}")
        sys.exit(1 if missing else 0)

    how = collections.Counter(c.how or 'not attached' for c in comments)
    print(f"{len(comments)} documentation comments: " + ', '.join(f"{v} {k}" for k, v in sorted(how.items())))
    print(f"planned: {len(plan.element_docs)} element descriptions, {len(plan.attribute_docs)} attribute descriptions, "
          f"{len(plan.child_docs)} child element descriptions, "
          f"{sum(len(v) for v in comments_plan.values())} comment blocks and {sum(len(v) for v in splits.values())} "
          f"split lines in {len(set(comments_plan) | set(splits))} files")
    if args.verbose:
        for c in comments:
            if c.target is not None and c.how != 'matched':
                print(f"  {c.location}: {c.how} -> <{c.target.name}> ({c.target.type.name})")
        for path in sorted(set(comments_plan) | set(splits)):
            for line in sorted(set(comments_plan.get(path, {})) | set(splits.get(path, {}))):
                print(f"\n{path}:{line}")
                for l in comments_plan.get(path, {}).get(line, []):
                    print('    //! ' + l)
                for text, doc in splits.get(path, {}).get(line, []):
                    for l in doc:
                        print('    //! ' + l)
                    print('    ' + text.strip())

    if args.apply:
        changed_ranges = {}
        changed = applyInsertions(comments_plan, splits, args.base_dir, plan.problems, changed_ranges)
        print(f"inserted {changed} comment blocks")
        if args.format:
            # only the modified lines (and the statements they belong to) are reformatted
            import subprocess
            for path, ranges in sorted(changed_ranges.items()):
                subprocess.run([args.format, '-i', '--style=file'] + [f'--lines={a}:{b}' for a, b in ranges]
                               + [os.path.join(args.base_dir, path)], check=True)
            print(f"formatted the modified lines in {len(changed_ranges)} files with {args.format}")

    for category, entries in plan.problems.items():
        print(f"\n{category} ({len(entries)}):")
        for e in entries:
            print('  ' + e)


if __name__ == '__main__':
    main()
