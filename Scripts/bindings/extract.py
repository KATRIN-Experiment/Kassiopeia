#!/usr/bin/env python3
"""
Extracts the XML bindings (elements, child elements and attributes) from the
C++ builder sources and writes them as JSON.

This is the only script that reads C++ code. All output formats (Markdown,
reStructuredText, GraphViz, ...) are produced by separate renderers reading
the JSON written here; see bindings_model.py for the format.
"""

import sys
import os
import re
import glob
import json
import argparse

from bindings_model import FORMAT_VERSION, MASTER_ROOT as ROOT_ELEMENT

CXX_EXTENSIONS = ['.cc', '.cxx', '.cpp', '.h', '.hh', '.hxx', '.hpp']

# A C++ type argument: anything up to the closing '>' that cannot leave the
# statement (may contain namespaces, nested templates and commas).
TYPE = r'([^;(){}"]+?)'

# <Name>Builder::Attribute<Type>("xml_name") +   (also SimpleElement/ComplexElement,
# and "Builder::template ComplexElement<...>" as used in dependent contexts)
REGEX_PATTERN = (r'(\w+)(Builder|Binding|ElementProcessor)\s*::\s*(?:template\s+)?(Attribute|SimpleElement|ComplexElement)\s*<\s*'
                 + TYPE + r'\s*>\s*\(\s*"(\w+)"\s*\)\s*[+;]')

# typedef KComplexElement<Type> <Name>Builder;
REGEX_PATTERN_TYPEDEF = r'typedef\s+K(SimpleElement|ComplexElement)\s*<\s*' + TYPE + r'\s*>\s*(\w+)(Builder|Binding)\s*;'
# using <Name>Builder = KComplexElement<Type>;
REGEX_PATTERN_USING = r'using\s+(\w+)(Builder|Binding)\s*=\s*K(SimpleElement|ComplexElement)\s*<\s*' + TYPE + r'\s*>\s*;'

# function-like macro definitions, including continuation lines
REGEX_DEFINE = re.compile(r'^[ \t]*#[ \t]*define[ \t]+(\w+)\(([^)]*)\)((?:[^\n]*\\\n)*[^\n]*)', re.M)


def collectNamespaces(root_paths):
    """Returns the names of all namespaces declared in C++ files below the given paths."""
    namespaces = {'std'}
    for path_name in root_paths:
        for file_name in glob.glob(path_name + '/**', recursive=True):
            if os.path.splitext(file_name)[1] not in CXX_EXTENSIONS or not os.path.isfile(file_name):
                continue
            with open(file_name, errors='replace') as f:
                namespaces.update(re.findall(r'^\s*namespace\s+(\w+)', f.read(), re.M))
    return namespaces


def normalizeType(name, namespaces):
    """Strips namespace qualifiers (but keeps class qualifiers) and normalizes whitespace, e.g.
    'KGBEMAttributor< KEMField::KMagnetostaticBasis, KEMField::KNeumannBoundary >'
    -> 'KGBEMAttributor<KMagnetostaticBasis, KNeumannBoundary>',
    'KGeoBag::KGPlanarPolyLine::LineArguments' -> 'KGPlanarPolyLine::LineArguments'."""
    name = re.sub(r'\b(\w+)\s*::\s*', lambda m: '' if m.group(1) in namespaces else m.group(1) + '::', name)
    name = re.sub(r'\s+', ' ', name).strip()
    name = re.sub(r'\s*([<>])\s*', r'\1', name)
    name = re.sub(r'\s*,\s*', ', ', name)
    return name


def collectMacros(text):
    macros = {}
    for m in REGEX_DEFINE.finditer(text):
        params = [p.strip() for p in m.group(2).split(',') if p.strip()]
        body = m.group(3).replace('\\\n', ' ')
        macros[m.group(1)] = (params, body)
    return macros


def splitArguments(text, start):
    """Parses a parenthesized, comma-separated argument list starting at text[start] == '('.
    Returns (arguments, end index after ')'), or None if unbalanced."""
    depth, args, current, i, quote = 0, [], '', start, None
    while i < len(text):
        c = text[i]
        if quote:
            current += c
            if c == '\\':
                current += text[i+1:i+2]
                i += 1
            elif c == quote:
                quote = None
        elif c in '"\'':
            quote = c
            current += c
        elif c == '(':
            depth += 1
            if depth > 1:
                current += c
        elif c == ')':
            depth -= 1
            if depth == 0:
                args.append(current.strip())
                return args, i + 1
            current += c
        elif c == ',' and depth == 1:
            args.append(current.strip())
            current = ''
        else:
            current += c
        i += 1
    return None


REGEX_TEMPLATE_FUNCTION = re.compile(r'template\s*<\s*(?:class|typename)\s+(\w+)\s*>\s*\w+\s+(\w+)\s*\(\s*\)\s*\{')


def expandTemplateFunctions(text):
    """Handles helper function templates that register elements for a builder
    given as template parameter, e.g.

        template<class Builder> int AddMatrixAsElement() { return Builder::template ComplexElement<X>("x"); }
        STATICINT s = AddMatrixAsElement<KKrylovChargeDensitySolverBuilder>();

    Each call is replaced by the function body with the template argument
    substituted; the definition itself is blanked out (keeping line numbers)."""
    functions = {}
    for m in list(REGEX_TEMPLATE_FUNCTION.finditer(text)):
        depth, i = 1, m.end()
        while i < len(text) and depth:
            depth += {'{': 1, '}': -1}.get(text[i], 0)
            i += 1
        body = text[m.end():i-1]
        if 'Element' not in body and 'Attribute' not in body:
            continue
        functions[m.group(2)] = (m.group(1), body, m.start(), i)

    for name, (param, body, start, end) in sorted(functions.items(), key=lambda f: -f[1][2]):
        text = text[:start] + '\n' * text.count('\n', start, end) + text[end:]
    for name, (param, body, start, end) in functions.items():
        body = re.sub(r'\breturn\b', '', body).replace('\n', ' ')
        text = re.sub(r'\b' + name + r'\s*<\s*(\w+)\s*>\s*\(\s*\)',
                      lambda m: re.sub(r'\b' + param + r'\b', m.group(1), body).strip().rstrip(';'), text)
    return text


def expandMacros(text, macros, passes=3):
    """Expands invocations of the given function-like macros (simple textual
    substitution with # and ## support). Macro definitions are blanked out first,
    keeping line numbers intact."""
    text = REGEX_DEFINE.sub(lambda m: '\n' * m.group(0).count('\n'), text)
    for _ in range(passes):
        changed = False
        for name, (params, body) in macros.items():
            pos = 0
            pattern = re.compile(r'\b' + name + r'\s*\(')
            while True:
                m = pattern.search(text, pos)
                if not m:
                    break
                parsed = splitArguments(text, m.end() - 1)
                if parsed is None or len(parsed[0]) != len(params):
                    pos = m.end()
                    continue
                args, end = parsed
                expansion = body
                for param, arg in zip(params, args):
                    expansion = re.sub(r'(?<![#\w])#\s*' + param + r'\b', lambda _: '"' + arg + '"', expansion)
                    expansion = re.sub(r'\b' + param + r'\b', lambda _: arg, expansion)
                expansion = re.sub(r'\s*##\s*', '', expansion)
                text = text[:m.start()] + expansion + text[end:]
                pos = m.start() + len(expansion)
                changed = True
        if not changed:
            break
    return text


def newElement(kind):
    return {
        'kind': kind,
        'tags': set(),
        'source_files': set(),
        'description': None,
        'examples': [],
        'deprecated': None,
        'defined_at': None,
        'children': {},
        'attributes': {},
    }


def newRelation(target, location):
    return {
        'type': target,
        'description': None,
        'deprecated': None,
        'registered_at': location,
    }


DOC_PREFIX = '//!'
DOC_TAG = re.compile(r'@(\w+)\b\s*(.*)')


def docCommentAbove(lines, line_number):
    """Returns the '//!' comment lines directly above the given (1-based) line, without the prefix."""
    doc = []
    i = line_number - 2
    while i >= 0 and lines[i].strip().startswith(DOC_PREFIX):
        text = lines[i].strip()[len(DOC_PREFIX):]
        doc.insert(0, text[1:] if text.startswith(' ') else text)
        i -= 1
    return doc


def parseDocComment(doc):
    """Parses '//!' lines into description, examples, deprecation note and
    @attribute/@child descriptions. A tag line continues on the following
    lines until the next tag."""
    result = {'description': [], 'examples': [], 'deprecated': None, 'attributes': {}, 'children': {}, 'unknown': []}
    current = result['description']
    for line in doc:
        m = DOC_TAG.match(line.strip())
        if not m:
            current.append(line)
            continue
        tag, rest = m.groups()
        if tag == 'example':
            current = [rest] if rest else []
            result['examples'].append(current)
        elif tag == 'deprecated':
            current = [rest] if rest else []
            result['deprecated'] = current
        elif tag in ('attribute', 'child'):
            name, _, text = rest.partition(':')
            current = [text.strip()] if text.strip() else []
            result['attributes' if tag == 'attribute' else 'children'][name.strip()] = current
        else:
            result['unknown'].append(tag)
            current = []

    def join(lines, keep_indent=False):
        if lines is None:
            return None
        if keep_indent:
            text = '\n'.join(lines)
            # remove common indentation, keep relative indentation of examples
            indents = [len(l) - len(l.lstrip()) for l in lines if l.strip()]
            cut = min(indents) if indents else 0
            text = '\n'.join(l[cut:] for l in lines)
        else:
            text = '\n'.join(l.strip() for l in lines)
        return text.strip('\n').rstrip()

    return {
        'description': join(result['description']) or None,
        'examples': [join(e, keep_indent=True) for e in result['examples'] if join(e, keep_indent=True)],
        'deprecated': join(result['deprecated']) if result['deprecated'] is not None else None,
        'attributes': {k: join(v) for k, v in result['attributes'].items()},
        'children': {k: join(v) for k, v in result['children'].items()},
        'unknown': result['unknown'],
    }


def warn(location, message):
    print(f"{location}: warning: {message}", file=sys.stderr)


def getBindingsFiles(root_paths):
    file_list = []
    for path_name in root_paths:
        for file_name in glob.glob(path_name + '/**', recursive=True):
            root, ext = os.path.splitext(file_name)
            if ext not in CXX_EXTENSIONS:
                continue
            if not root.endswith("Builder"):
                continue

            file_list.append(file_name)

    return sorted(file_list)


# a registration call: "...::Attribute<" etc. (as opposed to typedefs or template definitions)
REGEX_REGISTRATION_CALL = re.compile(r'::\s*(?:template\s+)?(Attribute|SimpleElement|ComplexElement)\s*<')


def warnUnmatched(file_name, buffer):
    """Warns about registration calls that REGEX_PATTERN does not understand,
    so that bindings missing from the output do not go unnoticed."""
    matched = {m.start(3) for m in re.finditer(REGEX_PATTERN, buffer)}
    for m in REGEX_REGISTRATION_CALL.finditer(buffer):
        if m.start(1) in matched:
            continue
        line = buffer.count('\n', 0, m.start()) + 1
        context = re.sub(r'\s+', ' ', buffer[max(0, m.start() - 40):m.end() + 60]).strip()
        print(f"{file_name}:{line}: warning: unrecognized binding registration: ...{context}...", file=sys.stderr)


def processFiles(file_list, namespaces):
    aliases = {}
    elements = {}

    buffers = {}
    for file_name in file_list:
        with open(file_name) as f:
            buffers[file_name] = ' '.join(f.readlines())

    macros = {}
    for buffer in buffers.values():
        macros.update(collectMacros(buffer))
    for file_name in file_list:
        buffers[file_name] = expandTemplateFunctions(expandMacros(buffers[file_name], macros))

    lines = {file_name: buffer.split('\n') for file_name, buffer in buffers.items()}
    lineOf = lambda buffer, pos: buffer.count('\n', 0, pos) + 1

    # first collect builder aliases (typedef/using) and element documentation
    element_docs = []  # (target, location, parsed doc)
    for file_name in file_list:
        buffer = buffers[file_name]
        found = [(m.start(), m.group(1), m.group(2), m.group(3)) for m in re.finditer(REGEX_PATTERN_TYPEDEF, buffer)]
        found += [(m.start(), m.group(3), m.group(4), m.group(1)) for m in re.finditer(REGEX_PATTERN_USING, buffer)]
        for pos, kind, target, alias in sorted(found):
            target = normalizeType(target, namespaces)
            line = lineOf(buffer, pos)
            location = f"{file_name}:{line}"

            if not target in elements:
                elements[target] = newElement(kind)
            if elements[target]['defined_at'] is None:
                elements[target]['defined_at'] = location

            if alias != target:
                aliases[alias] = target

            doc = docCommentAbove(lines[file_name], line)
            if doc:
                element_docs.append((target, location, parseDocComment(doc)))

    # now collect registrations of attributes and child elements
    for file_name in file_list:
        buffer = buffers[file_name]
        warnUnmatched(file_name, buffer)
        matches = list(re.finditer(REGEX_PATTERN, buffer))
        registrations_per_line = {}
        for m in matches:
            registrations_per_line.setdefault(lineOf(buffer, m.start()), []).append(m)

        for line, line_matches in registrations_per_line.items():
            location = f"{file_name}:{line}"
            doc = docCommentAbove(lines[file_name], line)
            parsed = parseDocComment(doc) if doc else None
            if parsed and len(line_matches) > 1:
                warn(location, "documentation comment above a line with several registrations is ignored; "
                               "use @attribute/@child in the element documentation instead")
                parsed = None
            if parsed and (parsed['examples'] or parsed['attributes'] or parsed['children'] or parsed['unknown']):
                warn(location, "only a description and @deprecated are supported above a registration")

            for m in line_matches:
                source, suffix, kind, target, name = m.groups()
                target = normalizeType(target, namespaces)

                if source == 'K' and suffix == 'ElementProcessor':
                    source = ROOT_ELEMENT  # KElementProcessor registers top-level elements
                elif source in aliases:
                    source = aliases[source]

                if not source in elements:
                    elements[source] = newElement('ComplexElement')

                if kind == 'Attribute':
                    relations = elements[source]['attributes']
                elif kind == 'ComplexElement' or kind == 'SimpleElement':
                    if not target in elements:
                        elements[target] = newElement(kind)
                    elements[target]['tags'].add(name)
                    elements[target]['source_files'].add(file_name)
                    relations = elements[source]['children']
                else:
                    raise RuntimeError("Unrecognized element type: %s" % kind)

                if name not in relations:
                    relations[name] = newRelation(target, location)
                elif relations[name]['type'] != target:
                    # KComplexElement keeps the first registration (std::map::insert)
                    warn(location, f"'{name}' is already registered for {source} as {relations[name]['type']} "
                                   f"at {relations[name]['registered_at']}; this registration ({target}) has no effect")
                if parsed:
                    applyRelationDoc(relations[name], parsed['description'], parsed['deprecated'], location)

    # element documentation (after all registrations, so that @attribute/@child can refer to them)
    for target, location, parsed in element_docs:
        element = elements[target]
        for tag in parsed['unknown']:
            warn(location, f"unknown documentation tag @{tag}")
        if parsed['description'] or parsed['examples']:
            if element['description'] or element['examples']:
                warn(location, f"{target} is already documented, ignoring this comment")
            else:
                element['description'] = parsed['description']
                element['examples'] = parsed['examples']
        if parsed['deprecated'] is not None:
            element['deprecated'] = parsed['deprecated']
        for key, relations in (('attributes', element['attributes']), ('children', element['children'])):
            for name, text in parsed[key].items():
                if name not in relations:
                    warn(location, f"documented {key[:-1] if key == 'children' else 'attribute'} '{name}' is not registered for {target}")
                    continue
                applyRelationDoc(relations[name], text, None, location)

    for element in elements.values():
        element['tags'] = sorted(element['tags'])
        element['source_files'] = sorted(element['source_files'])

    return (elements, aliases)


def applyRelationDoc(relation, description, deprecated, location):
    if description:
        if relation['description'] and relation['description'] != description:
            warn(location, f"'{relation['type']}' registration is already documented, ignoring this comment")
        else:
            relation['description'] = description
    if deprecated is not None:
        relation['deprecated'] = deprecated


def parseArguments():
    parser = argparse.ArgumentParser(description='Extract XML bindings from C++ builder sources and write them as JSON.')
    parser.add_argument('paths', metavar='PATH', type=str, nargs='+',
                        help='base paths to search for bindings files')
    parser.add_argument('-o', '--output', metavar='FILE', type=str, default='-',
                        help='output file (default: standard output)')
    return parser.parse_args()


if __name__ == "__main__":
    args = parseArguments()

    file_list = getBindingsFiles(args.paths)
    if not file_list:
        print("The provided path doesn't contain any bindings files.", file=sys.stderr)
        sys.exit(1)

    elements, aliases = processFiles(file_list, collectNamespaces(args.paths))
    if not elements:
        print("No valid bindings were found in the input files.", file=sys.stderr)
        sys.exit(1)

    data = {
        'format_version': FORMAT_VERSION,
        'elements': elements,
        'aliases': aliases,
    }

    out = sys.stdout if args.output == '-' else open(args.output, 'w')
    json.dump(data, out, indent=1)
    out.write('\n')
