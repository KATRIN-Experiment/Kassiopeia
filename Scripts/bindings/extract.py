#!/usr/bin/env python3
"""
Extracts the XML bindings (elements, child elements and attributes) from the
C++ builder sources and writes them as JSON.

This is the only script that reads C++ code. All output formats (Markdown,
reStructuredText, XML example, ...) are produced by separate renderers reading
the JSON written here; see bindings_model.py for the format.
"""

import sys
import os
import re
import glob
import json
import argparse
import collections

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
        'images': [],
        'deprecated': None,
        'defined_at': None,
        'bases': [],
        'creates': None,
        'children': {},
        'attributes': {},
    }


def newRelation(target, location):
    return {
        'type': target,
        'description': None,
        'deprecated': None,
        'reference': None,   # attributes naming another object: {'type': <C++ type>} or {'geometry': 'surfaces'|...}
        'values': None,      # choice attributes: {'values': [...], 'open': bool (other values possible)}
        'registered_at': location,
    }


def parseReference(text):
    if text in ('surface', 'surfaces', 'space', 'spaces'):
        return {'geometry': text}
    return {'type': text}


###############################################################################
# references: string attributes naming another object

RE_ADD_ATTRIBUTE = re.compile(r'bool\s+(\w+?)(?:Builder|Binding)\s*::\s*AddAttribute\s*\(')
RE_ATTRIBUTE_CASE = re.compile(r'GetName\s*\(\s*\)\s*==\s*(?:std::)?(?:string\s*\()?\s*"(\w+)"')
# lookups of the attribute value (not of a fixed name like Get<KSRootMagneticField>("root_magnetic_field"))
RE_TOOLBOX_GET = re.compile(r'KToolbox::GetInstance\(\)\s*\.\s*Get\s*<\s*([\w:]+)\s*>\s*\(\s*(?!")')
RE_GEOMETRY_GET = re.compile(r'Retrieve(Surfaces|Spaces|Surface|Space)\b')


def braceBlock(text, start):
    """The statement starting at text[start]: a {...} block, or up to the next ';'."""
    brace, semicolon = text.find('{', start), text.find(';', start)
    if brace < 0 or 0 <= semicolon < brace:
        return text[start:semicolon + 1] if semicolon >= 0 else ''
    depth = 0
    for i in range(brace, len(text)):
        depth += {'{': 1, '}': -1}.get(text[i], 0)
        if depth == 0:
            return text[brace:i + 1]
    return text[brace:]


RE_MEMBER = re.compile(r'fObject\s*->\s*(f\w+)')
RE_END = re.compile(r'bool\s+(\w+?)(?:Builder|Binding)\s*::\s*End\s*\(\s*\)')
RE_SET = re.compile(r'\bSet\s*\(\s*(\w+)\s*\)')


def findReferences(buffer, namespaces):
    """Finds attributes that are resolved to other objects in a builder: {(builder name, attribute): reference}.
    An attribute is a reference if its case in AddAttribute calls KToolbox::Get<T>(...) or
    KGInterface::Retrieve*(...), or if it stores the value in a member of the builder's object that is
    later passed to KToolbox::Get<T>(fObject->fMember) (e.g. in End())."""
    found = {}
    for m in RE_ADD_ATTRIBUTE.finditer(buffer):
        body = braceBlock(buffer, m.end())
        for case in RE_ATTRIBUTE_CASE.finditer(body):
            block = braceBlock(body, case.end())
            get = RE_TOOLBOX_GET.search(block)
            if get:
                found[(m.group(1), case.group(1))] = {'type': normalizeType(get.group(1), namespaces)}
                continue
            geometry = RE_GEOMETRY_GET.search(block)
            if geometry:
                found[(m.group(1), case.group(1))] = {'geometry': geometry.group(1).lower()}
                continue
            members = set(RE_MEMBER.findall(block))
            for member in members:
                later = re.search(r'KToolbox::GetInstance\(\)\s*\.\s*Get\s*<\s*([\w:]+)\s*>\s*\(\s*fObject\s*->\s*'
                                  + member + r'\b', buffer)
                if later:
                    found[(m.group(1), case.group(1))] = {'type': normalizeType(later.group(1), namespaces)}
                    break
    return found


# a string literal the attribute value (or a variable holding it) is compared with
RE_VALUE_COMPARISON = re.compile(r'(\w[\w\s.\->()]*?)\s*==\s*(?:std::)?(?:string\s*\(\s*)?"([^"]*)"'
                                 r'|"([^"]*)"\s*==|\.compare\s*\(\s*"([^"]*)"\s*\)')
RE_VALUE_WORD = re.compile(r'^[\w.+-]+$')


def findValues(buffer):
    """Finds choice attributes: {(builder name, attribute): {'values': [...], 'open': bool}}, from string
    literals the value is compared with in the attribute's case in AddAttribute. Comparisons that are
    no complete values (e.g. a "#" prefix for colours) mark the list as open."""
    found = {}
    for m in RE_ADD_ATTRIBUTE.finditer(buffer):
        body = braceBlock(buffer, m.end())
        for case in RE_ATTRIBUTE_CASE.finditer(body):
            block = braceBlock(body, case.end())
            values, is_open = [], False
            for c in RE_VALUE_COMPARISON.finditer(block):
                if c.group(1) and 'GetName' in c.group(1):
                    continue
                value = c.group(2) if c.group(2) is not None else c.group(3) if c.group(3) is not None else c.group(4)
                if c.group(1) and 'substr' in c.group(1) or not RE_VALUE_WORD.match(value):
                    is_open = True
                elif value not in values:
                    values.append(value)
            if values:
                found[(m.group(1), case.group(1))] = {'values': values, 'open': is_open}
    return found


def findCreatedTypes(buffer, namespaces):
    """Builders whose element type is only a data wrapper create the actual object in End() and hand it
    over with Set(tObject): {builder name: C++ type of the created object}."""
    created = {}
    for m in RE_END.finditer(buffer):
        body = braceBlock(buffer, m.end())
        for variable in RE_SET.findall(body):
            if variable == 'fObject':
                continue
            decl = re.search(r'([\w:<>, ]+?)\s*\*\s*' + variable + r'\s*=\s*(?:new\s+([\w:<>, ]+?)\s*[(;{])?', body)
            if not decl:
                continue
            type_name = decl.group(2) if decl.group(1).strip() == 'auto' else decl.group(1)
            if type_name and type_name.strip() != 'auto':
                created[m.group(1)] = normalizeType(type_name.strip(), namespaces)
                break
    return created


# also template specializations: class KSComponentTemplate<XThisType, void, void, void> : virtual public KSComponent
RE_TYPE_TYPEDEF = re.compile(r'\btypedef\s+((?:typename\s+)?[\w:]+(?:\s*<[^;]*>)?)\s+(\w+)\s*;')
RE_TYPE_USING = re.compile(r'\busing\s+(\w+)\s*=\s*((?:typename\s+)?[\w:]+(?:\s*<[^;]*>)?)\s*;')
RE_CLASS = re.compile(r'\b(?:class|struct)\s+(\w+)\s*(?:<[^{;]*?>)?\s*(?:final\s*)?:(?!:)\s*([^{;]+)\{')


def splitTopLevel(text, separator=','):
    parts, depth, current = [], 0, ''
    for c in text:
        depth += {'<': 1, '>': -1}.get(c, 0)
        if c == separator and depth == 0:
            parts.append(current)
            current = ''
        else:
            current += c
    return parts + [current]


def collectClassBases(root_paths):
    """Direct base classes of all classes declared in headers below the given paths. The base is the
    last component of a qualified name (A::Visitor -> Visitor); template arguments only count for the
    CRTP pattern Template<Self, Base, ...> (e.g. KSComponentTemplate<KSGenValueSet, KSGenValue>)."""
    bases = collections.defaultdict(set)
    for path_name in root_paths:
        for file_name in glob.glob(path_name + '/**', recursive=True):
            if os.path.splitext(file_name)[1] not in ('.h', '.hh', '.hpp', '.hxx') or not os.path.isfile(file_name):
                continue
            with open(file_name, errors='replace') as f:
                text = f.read()
            for m in RE_CLASS.finditer(text):
                cls = m.group(1)
                for spec in splitTopLevel(m.group(2)):
                    spec = re.sub(r'\b(public|protected|private|virtual)\b', '', spec).strip()
                    name, _, args = spec.partition('<')
                    name = name.strip().split('::')[-1]
                    if name and name != cls:
                        bases[cls].add(name)
                    if args:
                        args = [re.sub(r'<.*', '', a).strip().split('::')[-1] for a in splitTopLevel(args.rsplit('>', 1)[0])]
                        if args and args[0] == cls:
                            bases[cls].update(a for a in args[1:] if a and a != cls)
    return bases


def ancestors(name, bases):
    seen, todo = set(), [name]
    while todo:
        for base in bases.get(todo.pop(), ()):
            if base not in seen:
                seen.add(base)
                todo.append(base)
    return seen


def className(type_name):
    """'KGWrappedSurface<KGRod>' -> 'KGWrappedSurface', 'KGPlanarPolyLoop::LineArguments' -> 'LineArguments'."""
    return re.sub(r'<.*', '', type_name).split('::')[-1]


DOC_PREFIX = '//!'
DOC_TAG = re.compile(r'@(\w+)\b\s*(.*)')


def stripComments(text):
    """Replaces the contents of // and /* */ comments by spaces (keeping newlines, so positions and line
    numbers stay the same), leaving string, character and raw string literals untouched. Used for
    matching code, so that commented-out registrations are not documented."""
    out, i, n = list(text), 0, len(text)
    while i < n:
        c = text[i]
        if text.startswith('R"', i) and (i == 0 or not (text[i - 1].isalnum() or text[i - 1] == '_')):
            open_paren = text.find('(', i + 2)
            delimiter = text[i + 2:open_paren] if open_paren >= 0 else ''
            end = text.find(')' + delimiter + '"', open_paren + 1) if open_paren >= 0 else -1
            i = end + len(delimiter) + 2 if end >= 0 else n
        elif c == '"' or c == "'":
            j = i + 1
            while j < n and text[j] != c and text[j] != '\n':
                j += 2 if text[j] == '\\' else 1
            i = j + 1
        elif text.startswith('//', i):
            j = text.find('\n', i)
            j = n if j < 0 else j
            for k in range(i, j):
                out[k] = ' '
            i = j
        elif text.startswith('/*', i):
            j = text.find('*/', i + 2)
            j = n if j < 0 else j + 2
            for k in range(i, j):
                if out[k] != '\n':
                    out[k] = ' '
            i = j
        else:
            i += 1
    return ''.join(out)


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
    result = {'description': [], 'examples': [], 'images': [], 'deprecated': None, 'attributes': {}, 'children': {},
              'reference': None, 'values': None, 'unknown': []}
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
        elif tag == 'image':
            # an image is shown where it appears in the description; the description continues after it
            path, _, width = rest.strip().partition(' ')
            result['images'].append({'path': path, 'width': width.strip() or None, 'line': len(result['description'])})
            current = result['description']
        elif tag == 'deprecated':
            current = [rest] if rest else []
            result['deprecated'] = current
        elif tag == 'values':
            # the attribute accepts these values: "@values a, b, c" (", ..." if others are possible)
            result['values'] = [v.strip() for v in rest.split(',') if v.strip()]
            current = result['description']
        elif tag == 'reference':
            # the attribute names an object of this C++ type (or "surfaces"/"spaces": a KGeoBag path)
            result['reference'] = rest.strip()
            current = result['description']
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

    # image positions refer to the lines of the final (stripped) description
    raw = result['description']
    leading = next((k for k, l in enumerate(raw) if l.strip()), len(raw))
    description = join(raw, keep_indent=True)
    count = len(description.split('\n')) if description else 0
    for image in result['images']:
        image['line'] = min(max(image['line'] - leading, 0), count)

    return {
        'description': description or None,
        'examples': [join(e, keep_indent=True) for e in result['examples'] if join(e, keep_indent=True)],
        'images': result['images'],
        'deprecated': join(result['deprecated']) if result['deprecated'] is not None else None,
        'attributes': {k: join(v) for k, v in result['attributes'].items()},
        'children': {k: join(v) for k, v in result['children'].items()},
        'reference': result['reference'],
        'values': result['values'],
        'unknown': result['unknown'],
    }


def warn(location, message):
    print(f"{location}: warning: {message}", file=sys.stderr)


errors = []


def error(location, message):
    print(f"{location}: error: {message}", file=sys.stderr)
    errors.append(message)


# a file defines bindings if it contains a builder typedef/using or a registration call
RE_HAS_BINDINGS = re.compile(r'\bK(?:Simple|Complex)Element\s*<|(?:Builder|Binding|ElementProcessor)\s*::\s*(?:template\s+)?'
                             r'(?:Attribute|SimpleElement|ComplexElement)\s*<')
RE_CMAKE_FILE = re.compile(r'[\w.+-]+\.(?:cc|cxx|cpp|h|hh|hpp|hxx)\b')


def getBindingsFiles(root_paths, compiled_only=True):
    """Returns the C++ files below the given paths that define bindings. With compiled_only, files
    that are not referenced in any CMakeLists.txt below the paths (i.e. not part of the build) are
    skipped with a warning, so that bindings that do not exist at runtime are not documented."""
    candidates, cmake_names = [], set()
    for path_name in root_paths:
        for file_name in glob.glob(path_name + '/**', recursive=True):
            if os.path.basename(file_name) == 'CMakeLists.txt':
                with open(file_name, errors='replace') as f:
                    cmake_names.update(RE_CMAKE_FILE.findall(f.read()))
                continue
            if os.path.splitext(file_name)[1] not in CXX_EXTENSIONS or not os.path.isfile(file_name):
                continue
            with open(file_name, errors='replace') as f:
                if RE_HAS_BINDINGS.search(f.read()):
                    candidates.append(file_name)

    file_list = []
    for file_name in sorted(candidates):
        if compiled_only and os.path.basename(file_name) not in cmake_names:
            warn(file_name, "defines bindings but is not part of the build (not in any CMakeLists.txt), skipped")
            continue
        file_list.append(file_name)
    return file_list


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
    references = {}
    created = {}
    choices = {}

    buffers = {}
    original_lines = {}
    for file_name in file_list:
        with open(file_name) as f:
            buffers[file_name] = ' '.join(f.readlines())
        original_lines[file_name] = buffers[file_name].split('\n')

    macros = {}
    for buffer in buffers.values():
        macros.update(collectMacros(buffer))
    for file_name in file_list:
        buffers[file_name] = expandTemplateFunctions(expandMacros(buffers[file_name], macros))

    lines = {file_name: buffer.split('\n') for file_name, buffer in buffers.items()}
    lineOf = lambda buffer, pos: buffer.count('\n', 0, pos) + 1
    # code without comments (same positions), for matching; documentation comments are read from `lines`
    code = {file_name: stripComments(buffer) for file_name, buffer in buffers.items()}

    # C++ type aliases of element types, e.g.
    #   typedef KGBEMAttributor<KElectrostaticBasis, KDirichletBoundary> KGElectrostaticDirichletAttributor;
    # so that registrations using either name refer to the same element
    type_aliases = {}
    for buffer in code.values():
        found = [(m.group(2), m.group(1)) for m in RE_TYPE_TYPEDEF.finditer(buffer)]
        found += [(m.group(1), m.group(2)) for m in RE_TYPE_USING.finditer(buffer)]
        for alias, target in found:
            if not re.match(r'K(Simple|Complex)Element\b', target.strip()):
                type_aliases[alias] = normalizeType(target, namespaces)

    def canonical(name):
        seen = set()
        while name in type_aliases and name not in seen:
            seen.add(name)
            name = type_aliases[name]
        return name

    # first collect builder aliases (typedef/using) and element documentation
    element_docs = []  # (target, location, parsed doc)
    for file_name in file_list:
        buffer = code[file_name]
        found = [(m.start(), m.group(1), m.group(2), m.group(3)) for m in re.finditer(REGEX_PATTERN_TYPEDEF, buffer)]
        found += [(m.start(), m.group(3), m.group(4), m.group(1)) for m in re.finditer(REGEX_PATTERN_USING, buffer)]
        for pos, kind, target, alias in sorted(found):
            target = canonical(normalizeType(target, namespaces))
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
        buffer = code[file_name]
        warnUnmatched(file_name, buffer)
        references.update(findReferences(buffer, namespaces))
        created.update(findCreatedTypes(buffer, namespaces))
        choices.update(findValues(buffer))
        matches = list(re.finditer(REGEX_PATTERN, buffer))
        registrations_per_line = {}
        for m in matches:
            registrations_per_line.setdefault(lineOf(buffer, m.start()), []).append(m)

        for line, line_matches in registrations_per_line.items():
            location = f"{file_name}:{line}"
            doc = docCommentAbove(lines[file_name], line)
            parsed = parseDocComment(doc) if doc else None
            if parsed and len(line_matches) > 1 and not re.match(r'\s*(?:\w+\s*=\s*)?' + REGEX_PATTERN,
                                                                  re.sub(r'^\s*STATICINT\s+', '', original_lines[file_name][line - 1])):
                # the line does not start with a registration in the source (e.g. a macro generates several)
                warn(location, "documentation comment above a line with several registrations that are not written out "
                               "in the source is ignored; use @attribute/@child in the element documentation instead")
                parsed = None
            if parsed and (parsed['examples'] or parsed['images'] or parsed['attributes'] or parsed['children'] or parsed['unknown']):
                warn(location, "only a description, @deprecated and @reference are supported above a registration")

            for m in line_matches:
                source, suffix, kind, target, name = m.groups()
                target = canonical(normalizeType(target, namespaces))

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
                if parsed and m is line_matches[0]:
                    # a comment documents the registration that starts directly below it
                    applyRelationDoc(relations[name], parsed['description'], parsed['deprecated'], location)
                    if parsed['reference']:
                        relations[name]['reference'] = parseReference(parsed['reference'])
                        relations[name]['reference']['annotated'] = True
                    if parsed['values']:
                        values = [v for v in parsed['values'] if v != '...']
                        relations[name]['values'] = {'values': values, 'open': '...' in parsed['values'],
                                                     'annotated': True}

    # element documentation (after all registrations, so that @attribute/@child can refer to them)
    for target, location, parsed in element_docs:
        element = elements[target]
        for tag in parsed['unknown']:
            warn(location, f"unknown documentation tag @{tag}")
        for image in parsed['images']:
            if not os.path.isfile(image['path']):
                error(location, f"image '{image['path']}' does not exist (paths are relative to the repository root)")
        if parsed['description'] or parsed['examples'] or parsed['images']:
            if element['description'] or element['examples'] or element['images']:
                warn(location, f"{target} is already documented, ignoring this comment")
            else:
                element['description'] = parsed['description']
                element['examples'] = parsed['examples']
                element['images'] = parsed['images']
        if parsed['deprecated'] is not None:
            element['deprecated'] = parsed['deprecated']
        for key, relations in (('attributes', element['attributes']), ('children', element['children'])):
            for name, text in parsed[key].items():
                if name not in relations:
                    warn(location, f"documented {key[:-1] if key == 'children' else 'attribute'} '{name}' is not registered for {target}")
                    continue
                applyRelationDoc(relations[name], text, None, location)

    # references detected in the builders (an explicit @reference takes precedence)
    builders = {target: alias for alias, target in aliases.items()}
    for name, element in elements.items():
        builder = builders.get(name, name)
        for attribute, relation in element['attributes'].items():
            found = references.get((builder, attribute))
            if found and not relation['reference']:
                relation['reference'] = found
            values = choices.get((builder, attribute))
            if values and not relation['values']:
                relation['values'] = values

    for name, element in elements.items():
        builder = builders.get(name, name)
        element['creates'] = created.get(builder) if created.get(builder) not in (None, name) else None

    for element in elements.values():
        element['tags'] = sorted(element['tags'])
        element['source_files'] = sorted(element['source_files'])

    return (elements, aliases)


def addBases(elements, root_paths):
    """Stores the (transitive) base classes of every element type, so that renderers can list the
    elements an attribute with a reference can refer to."""
    bases = collectClassBases(root_paths)
    for name, element in elements.items():
        found = ancestors(className(name), bases)
        if element.get('creates'):  # the object put into the toolbox is of another type
            found |= {className(element['creates'])} | ancestors(className(element['creates']), bases)
        element['bases'] = sorted(found)
    for name, element in elements.items():
        for attribute, relation in element['attributes'].items():
            ref = relation['reference']
            if ref and 'type' in ref and not any(className(ref['type']) == className(n) or className(ref['type']) in e['bases']
                                                 for n, e in elements.items() if e['tags']):
                warn(relation['registered_at'], f"attribute '{attribute}' of {name} refers to {ref['type']}, "
                                                f"but no XML element creates such objects")


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
    parser.add_argument('--all-files', action='store_true',
                        help='also read files that are not referenced in any CMakeLists.txt')
    parser.add_argument('-o', '--output', metavar='FILE', type=str, default='-',
                        help='output file (default: standard output)')
    parser.epilog = 'Run from the repository root: @image paths in documentation comments are relative to it.'
    return parser.parse_args()


if __name__ == "__main__":
    args = parseArguments()

    file_list = getBindingsFiles(args.paths, not args.all_files)
    if not file_list:
        print("The provided path doesn't contain any bindings files.", file=sys.stderr)
        sys.exit(1)

    elements, aliases = processFiles(file_list, collectNamespaces(args.paths))
    if not elements:
        print("No valid bindings were found in the input files.", file=sys.stderr)
        sys.exit(1)
    addBases(elements, args.paths)

    data = {
        'format_version': FORMAT_VERSION,
        'elements': elements,
        'aliases': aliases,
    }

    if errors:
        print(f"{len(errors)} error(s) in the binding documentation", file=sys.stderr)
        sys.exit(1)

    out = sys.stdout if args.output == '-' else open(args.output, 'w')
    json.dump(data, out, indent=1)
    out.write('\n')
