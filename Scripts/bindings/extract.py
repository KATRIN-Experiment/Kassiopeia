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

from bindings_model import FORMAT_VERSION

STRIP_NAMESPACES = ['std', 'detail', 'katrin', 'KGeoBag', 'KEMField']

#### https://regex101.com/r/ieEZ8y/1
REGEX_PATTERN = r'(\s*(\w+)(Builder|Binding)::(Attribute|SimpleElement|ComplexElement)<\s*([\w]+::)?([\w<> ]+)\s*>\s*\(\s*"(\w+)"\s*\)\s*[+;])'

REGEX_PATTERN_TYPEDEF = r'(typedef\s+K(SimpleElement|ComplexElement)\s*<\s*([\w]+::)?([\w<> ]+)\s*>\s*(\w+)(Builder|Binding)\s*;)'
REGEX_PATTERN_USING = r'(using\s+(\w+)(Builder|Binding)\s*=\s*K(SimpleElement|ComplexElement)\s*<\s*([\w]+::)?([\w<> ]+)\s*>\s*;)'


def newElement(kind):
    return {
        'kind': kind,
        'tags': set(),
        'source_files': set(),
        'children': {},
        'attributes': {},
    }


def getBindingsFiles(root_paths):
    file_list = []
    for path_name in root_paths:
        for file_name in glob.glob(path_name + '/**', recursive=True):
            root, ext = os.path.splitext(file_name)
            if ext not in ['.cc', '.cxx', '.cpp', '.h', '.hh', '.hxx', '.hpp']:
                continue
            if not root.endswith("Builder"):
                continue

            file_list.append(file_name)

    return sorted(file_list)


def processFiles(file_list):
    aliases = {}
    elements = {}

    # first process header files
    for file_name in file_list:
        with open(file_name) as f:
            buffer = ' '.join(f.readlines())

            match = re.findall(REGEX_PATTERN_TYPEDEF, buffer)
            for m in match:
                kind, prefix, target, alias, builder = m[1:]
                target = target.strip()

                if prefix and prefix[:-2] not in STRIP_NAMESPACES:
                    target = prefix[:-2] + target

                if not target in elements:
                    elements[target] = newElement(kind)

                if alias != target:
                    aliases[alias] = target

            match = re.findall(REGEX_PATTERN_USING, buffer)
            for m in match:
                alias, builder, kind, prefix, target = m[1:]
                target = target.strip()

                if prefix and prefix[:-2] not in STRIP_NAMESPACES:
                    target = prefix[:-2] + target

                if not target in elements:
                    elements[target] = newElement(kind)

                if alias != target:
                    aliases[alias] = target

    # now process source files
    for file_name in file_list:
        with open(file_name) as f:
            buffer = ' '.join(f.readlines())

            match = re.findall(REGEX_PATTERN, buffer)
            for m in match:
                source, builder, kind, prefix, target, name = m[1:]
                target = target.strip()

                if source in aliases:
                    source = aliases[source]

                if prefix and prefix[:-2] not in STRIP_NAMESPACES:
                    target = prefix[:-2] + target

                if not source in elements:
                    elements[source] = newElement('ComplexElement')

                if kind == 'Attribute':
                    elements[source]['attributes'][name] = target

                elif kind == 'ComplexElement' or kind == 'SimpleElement':
                    if not target in elements:
                        elements[target] = newElement(kind)
                    elements[target]['tags'].add(name)
                    elements[target]['source_files'].add(file_name)
                    elements[source]['children'][name] = target

                else:
                    raise RuntimeError("Unrecognized element type: %s" % kind)

    for element in elements.values():
        element['tags'] = sorted(element['tags'])
        element['source_files'] = sorted(element['source_files'])

    return (elements, aliases)


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

    elements, aliases = processFiles(file_list)
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
