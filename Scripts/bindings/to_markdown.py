#!/usr/bin/env python3
"""
Renders the bindings JSON as Markdown tables. Section breaks and XML examples are optional.
"""

import os
import argparse

import bindings_model
from to_xml import makeExampleXML

NUM_COLS = 6
COL_WIDTH = 5
COL_HEADERS = ('element name', 'source files', 'child elements', 'child types', 'attributes', 'attribute types')


def escapeStr(txt):
    return txt.translate(dict([(ord(c), '\\' + c) for c in "<>"]))  # escape invalid chars


def cleanStr(txt):
    return txt.translate(dict([(ord(c), '_') for c in "<>:, "]))  # strip invalid chars


def printRow(fields):
    print((('|%%-%ds' % COL_WIDTH)*NUM_COLS) % tuple(fields) + '|')


def printHeader():
    print()
    printRow(COL_HEADERS)
    print(('|%s' % (COL_WIDTH*'-'))*NUM_COLS + '|')


def makeTableMD(element, stack, nodes, example_stack, base_url='', level=0, key='', with_sections=False, with_examples=False):
    if not key and element.tags:
        key = element.tags[0]

    full_name = cleanStr(f'{element.name}')
    if full_name in nodes:
        return
    nodes.add(full_name)

    if with_sections:
        if element.children or element.attributes or element.tags:
            heading = (level+1)*'#'
            print()
            print(f'{heading} {escapeStr(element.name)}')

            if with_examples and element.tags:
                print("Example:")
                print("```")
                makeExampleXML(element, example_stack, max_children=1, max_attributes=3, include_root=False)
                print("```")

            printHeader()
    elif level == 0:
        printHeader()

    self_node = f'<a name="{escapeStr(element.name.lower())}">`{key}`</a>' if key else "—"

    if element.source_files:
        source_files = '<br>'.join([f'[*{os.path.basename(file)}*]({base_url}{file})' for file in element.source_files])
    else:
        source_files = "—"

    sorted_children = sorted(element.children.keys())
    if sorted_children:
        child_nodes = '<br>'.join([f'[`{k}`](#{escapeStr(element.children[k].name.lower())})' for k in sorted_children])
        child_types = '<br>'.join([f'*`{element.children[k].name}`*' for k in sorted_children])
    else:
        child_nodes = child_types = "—"

    sorted_attributes = sorted(element.attributes.keys())
    if sorted_attributes:
        attr_nodes = '<br>'.join([f'`{k}`' for k in sorted_attributes])
        attr_types = '<br>'.join([f'*`{element.attributes[k]}`*' for k in sorted_attributes])
    else:
        attr_nodes = attr_types = "—"

    printRow((self_node, source_files, child_nodes, child_types, attr_nodes, attr_types))

    for node_key in sorted_children:
        child = element.children[node_key]
        full_name = f"{element.name}__{node_key}"
        if not full_name in stack:
            stack.append(full_name)
            makeTableMD(child, stack, nodes, example_stack, base_url, level+1, key=node_key,
                        with_sections=with_sections, with_examples=with_examples)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    bindings_model.add_common_arguments(parser)
    parser.add_argument('-b', '--base-url', metavar='URL', type=str, default="",
                        help='base url for links to files (e.g. GitHub tree)')
    parser.add_argument('--with-sections', action='store_true',
                        help='include section dividers')
    parser.add_argument('--with-examples', action='store_true',
                        help='include syntax examples')
    args = parser.parse_args()

    stack = []
    nodes = set()
    example_stack = []
    for root in bindings_model.load_roots(args):
        makeTableMD(root, stack, nodes, example_stack, args.base_url,
                    with_sections=args.with_sections, with_examples=args.with_examples)
