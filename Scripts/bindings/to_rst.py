#!/usr/bin/env python3
"""
Renders the bindings JSON as reStructuredText (RST) tables. Section breaks are optional.
"""

import os
import argparse

import bindings_model

NUM_COLS = 6
COL_WIDTHS = (80, 240, 80, 160, 80, 40)  # in RST the columns all must have the same width
COL_HEADERS = ('element name', 'source files', 'child elements', 'child types', 'attributes', 'attribute types')
HEAD_SEP = ["-", "~", "^", "'"]  # starts at level 3


def escapeStr(txt):
    return txt.translate(dict([(ord(c), '\\' + c) for c in "<>"]))  # escape invalid chars


def cleanStr(txt):
    return txt.translate(dict([(ord(c), '_') for c in "<>:, "]))  # strip invalid chars


def printLine(separators):
    assert(len(separators) == NUM_COLS)
    print(''.join([('+%s' % (COL_WIDTHS[i]*separators[i])) for i in range(NUM_COLS)]) + '+')


def printFields(fields):
    assert(len(fields) == NUM_COLS)
    print(''.join([(('|%%-%ds' % COL_WIDTHS[i]) % fields[i]) for i in range(NUM_COLS)]) + '|')


def printHeader():
    print()
    printLine('-'*NUM_COLS)
    printFields(COL_HEADERS)
    printLine('='*NUM_COLS)  # header separator


def makeTableRST(element, stack, nodes, base_url='', level=0, key='', with_sections=False):
    if not key and element.tags:
        key = element.tags[0]

    full_name = cleanStr(f'{element.name}')
    if full_name in nodes:
        return
    nodes.add(full_name)

    if with_sections:
        if element.children or element.attributes:
            if level < len(HEAD_SEP):
                print()
                print(f'.. _{cleanStr(element.name)}:')
                print()
                print(f'{element.name}')
                print(100*HEAD_SEP[level])

            printHeader()
    elif level == 0:
        printHeader()

    self_node = f'``{key}``' if key else "—"
    sorted_files = element.source_files
    sorted_children = sorted(element.children.keys())
    sorted_attributes = sorted(element.attributes.keys())

    numLines = max(len(sorted_files), len(sorted_children), len(sorted_attributes))
    for i in range(numLines):
        if not sorted_files and i == 0:
            source_files = "—"
        elif i < len(sorted_files):
            file = sorted_files[i]
            source_files = f'`{os.path.basename(file)} <{base_url}{file}>`_'
        else:
            source_files = ""

        if not sorted_children and i == 0:
            child_nodes = child_types = "—"
        elif i < len(sorted_children):
            node_key = sorted_children[i]
            child = element.children[node_key]
            child_nodes = f'``{node_key}``'
            child_types = f':ref:`{escapeStr(child.name)} <{cleanStr(child.name)}>`'
        else:
            child_nodes = child_types = ""

        if not sorted_attributes and i == 0:
            attr_nodes = attr_types = "—"
        elif i < len(sorted_attributes):
            attr_key = sorted_attributes[i]
            attr_nodes = f'``{attr_key}``'
            attr_types = f'*{element.attributes[attr_key]}*'
        else:
            attr_nodes = attr_types = ""

        printFields([self_node, source_files, child_nodes, child_types, attr_nodes, attr_types])

        if i < numLines-1:
            # add lines between rows, but take care of multi-row segments
            printLine([' ',
                       '-' if i < len(sorted_files)-1 else ' ',
                       '-' if i < len(sorted_children)-1 else ' ',
                       '-' if i < len(sorted_children)-1 else ' ',
                       '-' if i < len(sorted_attributes)-1 else ' ',
                       '-' if i < len(sorted_attributes)-1 else ' '])

        self_node = source_files = ""

    printLine('-'*NUM_COLS)  # end of table

    for node_key in sorted_children:
        child = element.children[node_key]
        full_name = f"{element.name}__{node_key}"
        if not full_name in stack:
            stack.append(full_name)
            makeTableRST(child, stack, nodes, base_url, level+1, key=node_key, with_sections=with_sections)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    bindings_model.add_common_arguments(parser)
    parser.add_argument('-b', '--base-url', metavar='URL', type=str, default="",
                        help='base url for links to files (e.g. GitHub tree)')
    parser.add_argument('--with-sections', action='store_true',
                        help='include section dividers')
    args = parser.parse_args()

    stack = []
    nodes = set()
    for root in bindings_model.load_roots(args):
        makeTableRST(root, stack, nodes, args.base_url, with_sections=args.with_sections)
