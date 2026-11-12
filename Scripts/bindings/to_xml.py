#!/usr/bin/env python3
"""
Renders the bindings JSON as an XML example snippet showing elements, child elements and attributes.
"""

import argparse

import bindings_model

SPACES = '    '


def makeExampleXML(element, stack, level=0, key='', max_children=999, max_attributes=999, include_root=True):
    if not key and element.tags:
        key = element.tags[0]

    if not element.tags:
        if include_root:
            for node_key in list(sorted(element.children.keys()))[:max_children]:
                child = element.children[node_key]
                if not child.name in stack:
                    makeExampleXML(child, stack, level, max_children=max_children, max_attributes=max_attributes)
        return

    indent = level*SPACES

    if not element.children and not element.attributes:
        print(f'{indent}<{key}/>')
        return

    if element.attributes:
        print(f'{indent}<{key}')
        for attr_key in list(sorted(element.attributes.keys()))[:max_attributes]:
            print(f'{indent}{SPACES}{attr_key}="({element.attributes[attr_key]})"')
        print(f'{indent}>')
    else:
        print(f'{indent}<{key}>')

    for node_key in list(sorted(element.children.keys()))[:max_children]:
        child = element.children[node_key]
        full_name = f"{element.name}__{node_key}"
        if not full_name in stack:
            stack.append(full_name)
            makeExampleXML(child, stack, level+1, key=node_key, max_children=max_children, max_attributes=max_attributes)
            print()

    print(f'{indent}</{key}>')


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    bindings_model.add_common_arguments(parser)
    args = parser.parse_args()

    stack = []
    for root in bindings_model.load_roots(args):
        makeExampleXML(root, stack)
