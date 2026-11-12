#!/usr/bin/env python3
"""
Renders the bindings JSON as a GraphViz (DOT) graph. Optionally includes attribute nodes.
"""

import argparse

import bindings_model

FONT_SIZE = 11
FONT_FACE = "sans-serif"

#### import seaborn as sns
#### print([ '#%02x%02x%02x' % (int(255*x[0]),int(255*x[1]),int(255*x[2])) for x in sns.color_palette("bright") ])
NODE_COLORS = ['#023eff', '#ff7c00', '#1ac938', '#e8000b', '#8b2be2', '#9f4800', '#f14cc1', '#a3a3a3', '#ffc400', '#00d7ff']
NODE_ALPHA = 'a0'
NODE_SHAPE = 'box'

ATTR_COLOR = '#f0f0f0'
ATTR_ALPHA = 'a0'
ATTR_SHAPE = 'box'

CLUSTER_COLOR = '#a0a0a0'
CLUSTER_ALPHA = '20'


def cleanStr(txt):
    return txt.translate(dict([(ord(c), '_') for c in "<>-:, "]))  # strip invalid chars


def makeGraph(element, stack, nodes, level=0, key='', with_children=True, with_attributes=False):
    node_color = NODE_COLORS[(level+1) % len(NODE_COLORS)]
    make_subgraph = (level == 0) or (with_children and element.children) or (with_attributes and element.attributes)
    if not key and element.tags:
        key = element.tags[0]

    full_name = cleanStr(f'{element.name}__{key}')
    if full_name in nodes:
        return
    nodes.add(full_name)

    if make_subgraph:
        print(f'subgraph cluster_{cleanStr(element.name)} {{')
        print(f'label="{element.name}"; fontsize={FONT_SIZE}; fontname="{FONT_FACE},bold"; style=filled; fillcolor="{CLUSTER_COLOR}{CLUSTER_ALPHA}";')
        print(f'node [shape={NODE_SHAPE}, style=filled, fillcolor="{node_color}{NODE_ALPHA}", fontsize={FONT_SIZE}, fontname="{FONT_FACE},bold"];')

    print(f'"{full_name}" [label="{key}", fontsize={FONT_SIZE}, fontname="{FONT_FACE},bold"];')
    if not full_name in stack:
        stack.append(full_name)

    if with_attributes:
        for attr_key in sorted(element.attributes.keys()):
            print(f'"{full_name}__{attr_key}" [label="{attr_key}", shape={ATTR_SHAPE}, fontsize={FONT_SIZE-1}, fontname="{FONT_FACE}", fillcolor="{ATTR_COLOR}{ATTR_ALPHA}"];')
            print(f'"{full_name}" -> "{full_name}__{attr_key}" [fontsize={FONT_SIZE}, fontname="{FONT_FACE},italic"];')

    if with_children:
        sorted_keys = sorted(element.children.keys())
        child_nodes = ' '.join([cleanStr(f'"{element.children[k].name}__{k}"') for k in sorted_keys])
        if child_nodes:
            print(f'{{ rank=same {child_nodes} }}')

        for node_key in sorted_keys:
            child = element.children[node_key]
            node_full_name = cleanStr(f'{child.name}__{node_key}')
            if not node_full_name in stack:
                stack.append(full_name)
            print(f'"{full_name}" -> "{node_full_name}" [fontsize={FONT_SIZE}, fontname="{FONT_FACE}"];')
            makeGraph(child, stack, nodes, level+1, key=node_key, with_children=with_children, with_attributes=with_attributes)

    if make_subgraph:
        print(f'}}')


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    bindings_model.add_common_arguments(parser)
    parser.add_argument('--with-attributes', action='store_true',
                        help='include element attributes')
    args = parser.parse_args()

    stack = []
    nodes = set()
    print('digraph {')
    print('rankdir=LR; penwidth=2; splines=spline;')
    for root in bindings_model.load_roots(args):
        makeGraph(root, stack, nodes, with_attributes=args.with_attributes)
    print('}')
