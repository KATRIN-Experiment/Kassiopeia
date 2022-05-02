#!/usr/bin/env python3

import sys
import os
import re
import glob
import argparse

MASTER_ROOT = 'KRoot'

STRIP_NAMESPACES = ['std', 'detail', 'katrin', 'KGeoBag', 'KEMField']

#### https://regex101.com/r/c9ICU1/3
REGEX_PATTERN = r'(\s*(\w+)(Builder|Binding)::(Attribute|SimpleElement|ComplexElement)<\s*([\w]+::)?([\w]+)\s*>\s*\(\s*"(\w+)"\s*\)\s*[+;])'

REGEX_PATTERN_TYPEDEF = r'(typedef\s+K(SimpleElement|ComplexElement)\s*<\s*([\w]+::)?([\w]+)\s*>\s*(\w+)(Builder|Binding)\s*;)'
REGEX_PATTERN_USING = r'(using\s+(\w+)(Builder|Binding)\s*=\s*K(SimpleElement|ComplexElement)\s*<\s*([\w]+::)?([\w]+)\s*>\s*;)'

# GraphViz options
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



node_list = {}

class Node:
    def __init__(self, name, type):
        self.name = name
        self.type = type
        self.source_file = None
        self.xml_name = None
        self.data_type = None

        self.parents = set()
        self.children = set()
        self.attributes = []

    def __repr__(self):
        return self.name

    def __str__(self):
        return f"{self.name} = {self.xml_name}"

    def __lt__(self, other):
        return self.name < other.name

    def pprint(self, level=0, stack=[]):
        indent = level*'  '

        print(indent + f"{self.xml_name} = {self.name}")  #  ({self.source_file})
        for attr in sorted(self.attributes):
            print(indent + '  ' + f"{attr.xml_name} = {attr.name}")
        for node in sorted(self.children):
            if not node.name in stack:
                stack.append(self.name)
                node.pprint(level+1, stack)

    def makeGraph(self, level=0, stack=[], with_children=True, with_attributes=False):
        node_color = NODE_COLORS[(level+1) % len(NODE_COLORS)]
        make_subgraph = (level == 0) or (with_children and self.children) or (with_attributes and self.attributes)

        if make_subgraph:
            print(f'subgraph cluster_{self.name} {{')
            print(f'## {self.children}')
            print(f'label="{self.name}"; fontsize={FONT_SIZE}; fontname="{FONT_FACE},bold"; style=filled; fillcolor="{CLUSTER_COLOR}{CLUSTER_ALPHA}";')
            print(f'node [shape={NODE_SHAPE}, style=filled, fillcolor="{node_color}{NODE_ALPHA}", fontsize={FONT_SIZE}, fontname="{FONT_FACE},bold"];')

        if not self.name in stack:
            print(f'"{self.name}" [label="{self.xml_name}", fontsize={FONT_SIZE}, fontname="{FONT_FACE},bold"];')

        #print(f'"{self.name}";')
        if with_attributes:
            sorted_attributes = sorted(self.attributes)
            for attr in self.attributes:
                print(f'"{self.name}__{attr.xml_name}" [label="{attr.xml_name}", shape={ATTR_SHAPE}, fontsize={FONT_SIZE-1}, fontname="{FONT_FACE}", fillcolor="{ATTR_COLOR}{ATTR_ALPHA}"];')
                #print(f'"{self.name}" -> "{self.name}__{attr.xml_name}" [label="{attr.name}", fontsize={FONT_SIZE}, fontname="{FONT_FACE},italic"];')
                print(f'"{self.name}" -> "{self.name}__{attr.xml_name}" [fontsize={FONT_SIZE}, fontname="{FONT_FACE},italic"];')

        if with_children:
            sorted_children = sorted(self.children)
            child_nodes = ' '.join([f'"{node.name}"' for node in sorted_children])
            if child_nodes:
                print(f'{{ rank=same {child_nodes} }}')

            for node in sorted_children:
                #print(f'node [shape={NODE_SHAPE}, style=filled, fillcolor="{node_color}{NODE_ALPHA}", fontsize={FONT_SIZE}, fontname="{FONT_FACE},bold"];')
                #print(f'"{node.name}" [label="{node.xml_name}", fontsize={FONT_SIZE}, fontname="{FONT_FACE},bold"];')
                #print(f'"{self.name}" -> "{node.name}" [label="{node.xml_name}", fontsize={FONT_SIZE}, fontname="{FONT_FACE}"];')
                if not node.name in stack:
                    stack.append(self.name)
                    print(f'"{self.name}" -> "{node.name}" [fontsize={FONT_SIZE}, fontname="{FONT_FACE}"];')
                    node.makeGraph(level+1, stack, with_children=with_children, with_attributes=with_attributes)

        if make_subgraph:
            print(f'}}')

    def makeExampleXML(self, level=0, stack=[], max_children=999, max_attributes=999, include_root=True):
        spaces = '    '

        if not self.xml_name:
            if include_root:
                for node in list(sorted(self.children))[:max_children]:
                    if not node.name in stack:
                        stack.append(self.name)
                        node.makeExampleXML(level, stack, max_children=max_children, max_attributes=max_attributes)
            return

        indent = level*spaces

        if not self.children and not self.attributes:
           print(f'{indent}<{self.xml_name}/>')
           return

        if self.attributes:
            print(f'{indent}<{self.xml_name}')
            for attr in list(sorted(self.attributes))[:max_attributes]:
                print(f'{indent}{spaces}{attr.xml_name}="({attr.name})"')
            #if not self.children:
            #    print(f'{indent}/>')
            #    return
            print(f'{indent}>')
        else:
            print(f'{indent}<{self.xml_name}>')

        for node in list(sorted(self.children))[:max_children]:
            if not node.name in stack:
                stack.append(self.name)
                node.makeExampleXML(level+1, stack, max_children=max_children, max_attributes=max_attributes)
                print()

        print(f'{indent}</{self.xml_name}>')

    def makeTableRST(self, level=0, stack=[], with_sections=False):
        colWidth = 75
        headings = ['-', '~', '^']
        if with_sections:
            if self.children or self.attributes:
                if level < len(headings):
                    heading = (colWidth*headings[level])
                    print()
                    print(f'{self.name}')
                    print(heading)

                print()
                print(('+%s'%(colWidth*'-'))*6 + '+')
                print((('|%%-%ds'%colWidth)*6)%('element name', 'source file', 'child elements', 'child types', 'attributes', 'attribute types') + '|')
                print(('+%s'%(colWidth*'='))*6 + '+')

        else:
            if level == 0:
                print()
                print(('+%s'%(colWidth*'-'))*6 + '+')
                print((('|%%-%ds'%colWidth)*6)%('element name', 'source file', 'child elements', 'child types', 'attributes', 'attribute types') + '|')
                print(('+%s'%(colWidth*'='))*6 + '+')

        self_node = f'``{self.xml_name}``' if self.xml_name else "—"
        source_file = f'``{os.path.basename(self.source_file)}``' if self.source_file else "—"
        sorted_children = sorted(self.children) if self.children else []
        sorted_attributes = sorted(self.attributes) if self.attributes else []
        for i in range(max(len(sorted_children), len(sorted_attributes))):
            if not sorted_children and i == 0:
                child_nodes = child_types = "—"
            elif i < len(sorted_children):
                node = sorted_children[i]
                child_nodes = f'- ``{node.xml_name}``'
                child_types = f'- ``{node.name}``'
            else:
                child_nodes = child_types = ""
            if not sorted_attributes and i == 0:
                attr_nodes = attr_types = "—"
            elif i < len(sorted_attributes):
                attr = sorted_attributes[i]
                attr_nodes = f'- ``{attr.xml_name}``'
                attr_types = f'- ``{attr.name}``'
            else:
                attr_nodes = attr_types = ""
            print((('|%%-%ds'%colWidth)*6)%(self_node, source_file, child_nodes, child_types, attr_nodes, attr_types) + '|')
            self_node = source_file = ""
        print(('+%s'%(colWidth*'-'))*6 + '+')

        if self.children:
            for node in sorted_children:
                if not node.name in stack:
                    stack.append(self.name)
                    node.makeTableRST(level+1, stack, with_sections=with_sections)

    def makeTableMD(self, level=0, stack=[], with_sections=False, with_examples=False):
        colWidth = 75
        if with_sections:
            if self.children or self.attributes:
                heading = (level+1)*'#'
                print()
                print(heading + f' {self.name}')

                if with_examples:
                    print("Example:")
                    print("```")
                    self.makeExampleXML(max_children=1, max_attributes=3, include_root=False);
                    print("```")

                print()
                print((('|%%-%ds'%colWidth)*6)%('element name', 'source file', 'child elements', 'child types', 'attributes', 'attribute types') + '|')
                print(('|%s'%(colWidth*'-'))*6 + '|')
        else:
            if level == 0:
                print()
                print((('|%%-%ds'%colWidth)*6)%('element name', 'source file', 'child elements', 'child types', 'attributes', 'attribute types') + '|')
                print(('|%s'%(colWidth*'-'))*6 + '|')

        #print('| name | children | attributes |')

        self_node = f'<a name="{self.name.lower()}">`{self.xml_name}`</a>' if self.xml_name else "—"
        source_file = f'[*{os.path.basename(self.source_file)}*]({self.source_file})' if self.source_file else "—"
        if self.children:
            sorted_children = sorted(self.children)
            child_nodes = '<br>'.join([f'[`{node.xml_name}`](#{node.name.lower()})' for node in sorted_children])
            child_types = '<br>'.join([f'[*`{node.name}`*](#{node.name.lower()})' for node in sorted_children])
        else:
            child_nodes = child_types = "—"
        if self.attributes:
            sorted_attributes = sorted(self.attributes)
            attr_nodes = '<br>'.join([f'`{attr.xml_name}`' for attr in sorted_attributes])
            attr_types = '<br>'.join([f'`{attr.name}`' for attr in sorted_attributes])
        else:
            attr_nodes = attr_types = "—"
        print((('|%%-%ds'%colWidth)*6)%(self_node, source_file, child_nodes, child_types, attr_nodes, attr_types) + '|')

        if self.children:
            for node in sorted_children:
                if not node.name in stack:
                    stack.append(self.name)
                    node.makeTableMD(level+1, stack, with_sections=with_sections, with_examples=with_examples)

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
    node_aliases = {}
    node_list = {}

    # first process header files
    for file_name in file_list:
        with open(file_name) as f:
            buffer = ' '.join(f.readlines())

            match = re.findall(REGEX_PATTERN_TYPEDEF, buffer)
            for m in match:
                type, prefix, target, alias, builder = m[1:]
                #print(f"{target} := {alias} ({type})")
                if prefix and prefix[:-2] not in STRIP_NAMESPACES:
                    target = prefix[:-2] + target

                if not target in node_list:
                    node_list[target] = Node(target, type)
                if alias != target:
                    node_aliases[alias] = target

            match = re.findall(REGEX_PATTERN_USING, buffer)
            for m in match:
                alias, builder, type, prefix, target = m[1:]
                #print(f"{target} := {alias} ({type})")
                if prefix and prefix[:-2] not in STRIP_NAMESPACES:
                    target = prefix[:-2] + target

                if not target in node_list:
                    node_list[target] = Node(target, type)
                if alias != target:
                    node_aliases[alias] = target

    # now process source files
    for file_name in file_list:
        #print(f'** {file_name}:')
        with open(file_name) as f:
            buffer = ' '.join(f.readlines())

            match = re.findall(REGEX_PATTERN, buffer)
            for m in match:
                source, builder, type, prefix, target, name = m[1:]
                if source in node_aliases:
                    source = node_aliases[source]
                if prefix and prefix[:-2] not in STRIP_NAMESPACES:
                    target = prefix[:-2] + target
                #print(f"{source} -> {target} ({type}) [{name}]")

                if not source in node_list:
                    node_list[source] = Node(source, 'ComplexElement')

                if type == 'Attribute':
                    attr = Node(target, type)
                    attr.xml_name = name
                    attr.parents.add(node_list[source])
                    node_list[source].attributes.append(attr)
                elif type == 'ComplexElement' or type == 'SimpleElement':
                    if not target in node_list:
                        node_list[target] = Node(target, type)
                    if not node_list[target].source_file:
                        node_list[target].source_file = file_name
                    if not node_list[target].xml_name:
                        node_list[target].xml_name = name
                    node_list[target].parents.add(node_list[source])
                    node_list[source].children.add(node_list[target])
                else:
                    raise RuntimeError("Unrecognized element type: %s" % type)

    return (node_list, node_aliases)

def findRootNodes(node_list):
    root_nodes = []
    if MASTER_ROOT in node_list:
        for node in node_list[MASTER_ROOT].children:
            root_nodes.append(node)
    else:
        for node in node_list.values():
            if node.parents == set():
                root_nodes.append(node)
    return root_nodes

def parseArguments():
    parser = argparse.ArgumentParser(description='Process some integers.')
    parser.add_argument('path(s)', metavar='N', type=str, nargs='+',
                        help='base paths to search for bindings files')
    parser.add_argument('-r', '--root', metavar='NAME', type=str,
                        help='root node of the resulting tree')
    parser.add_argument('--xml', action='store_true',
                        help='produce XML examples (.xml file)')
    parser.add_argument('--gv', action='store_true',
                        help='produce GraphViz tree (.dot file)')
    parser.add_argument('--rst', action='store_true',
                        help='produce reStructuredText tables (.rst file)')
    parser.add_argument('--md', action='store_true',
                        help='produce Markdown tables (.md file)')
    parser.add_argument('--with-sections', action='store_true',
                        help='include section dividers in reStructuredText/Markdown output')
    parser.add_argument('--with-examples', action='store_true',
                        help='include syntax exaples in reStructuredText/Markdown output')
    parser.add_argument('--with-attributes', action='store_true',
                        help='include element attributes in GraphViz output')

    args = parser.parse_args()
    return args

if __name__ == "__main__":
    args = parseArguments()

    file_list = getBindingsFiles(sys.argv[1:])
    if not file_list:
        print("The provided path doesn't contain any bindings files.")
        sys.exit(1)

    node_list, node_aliases = processFiles(file_list)
    if not node_list:
        print("No valid bindings were found in the input files.")
        sys.exit(1)

    root_node_list = []
    if args.root:
        if args.root in node_list:
            if args.root in node_aliases:
                root_node_list.append(node_list[node_aliases[args.root]])
            else:
                root_node_list.append(node_list[args.root])
        else:
            raise KeyError("Root node %s was not found in bindings." % args.root)
    else:
        root_node_list = findRootNodes(node_list)

    if not root_node_list:
        print("Root node not found! Listing all nodes instead:")
        for node in node_list.values():
            print(node)
        sys.exit(0)

    if args.gv:
        print('digraph {')
        print('rankdir=LR; penwidth=2; splines=spline;')
        for root_node in root_node_list:
            root_node.makeGraph(with_attributes=args.with_attributes)
        print('}')
    elif args.rst:
        for root_node in root_node_list:
            root_node.makeTableRST(with_sections=args.with_sections)
    elif args.md:
        for root_node in root_node_list:
            root_node.makeTableMD(with_sections=args.with_sections, with_examples=args.with_examples)
    elif args.xml:
        for root_node in root_node_list:
            root_node.makeExampleXML()
    else:
        for root_node in root_node_list:
            root_node.pprint()
