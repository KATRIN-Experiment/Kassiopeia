#!/usr/bin/env python3

import sys
import os
import re
import glob
import argparse

MASTER_ROOT = 'KRoot'

STRIP_NAMESPACES = ['std', 'detail', 'katrin', 'KGeoBag', 'KEMField']

#### https://regex101.com/r/ih2tjR/1
REGEX_PATTERN = r'(\s*(\w+)(Builder|Binding)::(Attribute|SimpleElement|ComplexElement)<\s*([\w]+::)?([\w<>]+)\s*>\s*\(\s*"(\w+)"\s*\)\s*[+;])'

REGEX_PATTERN_TYPEDEF = r'(typedef\s+K(SimpleElement|ComplexElement)\s*<\s*([\w]+::)?([\w<>]+)\s*>\s*(\w+)(Builder|Binding)\s*;)'
REGEX_PATTERN_USING = r'(using\s+(\w+)(Builder|Binding)\s*=\s*K(SimpleElement|ComplexElement)\s*<\s*([\w]+::)?([\w<>]+)\s*>\s*;)'

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
        self.data_type = None
        self.source_files = set()
        self.xml_names = set()

        self.parents = set()
        self.children = dict()
        self.attributes = dict()

    def __repr__(self):
        return self.name

    def __str__(self):
        return f"{self.name} = {self.xml_names}"

    def __lt__(self, other):
        return self.name < other.name

    """
    Procuces node documentation in simple ASCII format.
    """
    def pprint(self, level=0, stack=[], key=''):
        indent = level*'  '
        if not key and self.xml_names:
            key = list(self.xml_names)[0]

        #print(indent + f"{key} = {self.name}")  #  ({self.source_file})
        for attr_key in sorted(self.attributes.keys()):
            attr = self.attributes[attr_key]
            print(indent + f"{attr_key} = {attr.name}")

        for node_key in sorted(self.children.keys()):
            node = self.children[node_key]
            full_name = f'"{node.name}__{node_key}'
            if not full_name in stack:
                stack.append(full_name)
                print(indent + f"{node_key} = {node.name}")  #  ({self.source_file})
                node.pprint(level+1, stack, key=node_key)

    """
    Produces node documentation in GraphViz (DOT) format. Optionally includes child elements and attribute nodes.
    """
    def makeGraph(self, level=0, stack=[], nodes=set(), key='', with_children=True, with_attributes=False):
        node_color = NODE_COLORS[(level+1) % len(NODE_COLORS)]
        make_subgraph = (level == 0) or (with_children and self.children) or (with_attributes and self.attributes)
        if not key and self.xml_names:
            key = list(self.xml_names)[0]

        cleanStr = lambda txt: txt.translate(dict([ (ord(c),'_') for c in "<>-" ]))  # strip invalid chars

        full_name = cleanStr(f'{self.name}__{key}')
        if full_name in nodes:
            return nodes
        nodes.add(full_name)

        if make_subgraph:
            print(f'subgraph cluster_{cleanStr(self.name)} {{')
            print(f'label="{self.name}"; fontsize={FONT_SIZE}; fontname="{FONT_FACE},bold"; style=filled; fillcolor="{CLUSTER_COLOR}{CLUSTER_ALPHA}";')
            print(f'node [shape={NODE_SHAPE}, style=filled, fillcolor="{node_color}{NODE_ALPHA}", fontsize={FONT_SIZE}, fontname="{FONT_FACE},bold"];')

        print(f'"{full_name}" [label="{key}", fontsize={FONT_SIZE}, fontname="{FONT_FACE},bold"];')
        if not full_name in stack:
            stack.append(full_name)

        if with_attributes:
            sorted_keys = sorted(self.attributes.keys())
            for attr_key in sorted_keys:
                attr = self.attributes[attr_key]
                print(f'"{full_name}__{attr_key}" [label="{attr_key}", shape={ATTR_SHAPE}, fontsize={FONT_SIZE-1}, fontname="{FONT_FACE}", fillcolor="{ATTR_COLOR}{ATTR_ALPHA}"];')
                print(f'"{full_name}" -> "{full_name}__{attr_key}" [fontsize={FONT_SIZE}, fontname="{FONT_FACE},italic"];')

        if with_children:
            sorted_keys = sorted(self.children.keys())
            child_nodes = ' '.join([cleanStr(f'"{self.children[key].name}__{key}"') for key in sorted_keys])
            if child_nodes:
                print(f'{{ rank=same {child_nodes} }}')

            for node_key in sorted_keys:
                node = self.children[node_key]
                node_full_name = cleanStr(f'{node.name}__{node_key}')
                if not node_full_name in stack:
                    stack.append(full_name)
                print(f'"{full_name}" -> "{node_full_name}" [fontsize={FONT_SIZE}, fontname="{FONT_FACE}"];')
                new_nodes = node.makeGraph(level+1, stack, nodes, key=node_key, with_children=with_children, with_attributes=with_attributes)
                nodes.union(new_nodes)

        if make_subgraph:
            print(f'}}')

        return nodes

    """
    Produces node documentation in XML format (example snippet). Includes up a given number of child elements.
    """
    def makeExampleXML(self, level=0, stack=[], key='', max_children=999, max_attributes=999, include_root=True):
        spaces = '    '
        if not key and self.xml_names:
            key = list(self.xml_names)[0]

        if not self.xml_names:
            if include_root:
                for node_key in list(sorted(self.children.keys()))[:max_children]:
                    node = self.children[node_key]
                    if not node.name in stack:
                        node.makeExampleXML(level, stack, max_children=max_children, max_attributes=max_attributes)
            return

        indent = level*spaces

        if not self.children and not self.attributes:
           print(f'{indent}<{key}/>')
           return

        if self.attributes:
            print(f'{indent}<{key}')
            for attr_key in list(sorted(self.attributes.keys()))[:max_attributes]:
                attr = self.attributes[attr_key]
                print(f'{indent}{spaces}{attr_key}="({attr.name})"')
            #if not self.children:
            #    print(f'{indent}/>')
            #    return
            print(f'{indent}>')
        else:
            print(f'{indent}<{key}>')

        for node_key in list(sorted(self.children.keys()))[:max_children]:
            node = self.children[node_key]
            full_name = f"{self.name}__{node_key}"
            if not full_name in stack:
                stack.append(full_name)
                node.makeExampleXML(level+1, stack, key=node_key, max_children=max_children, max_attributes=max_attributes)
                print()

        print(f'{indent}</{key}>')

    """
    Produces node documentation in reStructuredText (RST) format. Includes child elements. Section breaks are optional.
    """
    def makeTableRST(self, level=0, stack=[], nodes=set(), key='', with_sections=False):
        global base_url
        numCols = 6
        colWidths = (80, 240, 80, 160, 80, 40)  # in RST the columns all must have the same width
        colHeaders = ('element name', 'source files', 'child elements', 'child types', 'attributes', 'attribute types')
        headSep = ["-", "~", "^", "'"]  # starts at level 3
        if not key and self.xml_names:
            key = list(self.xml_names)[0]

        escapeStr = lambda txt: txt.translate(dict([ (ord(c),f'\{c}') for c in "<>" ]))  # escape invalid chars
        cleanStr = lambda txt: txt.translate(dict([ (ord(c),f'_') for c in "<>" ]))  # strip invalid chars

        full_name = cleanStr(f'{self.name}')
        if full_name in nodes:
            return nodes
        nodes.add(full_name)

        def printLine(separators):
            assert(len(separators) == numCols)
            print(''.join([('+%s'%(colWidths[i]*separators[i])) for i in range(numCols)]) + '+')

        def printFields(fields):
            assert(len(fields) == numCols)
            print(''.join([(('|%%-%ds'%colWidths[i])%fields[i]) for i in range(numCols)]) + '|')

        if with_sections:
            if self.children or self.attributes:
                if level < len(headSep):
                    print()
                    print(f'.. _{cleanStr(self.name)}:')
                    print()
                    print(f'{self.name}')
                    print(100*headSep[level])

                print()
                printLine('-'*numCols)
                printFields(colHeaders)
                printLine('='*numCols)  # header separator

        else:
            if level == 0:
                print()
                printLine('-'*numCols)
                printFields(colHeaders)
                printLine('='*numCols)  # header separator

        self_node = f'``{key}``' if key else "—"
        sorted_files = sorted(list(self.source_files)) if self.source_files else []
        sorted_children = sorted(self.children.keys()) if self.children else []
        sorted_attributes = sorted(self.attributes.keys()) if self.attributes else []

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
                node = self.children[node_key]
                child_nodes = f'``{node_key}``'
                child_types = f':ref:`{escapeStr(node.name)} <{cleanStr(node.name)}>`'
            else:
                child_nodes = child_types = ""

            if not sorted_attributes and i == 0:
                attr_nodes = attr_types = "—"
            elif i < len(sorted_attributes):
                attr_key = sorted_attributes[i]
                attr = self.attributes[attr_key]
                attr_nodes = f'``{attr_key}``'
                attr_types = f'*{attr.name}*'
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

        printLine('-'*numCols)  # end of table

        if self.children:
            for node_key in sorted_children:
                node = self.children[node_key]
                full_name = f"{self.name}__{node_key}"
                if not full_name in stack:
                    stack.append(full_name)
                    new_nodes = node.makeTableRST(level+1, stack, nodes, key=node_key, with_sections=with_sections)
                    nodes.union(new_nodes)

        return nodes

    """
    Produces node documentation in MarkDown (MD) format. Includes child elements. Section breaks and XML snippets are optional.
    """
    def makeTableMD(self, level=0, stack=[], nodes=set(), key='', with_sections=False, with_examples=False):
        global base_url
        numCols = 6
        colWidth = 5
        colHeaders = ('element name', 'source files', 'child elements', 'child types', 'attributes', 'attribute types')
        if not key and self.xml_names:
            key = list(self.xml_names)[0]

        escapeStr = lambda txt: txt.translate(dict([ (ord(c),f'\{c}') for c in "<>" ]))  # escape invalid chars
        cleanStr = lambda txt: txt.translate(dict([ (ord(c),f'_') for c in "<>" ]))  # strip invalid chars

        full_name = cleanStr(f'{self.name}')
        if full_name in nodes:
            return nodes
        nodes.add(full_name)

        if with_sections:
            if self.children or self.attributes or self.xml_names:
                heading = (level+1)*'#'
                print()
                print(f'{heading} {escapeStr(self.name)}')

                if with_examples and self.xml_names:
                    print("Example:")
                    print("```")
                    self.makeExampleXML(max_children=1, max_attributes=3, include_root=False);
                    print("```")

                print()
                print((('|%%-%ds'%colWidth)*numCols)%colHeaders + '|')
                print(('|%s'%(colWidth*'-'))*numCols + '|')
        else:
            if level == 0:
                print()
                print((('|%%-%ds'%colWidth)*numCols)%colHeaders + '|')
                print(('|%s'%(colWidth*'-'))*numCols + '|')

        #print('| name | children | attributes |')

        self_node = f'<a name="{escapeStr(self.name.lower())}">`{key}`</a>' if key else "—"

        if self.source_files:
            sorted_files = sorted(list(self.source_files))
            source_files = '<br>'.join([f'[*{os.path.basename(file)}*]({base_url}{file})' for file in sorted_files])
        else:
            source_files = "—"

        if self.children:
            sorted_children = sorted(self.children.keys())
            child_nodes = '<br>'.join([f'[`{key}`](#{escapeStr(self.children[key].name.lower())})' for key in sorted_children])
            child_types = '<br>'.join([f'*`{self.children[key].name}`*' for key in sorted_children])
        else:
            child_nodes = child_types = "—"

        if self.attributes:
            sorted_attributes = sorted(self.attributes.keys())
            attr_nodes = '<br>'.join([f'`{key}`' for key in sorted_attributes])
            attr_types = '<br>'.join([f'*`{self.attributes[key].name}`*' for key in sorted_attributes])
        else:
            attr_nodes = attr_types = "—"

        print((('|%%-%ds'%colWidth)*numCols)%(self_node, source_files, child_nodes, child_types, attr_nodes, attr_types) + '|')

        if self.children:
            for node_key in sorted_children:
                node = self.children[node_key]
                full_name = f"{self.name}__{node_key}"
                if not full_name in stack:
                    stack.append(full_name)
                    new_nodes = node.makeTableMD(level+1, stack, key=node_key, with_sections=with_sections, with_examples=with_examples)
                    nodes.union(new_nodes)

        return nodes

    ## END OF class Node

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
                    attr.xml_names.add(name)
                    attr.parents.add(node_list[source])
                    node_list[source].attributes[name] = attr

                elif type == 'ComplexElement' or type == 'SimpleElement':
                    if not target in node_list:
                        node_list[target] = Node(target, type)
                    node_list[target].xml_names.add(name)
                    node_list[target].source_files.add(file_name)
                    node_list[target].parents.add(node_list[source])
                    node_list[source].children[name] = node_list[target]

                else:
                    raise RuntimeError("Unrecognized element type: %s" % type)

    return (node_list, node_aliases)

def findRootNodes(node_list):
    root_nodes = []
    if MASTER_ROOT in node_list:
        for node in node_list[MASTER_ROOT].children.values():
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
    parser.add_argument('-b', '--base-url', metavar='URL', type=str, default="",
                        help='base url for links to files (e.g. GitHub tree)')
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

    base_url = args.base_url

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
