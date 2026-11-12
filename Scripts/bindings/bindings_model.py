"""
Shared data model for the XML bindings JSON written by extract.py.

JSON format (format_version 1):

    {
      "format_version": 1,
      "elements": {
        "<C++ type>": {
          "kind": "ComplexElement" | "SimpleElement",
          "tags": ["<XML tag name>", ...],          # names this type is bound to
          "source_files": ["<path>", ...],          # files registering this type as a child
          "children": {"<XML tag name>": "<C++ type>", ...},
          "attributes": {"<attribute name>": "<C++ type>", ...}
        },
        ...
      },
      "aliases": {"<builder name>": "<C++ type>", ...}
    }

Renderers load this with load() and select their starting points with
select_roots().
"""

import sys
import json

FORMAT_VERSION = 1

MASTER_ROOT = 'KRoot'


class Element:
    def __init__(self, name, kind, tags, source_files, attributes):
        self.name = name
        self.kind = kind
        self.tags = tags                  # sorted list of XML tag names
        self.source_files = source_files  # sorted list of paths
        self.attributes = attributes      # attribute name -> C++ type name
        self.children = {}                # XML tag name -> Element
        self.parents = []

    def __repr__(self):
        return self.name

    def __lt__(self, other):
        return self.name < other.name


class Bindings:
    def __init__(self, elements, aliases):
        self.elements = elements  # C++ type name -> Element
        self.aliases = aliases    # builder name -> C++ type name


def load(path):
    """Loads a bindings JSON file ('-' for standard input)."""
    if path == '-':
        data = json.load(sys.stdin)
    else:
        with open(path) as f:
            data = json.load(f)

    version = data.get('format_version')
    if version != FORMAT_VERSION:
        raise ValueError(f"Unsupported bindings format version {version} (expected {FORMAT_VERSION})")

    elements = {}
    for name, entry in data['elements'].items():
        elements[name] = Element(name, entry['kind'], entry['tags'], entry['source_files'], entry['attributes'])

    for name, entry in data['elements'].items():
        for tag, child_name in entry['children'].items():
            child = elements[child_name]
            elements[name].children[tag] = child
            if elements[name] not in child.parents:
                child.parents.append(elements[name])

    return Bindings(elements, data.get('aliases', {}))


def select_roots(bindings, root=None):
    """
    Returns the elements to start rendering from: the given root element
    (C++ type or builder name), else the children of KRoot, else all
    elements without parents.
    """
    elements = bindings.elements
    if root:
        name = bindings.aliases.get(root, root)
        if name not in elements:
            raise KeyError("Root node %s was not found in bindings." % root)
        return [elements[name]]

    if MASTER_ROOT in elements:
        return list(elements[MASTER_ROOT].children.values())

    return [element for element in elements.values() if not element.parents]


def add_common_arguments(parser):
    parser.add_argument('input', metavar='JSON', type=str,
                        help="bindings JSON file written by extract.py ('-' for standard input)")
    parser.add_argument('-r', '--root', metavar='NAME', type=str,
                        help='root node of the resulting tree')


def load_roots(args):
    """Loads the JSON named in the arguments and returns the root elements; exits if there are none."""
    bindings = load(args.input)
    roots = select_roots(bindings, args.root)
    if not roots:
        print("Root node not found! Listing all nodes instead:", file=sys.stderr)
        for element in bindings.elements.values():
            print(f"{element.name} = {element.tags}", file=sys.stderr)
        sys.exit(1)
    return roots
