"""
Shared data model for the XML bindings JSON written by extract.py.

JSON format (format_version 2):

    {
      "format_version": 2,
      "elements": {
        "<C++ type>": {
          "kind": "ComplexElement" | "SimpleElement",
          "tags": ["<XML tag name>", ...],          # names this type is bound to
          "source_files": ["<path>", ...],          # files registering this type as a child
          "description": "<text>" | null,           # from //! comments above the builder typedef/using
          "examples": ["<XML snippet>", ...],       # from //! @example
          "images": [{"path": "<path>", "width": "<width>" | null,   # from //! @image, shown before
                      "line": <line of the description>}, ...]     # that line of the description
          "deprecated": "<text>" | null,            # from //! @deprecated
          "defined_at": "<path>:<line>" | null,     # builder typedef/using
          "creates": "<C++ type>" | null,           # type of the object actually created, if the element
                                                    # type is only a data wrapper (detected from Set() in End())
          "bases": ["<C++ class>", ...],            # all base classes of the type (or of "creates")
          "children": {
            "<XML tag name>": {
              "type": "<C++ type>",
              "description": "<text>" | null,       # from //! comments above the registration
              "deprecated": "<text>" | null,
              "reference": null | {"type": "<C++ type>"} | {"geometry": "surfaces" | "spaces" | ...},
              "values": null | {"values": ["<value>", ...], "open": <bool>},
                                                    # choice attributes: values compared with in the
                                                    # builder, or given with //! @values a, b, c
                                                    # (open: other values are possible as well)
                                                    # attributes naming another object (detected from
                                                    # KToolbox::Get<T> / KGInterface::Retrieve* in the
                                                    # builder, or given with //! @reference)
              "registered_at": "<path>:<line>"
            }, ...
          },
          "attributes": {
            "<attribute name>": { same fields as children, "type" is the attribute's C++ type }, ...
          }
        },
        ...
      },
      "aliases": {"<builder name>": "<C++ type>", ...}
    }

Documentation comments in the C++ builder sources (see extract.py):

    //! Description of the element (several lines possible, reStructuredText).
    //! @example <element attribute="value"/>
    //! @image Documentation/gh-pages/source/_images/element.png 400pt
    //! @attribute name: description, for attributes registered by macros
    //! @child tag: description, for child elements registered by macros
    //! @deprecated what to use instead
    typedef KComplexElement<KType> KTypeBuilder;

    STATICINT sKTypeStructure =
        //! Description of this attribute.
        KTypeBuilder::Attribute<double>("value") +
        //! @deprecated use <new_child> instead
        KTypeBuilder::ComplexElement<KChild>("old_child") +
        //! Name of the magnetic field to use.
        //! @reference KSMagneticField
        KTypeBuilder::Attribute<std::string>("magnetic_field") +
        //! @values linear, exponential, ...
        KTypeBuilder::Attribute<std::string>("ramping_type");

An @image line is shown at its position in the description (which continues
after it); @example blocks continue until the next @ tag and are shown after
the attribute table (an example in the middle of the text can be written as a
".. code-block:: xml" in the description). @image paths are relative to the repository root; extract.py fails if an
image does not exist. Images must be inside the documentation sources
(Documentation/gh-pages/source) to be shown there.

A comment documents the registration that starts directly below it (if
several registrations share that line, the first one).

Renderers load this with load() and select their starting points with
select_roots().
"""

import sys
import json

FORMAT_VERSION = 2

MASTER_ROOT = 'KRoot'


class Relation:
    """An attribute or child element registration of an element."""
    def __init__(self, entry):
        self.type = entry['type']                    # C++ type name
        self.description = entry.get('description')
        self.deprecated = entry.get('deprecated')
        self.registered_at = entry.get('registered_at')
        self.reference = entry.get('reference')
        self.values = entry.get('values')


class Element:
    def __init__(self, name, entry):
        self.name = name
        self.kind = entry['kind']
        self.tags = entry['tags']                  # sorted list of XML tag names
        self.source_files = entry['source_files']  # sorted list of paths
        self.description = entry.get('description')
        self.examples = entry.get('examples', [])
        self.images = entry.get('images', [])
        self.deprecated = entry.get('deprecated')
        self.defined_at = entry.get('defined_at')
        self.bases = entry.get('bases', [])
        self.attribute_info = {k: Relation(v) for k, v in entry['attributes'].items()}  # name -> Relation
        self.child_info = {k: Relation(v) for k, v in entry['children'].items()}        # tag -> Relation
        self.attributes = {k: v.type for k, v in self.attribute_info.items()}  # attribute name -> C++ type name
        self.children = {}                         # XML tag name -> Element
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
        elements[name] = Element(name, entry)

    for name, element in elements.items():
        for tag, relation in element.child_info.items():
            child = elements[relation.type]
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
