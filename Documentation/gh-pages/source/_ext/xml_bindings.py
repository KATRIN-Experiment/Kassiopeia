"""
Sphinx extension embedding the XML bindings documentation (written as //! comments
next to the C++ bindings, see Scripts/bindings/bindings_model.py) into the docs.

Directives:

    .. xml-element:: flattened_circle_surface

        Renders the documentation of an element (description, images, child elements,
        attributes, examples) at this place, with a link to its reference page. The
        argument is an XML tag, "parent_tag/tag" if the tag is bound to different
        elements depending on the parent, or a C++ type name.

    .. xml-element-usages:: KGFlattenedCircleSurface

        Used on the generated reference pages: links to all places where the element
        is documented with xml-element (e.g. in the usage guide).

Configuration (conf.py):

    xml_bindings_json      path of the bindings JSON written by extract.py
    xml_bindings_scripts   path of Scripts/bindings (for the shared renderer)
    xml_bindings_base_url  base URL for links to source files
"""

import os
import sys
import re

from docutils import nodes
from docutils.parsers.rst import Directive
from docutils.statemachine import StringList
from sphinx.util import logging
from sphinx.util.nodes import nested_parse_with_titles

logger = logging.getLogger(__name__)

_cache = {}


def bindings(app):
    path = app.config.xml_bindings_json
    if path not in _cache:
        scripts = app.config.xml_bindings_scripts
        if scripts not in sys.path:
            sys.path.insert(0, scripts)
        import bindings_model
        _cache[path] = bindings_model.load(path)
    return _cache[path]


def resolve(app, spec):
    """Finds the element for 'tag', 'parent_tag/tag' or a C++ type name. Returns (tag, element) or raises ValueError."""
    b = bindings(app)
    if spec in b.elements:
        element = b.elements[spec]
        return (element.tags[0] if element.tags else spec), element
    parent_tag, _, tag = spec.rpartition('/')
    candidates = []
    for element in b.elements.values():
        for child_tag, info in element.child_info.items():
            if child_tag == tag and (not parent_tag or parent_tag in element.tags):
                candidates.append(b.elements[info.type])
    unique = {e.name: e for e in candidates}
    if not unique:
        raise ValueError(f"unknown XML element '{spec}'")
    if len(unique) > 1:
        options = ', '.join(sorted(f"{p.tags[0] if p.tags else p.name}/{tag}" for e in unique.values() for p in e.parents
                                   if any(i.type == e.name and t == tag for t, i in p.child_info.items())))
        raise ValueError(f"XML element '{spec}' is ambiguous, use one of: {options}")
    return tag, next(iter(unique.values()))


class usages_placeholder(nodes.General, nodes.Element):
    pass


class XmlElementDirective(Directive):
    required_arguments = 1
    final_argument_whitespace = True
    has_content = False

    def run(self):
        env = self.state.document.settings.env
        app = env.app
        try:
            tag, element = resolve(app, self.arguments[0])
        except ValueError as e:
            return [self.state.document.reporter.warning(str(e), line=self.lineno)]

        import to_rst
        target_id = 'xml-element-' + re.sub(r'[^a-z0-9]+', '-', element.name.lower()).strip('-')
        target = nodes.target('', '', ids=[target_id])
        target['xml_binding'] = element.name
        target['xml_binding_id'] = target_id  # Sphinx moves the id to the following node

        lines = [f'Reference: {to_rst.ref(element, "<" + tag + ">")}', '']
        lines += to_rst.renderElementBody(tag, element, bindings(app), app.config.xml_bindings_base_url)
        container = nodes.container(classes=['xml-element'])
        nested_parse_with_titles(self.state, StringList(lines, source=self.arguments[0]), container)

        env.note_dependency(os.path.abspath(app.config.xml_bindings_json))
        return [target, container]


class XmlElementUsagesDirective(Directive):
    required_arguments = 1
    final_argument_whitespace = True

    def run(self):
        node = usages_placeholder()
        node['xml_binding'] = self.arguments[0]
        return [node]


def collect_usages(app, doctree):
    """Records where elements are documented with xml-element, with the enclosing section title."""
    env = app.env
    if not hasattr(env, 'xml_binding_usages'):
        env.xml_binding_usages = []
    for target in doctree.findall(nodes.target):
        if 'xml_binding' not in target:
            continue
        section = target.parent
        while section is not None and not isinstance(section, nodes.section):
            section = section.parent
        title = section[0].astext() if section is not None else env.docname
        env.xml_binding_usages.append((target['xml_binding'], env.docname, target['xml_binding_id'], title))


def resolve_usages(app, doctree, docname):
    usages = getattr(app.env, 'xml_binding_usages', [])
    for node in list(doctree.findall(usages_placeholder)):
        found = sorted({(d, i, t) for b, d, i, t in usages if b == node['xml_binding']})
        if not found:
            node.replace_self([])
            continue
        paragraph = nodes.paragraph()
        paragraph += nodes.Text('Explained in: ')
        for k, (doc, target_id, section_title) in enumerate(found):
            if k:
                paragraph += nodes.Text(', ')
            # link text: the title of the page; the link leads to the element's section on it
            page_title = app.env.titles[doc].astext() if doc in app.env.titles else section_title
            uri = app.builder.get_relative_uri(docname, doc) + '#' + target_id
            paragraph += nodes.reference('', page_title, internal=True, refuri=uri, reftitle=section_title)
        node.replace_self(paragraph)


def purge(app, env, docname):
    if hasattr(env, 'xml_binding_usages'):
        env.xml_binding_usages = [u for u in env.xml_binding_usages if u[1] != docname]


def merge(app, env, docnames, other):
    if not hasattr(env, 'xml_binding_usages'):
        env.xml_binding_usages = []
    env.xml_binding_usages += [u for u in getattr(other, 'xml_binding_usages', []) if u[1] in docnames]


def setup(app):
    app.add_config_value('xml_bindings_json', '', 'env')
    app.add_config_value('xml_bindings_scripts', '', 'env')
    app.add_config_value('xml_bindings_base_url', '', 'env')
    app.add_node(usages_placeholder)
    app.add_directive('xml-element', XmlElementDirective)
    app.add_directive('xml-element-usages', XmlElementUsagesDirective)
    app.connect('doctree-read', collect_usages)
    app.connect('doctree-resolved', resolve_usages)
    app.connect('env-purge-doc', purge)
    app.connect('env-merge-info', merge)
    return {'version': '1.0', 'parallel_read_safe': True, 'parallel_write_safe': True}
