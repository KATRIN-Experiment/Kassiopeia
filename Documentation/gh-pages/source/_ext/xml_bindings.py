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
        is documented with xml-element (e.g. in the usage guide), and to the XML
        examples elsewhere in the documentation that use it ("Appears in the documentation").

In HTML output, the tags in all XML examples link to the documentation of their
elements (resolved along the nesting of the example).

Configuration (conf.py):

    xml_bindings_json      path of the bindings JSON written by extract.py
    xml_bindings_scripts   path of Scripts/bindings (for the shared renderer)
    xml_bindings_base_url  base URL for links to source files
"""

import os
import sys
import re
import html

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


# preprocessor elements (see Kommon/Core/Initialization): children of the "transparent" ones belong to
# the enclosing element, the others have no binding documentation
TRANSPARENT = {'tag', 'if', 'else', 'loop'}
PREPROCESSOR = TRANSPARENT | {'define', 'undefine', 'redefine', 'append', 'prepend', 'include', 'print', 'warning',
                              'error', 'assert', 'global_define', 'global_undefine', 'global_redefine',
                              'global_append', 'global_prepend', 'external_define', 'external_undefine',
                              'external_redefine', 'external_append', 'external_prepend', 'local_define'}


def unique_element(app, tag):
    """The element bound to a tag, if the tag is bound to exactly one element type."""
    cache = _cache.setdefault(('by_tag', app.config.xml_bindings_json), {})
    if not cache:
        b = bindings(app)
        for element in b.elements.values():
            for child_tag, info in element.child_info.items():
                cache.setdefault(child_tag, set()).add(info.type)
    types = cache.get(tag, set())
    return bindings(app).elements[next(iter(types))] if len(types) == 1 else None


def xml_tags(app, code, closing=False):
    """Returns the bound element (or None) for every opening tag in an XML snippet, in order (with
    closing=True also for every closing tag, as the element of its opening tag). Tags are resolved
    along the nesting (e.g. <start_point> inside <poly_loop>); a tag without known parent is
    resolved if it is bound to only one element type."""
    from pygments.lexers import XmlLexer
    from pygments.token import Name
    result, stack, opened = [], [], []  # stack: context for children; opened: element of each open tag
    for token, value in XmlLexer().get_tokens(code):
        if token is not Name.Tag:
            continue
        if value.startswith('</'):
            if stack:
                stack.pop()
            element = opened.pop() if opened else None
            if closing:
                result.append(element)
        elif value.startswith('<'):
            tag = value[1:].strip()
            parent = stack[-1] if stack else None
            if tag in TRANSPARENT:
                element, context = None, parent
            elif tag in PREPROCESSOR:
                element = context = None
            else:
                element = parent.children.get(tag) if parent is not None else None
                element = element or unique_element(app, tag)
                context = element
            result.append(element)
            stack.append(context)
            opened.append(element)
        elif value.strip() == '/>':
            if stack:
                stack.pop()
            if opened:
                opened.pop()
    return result


def xml_blocks(doctree):
    for node in doctree.findall(nodes.literal_block):
        if node.get('language') == 'xml':
            yield node


def enclosing_section(node):
    while node is not None and not isinstance(node, nodes.section):
        node = node.parent
    return node


def collect_usages(app, doctree):
    """Records where elements are documented with xml-element, and where they appear in XML examples."""
    env = app.env
    if not hasattr(env, 'xml_binding_usages'):
        env.xml_binding_usages = []
    if not hasattr(env, 'xml_binding_examples'):
        env.xml_binding_examples = []
    for target in doctree.findall(nodes.target):
        if 'xml_binding' not in target:
            continue
        section = enclosing_section(target.parent)
        title = section[0].astext() if section is not None else env.docname
        env.xml_binding_usages.append((target['xml_binding'], env.docname, target['xml_binding_id'], title))
    # the element documented by a reference page (its examples are also shown where it is explained)
    page_element = next((n['xml_binding'] for n in doctree.findall(usages_placeholder)), None)
    for block in xml_blocks(doctree):
        section = enclosing_section(block.parent)
        section_id = section['ids'][0] if section is not None and section['ids'] else ''
        title = section[0].astext() if section is not None else env.docname
        for element in {e.name for e in xml_tags(app, block.astext()) if e is not None}:
            env.xml_binding_examples.append((element, env.docname, section_id, title, page_element))


def element_uri(app, element, fromdoc):
    """URI of the documentation of an element, or None if it is documented on the page itself."""
    import to_rst
    std = app.env.get_domain('std')
    found = std.labels.get(to_rst.label(element))
    if not found or found[0] == fromdoc:
        return None
    docname, labelid, _ = found
    return app.builder.get_relative_uri(fromdoc, docname) + '#' + labelid


def link_examples(app, doctree, docname):
    """Makes the tags in XML examples link to the documentation of their elements (HTML only)."""
    if app.builder.format != 'html':
        return
    for block in list(xml_blocks(doctree)):
        if block.get('linenos') or block.get('highlight_args', {}).get('hl_lines'):
            continue
        code = block.astext()
        elements = xml_tags(app, code, closing=True)
        if not any(elements):
            continue
        highlighted = app.builder.highlighter.highlight_block(code, 'xml', location=block)
        tags = iter(elements)

        def link(m):
            element = next(tags, None)
            uri = element_uri(app, element, docname) if element is not None else None
            if not uri:
                return m.group(0)
            title = html.escape(element.name)
            return (f'<span class="nt">&lt;{m.group(1)}<a class="reference internal" href="{uri}" title="{title}">'
                    f'{m.group(2)}</a>{m.group(3)}</span>')

        # opening tags ("&lt;name") and closing tags ("&lt;/name&gt;"), in document order
        highlighted = re.sub(r'<span class="nt">&lt;(/?)([^<&\s/]+)(\s*&gt;|)</span>', link, highlighted)
        block.replace_self(nodes.raw('', f'<div class="highlight-xml notranslate">{highlighted}</div>\n',
                                     format='html'))


def resolve_usages(app, doctree, docname):
    usages = getattr(app.env, 'xml_binding_usages', [])
    examples = getattr(app.env, 'xml_binding_examples', [])
    for node in list(doctree.findall(usages_placeholder)):
        result = []
        found = sorted({(d, i, t) for b, d, i, t in usages if b == node['xml_binding']})
        if found:
            paragraph = nodes.paragraph()
            paragraph += nodes.Text('Explained in: ')
            for k, (doc, target_id, section_title) in enumerate(found):
                if k:
                    paragraph += nodes.Text(', ')
                # link text: the title of the page; the link leads to the element's section on it
                page_title = app.env.titles[doc].astext() if doc in app.env.titles else section_title
                uri = app.builder.get_relative_uri(docname, doc) + '#' + target_id
                paragraph += nodes.reference('', page_title, internal=True, refuri=uri, reftitle=section_title)
            result.append(paragraph)

        # XML examples elsewhere in the documentation that use the element, grouped by page
        explained = {u[0] for u in usages}
        pages = {}
        for b, doc, section_id, section_title, page_element in examples:
            # skip reference pages of elements explained elsewhere: their examples are listed there
            if b == node['xml_binding'] and doc != docname and page_element not in explained:
                pages.setdefault(doc, {})[section_id] = section_title
        if pages:
            paragraph = nodes.paragraph()
            paragraph += nodes.Text('Appears in the documentation: ')
            title = lambda d: app.env.titles[d].astext() if d in app.env.titles else d
            for k, doc in enumerate(sorted(pages, key=title)):
                if k:
                    paragraph += nodes.Text('; ')
                sections = sorted(pages[doc].items(), key=lambda s: s[1])
                uri = app.builder.get_relative_uri(docname, doc)
                if len(sections) == 1 and sections[0][1] == title(doc):
                    paragraph += nodes.reference('', title(doc), internal=True, refuri=uri)
                    continue
                paragraph += nodes.Text(title(doc) + ' (')
                for j, (section_id, section_title) in enumerate(sections):
                    if j:
                        paragraph += nodes.Text(', ')
                    paragraph += nodes.reference('', section_title, internal=True,
                                                 refuri=uri + ('#' + section_id if section_id else ''))
                paragraph += nodes.Text(')')
            result.append(paragraph)
        node.replace_self(result)


def purge(app, env, docname):
    for attr in ('xml_binding_usages', 'xml_binding_examples'):
        if hasattr(env, attr):
            setattr(env, attr, [u for u in getattr(env, attr) if u[1] != docname])


def merge(app, env, docnames, other):
    for attr in ('xml_binding_usages', 'xml_binding_examples'):
        if not hasattr(env, attr):
            setattr(env, attr, [])
        getattr(env, attr).extend(u for u in getattr(other, attr, []) if u[1] in docnames)


def setup(app):
    app.add_config_value('xml_bindings_json', '', 'env')
    app.add_config_value('xml_bindings_scripts', '', 'env')
    app.add_config_value('xml_bindings_base_url', '', 'env')
    app.add_node(usages_placeholder)
    app.add_directive('xml-element', XmlElementDirective)
    app.add_directive('xml-element-usages', XmlElementUsagesDirective)
    app.connect('doctree-read', collect_usages)
    app.connect('doctree-resolved', resolve_usages)
    app.connect('doctree-resolved', link_examples)
    app.connect('env-purge-doc', purge)
    app.connect('env-merge-info', merge)
    return {'version': '1.0', 'parallel_read_safe': True, 'parallel_write_safe': True}
