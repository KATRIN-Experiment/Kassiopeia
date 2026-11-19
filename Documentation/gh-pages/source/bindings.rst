.. _bindings-label:

Bindings Documentation
**********************

This section provides documentation for the **XML bindings**. Many Kasper
modules can parse XML files during intialization; the bindings classes define
the available XML elements, their attributes and their relation to C++ classes.

.. contents:: On this page
    :local:
    :depth: 1

Overview
========

The pages below show the allowed *elements* and *attributes* that can be used
in Kasper XML files, according to the syntax:

.. code-block:: xml

    <element attribute_1="value_1" attribute_2="value_2">
        <child_element attribute_3="value_3" />
    </element>

Elements can be nested according to what is shown here, and each element can
have a number of attributes. Note that in some cases, certain child elements
and attributes are required; in other cases, they are mutually exclusive. This
is only mentioned where the element descriptions document it.

The descriptions are written as ``//!`` comments next to the bindings in the
C++ sources (see
`Scripts/bindings/bindings_model.py <https://github.com/KATRIN-Experiment/Kassiopeia/blob/main/Scripts/bindings/bindings_model.py>`_
for the format), so additions and corrections are welcome there.

This documentation is generated from the C++ sources of the bindings by the scripts in
`Scripts/bindings <https://github.com/KATRIN-Experiment/Kassiopeia/tree/main/Scripts/bindings>`_.
The underlying data is also available for download:

* `bindings_full.json <bindings/bindings_full.json>`_: all elements, child
  elements and attributes in machine-readable form, e.g. for editor tooling.
* `bindings_full.xml <bindings/bindings_full.xml>`_: an XML file listing all
  elements and attributes, with placeholder values indicating their data type.
  It is not meant to be run, but serves as an almost-complete syntax example.

Element reference
=================

The pages below document every XML element: its description, attributes and child elements, the elements it can be
used in, and examples. For many elements, the child elements and attributes cannot be freely combined (e.g. some are
mutually exclusive); this is described in the element descriptions where documented.

.. toctree::
    :maxdepth: 1

    bindings_reference/geometry
    bindings_reference/kemfield
    bindings_reference/kassiopeia
    bindings_reference/other
