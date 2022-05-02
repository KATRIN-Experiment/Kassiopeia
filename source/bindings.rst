.. _bindings-label:

Bindings Documentation
**********************

This section provides documentation for the **XML bindings**. Many Kasper
modules can parse XML files during intialization; the bindings classes define
the available XML elements, their attributes and their relation to C++ classes.

Overview
--------

The files below show the allowed *elements* and *attributes* that can be used
in Kasper XML files; according to the syntax:

.. code-block:: xml

    <element attribute_1="value_1" attribute_2="value_2">
        <child_element attribute_3="value_3" />
    </element>

Elements can be nested according to what is shown here, and each element can
have a number of attributes. Note that in some cases, certain child elements
and attributes are required; in other cases, they are mutually exclusive. Note
that this information is not represented here in the documentation.

.. image:: _images/bindings_full.svg
    :width: 800px
    :alt: XML bindings graph

XML Bindings
------------

.. include:: ./bindings_full.rst
