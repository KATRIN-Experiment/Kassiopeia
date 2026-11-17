Bindings documentation
======================

The documentation of the **XML bindings** (the XML elements and attributes
available in configuration files, and the C++ classes they map to) is part of
the Kassiopeia website:

**https://katrin-experiment.github.io/Kassiopeia/bindings.html**

It is generated from the C++ sources whenever the website is built, so it
always matches the `main` branch. Besides the pages shown there,
the website also provides the underlying data for download:

* [bindings_full.json](https://katrin-experiment.github.io/Kassiopeia/bindings/bindings_full.json):
  all elements, child elements and attributes in machine-readable form, e.g.
  for editor tooling.
* [bindings_full.xml](https://katrin-experiment.github.io/Kassiopeia/bindings/bindings_full.xml):
  an XML file listing all elements and attributes, with placeholder values
  indicating their data type. It is not meant to be run, but serves as an
  almost-complete syntax example.

Generating the files locally
----------------------------

The files are produced in two steps by the Python scripts in
[Scripts/bindings](../../Scripts/bindings):

1. [extract.py](../../Scripts/bindings/extract.py) reads the C++ builder
   sources and writes all XML elements, their child elements and attributes
   to a JSON file. The JSON format is documented in
   [bindings_model.py](../../Scripts/bindings/bindings_model.py).
2. Separate renderers read that JSON and produce the output files:
   [to_rst.py](../../Scripts/bindings/to_rst.py) (reStructuredText reference pages) and
   [to_xml.py](../../Scripts/bindings/to_xml.py) (XML example).

[gen-bindings-docs.sh](../../Scripts/gen-bindings-docs.sh) runs all steps
(requires Python 3). Run it from the repository root; the
output is written to `Documentation/gh-pages/build/bindings` unless
`OUTPUT_DIR` is set:

    Scripts/gen-bindings-docs.sh

When building the website (see [Documentation/gh-pages](../gh-pages)), this
happens automatically via `make bindings`.
