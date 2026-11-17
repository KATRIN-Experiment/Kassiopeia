#!/bin/bash
#
# Extracts the XML bindings from the C++ sources into JSON (bindings/extract.py),
# then renders the documentation files from that JSON (bindings/to_*.py).
#
# Run from the repository root. The documentation build ("make bindings" in
# Documentation/gh-pages) calls this script and copies the results into the
# Sphinx sources.

set -e

BINDINGS_DIR="$(dirname $0)/bindings"
SOURCE_DIRS="Kassiopeia KEMField KGeoBag Kommon"
OUTPUT_DIR="${OUTPUT_DIR:-Documentation/gh-pages/build/bindings}"

GH_BASE="https://github.com/KATRIN-Experiment/Kassiopeia/blob/main/"

mkdir -p ${OUTPUT_DIR}

## Extract bindings
echo "Extracting global bindings"
JSON="${OUTPUT_DIR}/bindings_full.json"
python3 ${BINDINGS_DIR}/extract.py ${SOURCE_DIRS} -o "${JSON}"

## Generate reStructuredText reference pages and XML example
echo "Generating global bindings files"
rm -rf "${OUTPUT_DIR}/reference"
python3 ${BINDINGS_DIR}/to_rst.py "${JSON}" --base-url=${GH_BASE} -o "${OUTPUT_DIR}/reference"
python3 ${BINDINGS_DIR}/to_xml.py "${JSON}" --root='KRoot' > "${OUTPUT_DIR}/bindings_full.xml"
