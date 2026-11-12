#!/bin/bash
#
# Extracts the XML bindings from the C++ sources into JSON (bindings/extract.py),
# then renders the documentation files from that JSON (bindings/to_*.py).

BINDINGS_DIR="$(dirname $0)/bindings"
SOURCE_DIRS="Kassiopeia KEMField KGeoBag Kommon"
OUTPUT_DIR="${OUTPUT_DIR:-Documentation/Bindings}"

GH_BASE="https://github.com/KATRIN-Experiment/Kassiopeia/tree/main/"

mkdir -p ${OUTPUT_DIR}

## Extract bindings
echo "Extracting global bindings"
JSON="${OUTPUT_DIR}/bindings_full.json"
python3 ${BINDINGS_DIR}/extract.py ${SOURCE_DIRS} -o "${JSON}" || exit 1

## Generate Markdown files
echo "Generating global bindings files"
python3 ${BINDINGS_DIR}/to_markdown.py "${JSON}" --root='KRoot' --base-url=${GH_BASE} --with-sections --with-examples > "${OUTPUT_DIR}/bindings_full_examples.md"
python3 ${BINDINGS_DIR}/to_markdown.py "${JSON}" --root='KRoot' --base-url=${GH_BASE} --with-sections > "${OUTPUT_DIR}/bindings_full.md"
python3 ${BINDINGS_DIR}/to_rst.py "${JSON}" --root='KRoot' --base-url=${GH_BASE} --with-sections > "${OUTPUT_DIR}/bindings_full.rst"
python3 ${BINDINGS_DIR}/to_xml.py "${JSON}" --root='KRoot' > "${OUTPUT_DIR}/bindings_full.xml"

## Generate GraphViz files
echo "Generating global bindings graph"
python3 ${BINDINGS_DIR}/to_graphviz.py "${JSON}" --root='KRoot' > "${OUTPUT_DIR}/bindings_full.dot"
dot -Tpdf "${OUTPUT_DIR}/bindings_full.dot" > "${OUTPUT_DIR}/bindings_full.pdf"
dot -Tsvg "${OUTPUT_DIR}/bindings_full.dot" > "${OUTPUT_DIR}/bindings_full.svg"

# individual file per module
MODULE_JSON="$(mktemp)"
trap 'rm -f "${MODULE_JSON}"' EXIT
for dirname in ${SOURCE_DIRS}; do
    echo "Generating ${dirname} bindings graph"
    python3 ${BINDINGS_DIR}/extract.py ${dirname} KSC/${dirname} -o "${MODULE_JSON}" || continue
    python3 ${BINDINGS_DIR}/to_graphviz.py "${MODULE_JSON}" --with-attributes > "${OUTPUT_DIR}/bindings_${dirname}.dot" || continue
    dot -Tpdf "${OUTPUT_DIR}/bindings_${dirname}.dot" > "${OUTPUT_DIR}/bindings_${dirname}.pdf"
    dot -Tsvg "${OUTPUT_DIR}/bindings_${dirname}.dot" > "${OUTPUT_DIR}/bindings_${dirname}.svg"
done
