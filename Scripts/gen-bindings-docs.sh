#!/bin/bash

SCRIPT_NAME="$(dirname $0)/gen-bindings-docs.py"
SOURCE_DIRS="Kassiopeia KEMField KGeoBag Kommon"
OUTPUT_DIR="Documentation/Bindings"

mkdir -p ${OUTPUT_DIR}

## Generate Markdown files
echo "Generating global bindings files"
python3 ${SCRIPT_NAME} ${SOURCE_DIRS} --root='KRoot' --with-sections --with-examples --md > "${OUTPUT_DIR}/bindings_full.md"
python3 ${SCRIPT_NAME} ${SOURCE_DIRS} --root='KRoot' --with-sections --rst > "${OUTPUT_DIR}/bindings_full.rst"
python3 ${SCRIPT_NAME} ${SOURCE_DIRS} --root='KRoot' --xml > "${OUTPUT_DIR}/bindings_full.xml"

## Generate GraphViz files
echo "Generating global bindings graph"
python3 ${SCRIPT_NAME} ${SOURCE_DIRS} --root='KRoot' --gv > "${OUTPUT_DIR}/bindings_full.dot"
dot -Tpdf "${OUTPUT_DIR}/bindings_full.dot" > "${OUTPUT_DIR}/bindings_full.pdf"
dot -Tsvg "${OUTPUT_DIR}/bindings_full.dot" > "${OUTPUT_DIR}/bindings_full.svg"

# individual file per module
for dirname in ${SOURCE_DIRS}; do
    echo "Generating ${dirname} bindings graph"
    python3 ${SCRIPT_NAME} ${dirname} KSC/${dirname} --with-attributes --gv > "${OUTPUT_DIR}/bindings_${dirname}.dot" || continue
    dot -Tpdf "${OUTPUT_DIR}/bindings_${dirname}.dot" > "${OUTPUT_DIR}/bindings_${dirname}.pdf"
    dot -Tsvg "${OUTPUT_DIR}/bindings_${dirname}.dot" > "${OUTPUT_DIR}/bindings_${dirname}.svg"
done
