#!/bin/bash

# List all CMake package/module depenencies and create dot-graph.
# This will consider all `kasper_find_module` commands.
# The output file is named `module-graph.{dot,png}` by default.
#
# Usage:  `./module-deps-graph.sh [root]`
#   where root must contain at least one `CMakeLists.txt` file,
#   but it can be a Kasper subdirectory as well.

BASE=${1:-${PWD}}

if [ ! -f "${BASE}/../CMakeLists.txt" -o ! -d "${PWD}/../.git" ]; then
    printf "Script needs to be run in a build directory! Exiting...\n"
    exit 1
fi

NAME="${BASE}/../module-deps-graph"

cmake --graphviz="${NAME}.dot" .. || exit $?

dot -Tpng -o$NAME.png $NAME.dot

echo "Dependency scan of ${BASE} completed:"
echo "Output created in: ${NAME}.{dot,png}"
