#!/bin/bash

# List all CMake package/module depenencies and create dot-graph.
# This will consider all `kasper_find_module` commands.
# The output file is named `module-graph.{dot,png}` by default.
#
# Usage:  `./module-deps-graph.sh [root]`
#   where root must contain at least one `CMakeLists.txt` file,
#   but it can be a Kasper subdirectory as well.

BASE=${1:-${PWD}}

if [ ! -f "${BASE}/CMakeLists.txt" -o ! -d "${PWD}/.git" ]; then
    printf "Script needs to be run in a source directory! Exiting...\n"
    exit 1
fi

NAME="${BASE}/module-deps-graph"

cmake_files=$(git ls-tree --full-tree -r --name-only HEAD -- "${BASE}" \
        | egrep 'CMakeLists\.txt$' \
        | egrep -v '([Ee]xamples)\/' \
    )

# List of module names, based on CMakeLists.txt project definitions.
# This trick makes all actual CMake projects to be drawn as boxes.
# Unreferenced projects are hidden from the graph.
project_names=$(
    echo "${cmake_files}" \
        | xargs grep -H 'project' \
        | sed -nr \
            -e 's/^([a-zA-Z0-9_\/]+)\/CMakeLists\.txt:\s*project\s*\(\s*([a-zA-Z_-]+)\s*.*\)/"\2"/p' \
        | sort | uniq \
        | sed 's/\//\/\\n/g' \
    )

# List of module names, based on CMakeLists.txt location.
# If the module name is a CMake project, it will be drawn as a box (see above).
module_names=$(
    echo "${cmake_files}" \
        | xargs grep -H 'kasper_find_module' \
        | sed -nr \
            -e 's/^([a-zA-Z0-9_\/]+)\/CMakeLists\.txt:\s*kasper_find_module\s*\(\s*([a-zA-Z_-]+)\s*\)/"\1"/p' \
        | sort | uniq \
        | sed 's/\//\/\\n/g' \
    )

# List of module targets, based on kasper_find_module() expressions.
# If the module target is a CMake project, it will be drawn as a box (see above).
# All module targets are drawn in red color.
module_targets=$(
    echo "${cmake_files}" \
        | xargs grep -H 'kasper_find_module' \
        | sed -nr \
            -e 's/^([a-zA-Z0-9_\/]+)\/CMakeLists\.txt:\s*kasper_find_module\s*\(\s*([a-zA-Z_-]+)\s*\)/"\2"/p' \
        | sort | uniq \
        | sed 's/\//\/\\n/g' \
    )

# Graph of CMake files to Kasper modules; combines module names and module targets.
module_deps=$( \
    echo "${cmake_files}" \
        | xargs grep -H 'kasper_find_module' \
        | sed -nr \
            -e 's/^([a-zA-Z0-9_\/]+)\/CMakeLists\.txt:\s*kasper_find_module\s*\(\s*([a-zA-Z_-]+)\s*\)/"\1"->"\2"/p' \
        | sort | uniq \
        | sed 's/\//\/\\n/g' \
    )

if [ -z "${module_deps}" ]; then
    echo "Found NO dependencies for ${BASE}"
    exit -1
fi

echo > $NAME.dot
echo "digraph \"${NAME}\" {" >> $NAME.dot
echo "  splines=true; overlap=false; rankdir=LR; ordering=out;" >> $NAME.dot
echo "  node[shape=box];" >> $NAME.dot
echo "# package_names" >> $NAME.dot
for node in ${project_names}; do
    echo "  ${node} [style=invis,shape=box];" >> $NAME.dot
done
echo "# module_names" >> $NAME.dot
for node in ${module_names}; do
    echo "  ${node} [style=filled,color=lightgrey];" >> $NAME.dot
done
for node in ${project_names}; do
    echo "  ${node} [color=lightgreen];" >> $NAME.dot
done
echo "# module_targets" >> $NAME.dot
echo "subgraph cluster_targets {" >> $NAME.dot
echo "  style=filled; color=none;" >> $NAME.dot
for node in ${module_targets}; do
    echo "  ${node} [style=filled,color=lightblue];" >> $NAME.dot
done
echo "}" >> $NAME.dot
echo "# module_deps" >> $NAME.dot
for edge in ${module_deps}; do
    echo "  ${edge} [style=solid];" >> $NAME.dot
done
echo "}" >> $NAME.dot

dot -Tpng -o$NAME.png $NAME.dot

echo "Dependency scan of ${BASE} completed:"
echo "  $(echo "${project_names}" | wc -l) CMake projects"
echo "  $(echo "${module_targets}" | wc -l) Kasper modules"
echo "  $(echo "${module_deps}" | wc -l) internal dependencies"
echo "Output created in: ${NAME}.{dot,png}"
