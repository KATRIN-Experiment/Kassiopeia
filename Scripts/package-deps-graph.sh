#!/bin/bash

# List all CMake package/module depenencies and create dot-graph.
# This will consider all `find_package` commands.
# The output file is named `package-graph.{dot,png}` by default.
#
# Usage:  `./package-deps-graph.sh [root]`
#   where root must contain at least one `CMakeLists.txt` file,
#   but it can be a Kasper subdirectory as well.

BASE=${1:-${PWD}}

if [ ! -f "${BASE}/CMakeLists.txt" -o ! -d "${PWD}/.git" ]; then
    printf "Script needs to be run in a source directory! Exiting...\n"
    exit 1
fi

NAME="${BASE}/package-deps-graph"

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
            -e 's/^([a-zA-Z0-9_\/]+)\/CMakeLists\.txt:\s*project\s*\(\s*([a-zA-Z0-9_-]+)\s*.*\)/"\2"/p' \
        | sort | uniq \
        | sed 's/\//\/\\n/g' \
    )

# List of package names, based on CMakeLists.txt location.
# If the package name is a CMake project, it will be drawn as a box (see above).
package_names=$(
    echo "${cmake_files}" \
        | xargs grep -H 'find_package' \
        | sed -nr \
            -e 's/^([a-zA-Z0-9_\/]+)\/CMakeLists\.txt:\s*find_package\s*\(\s*([a-zA-Z0-9_-]+)\s*.*\)/"\1"/p' \
        | sort | uniq \
        | sed 's/\//\/\\n/g' \
    )

# List of package targets, based on kasper_find_module() expressions.
# If the package target is a CMake project, it will be drawn as a box (see above).
# All package targets are drawn in red color.
package_targets=$(
    echo "${cmake_files}" \
        | xargs grep -H 'find_package' \
        | sed -nr \
            -e 's/^([a-zA-Z0-9_\/]+)\/CMakeLists\.txt:\s*find_package\s*\(\s*([a-zA-Z0-9_-]+)\s*.*\)/"\2"/p' \
        | sort | uniq \
        | sed 's/\//\/\\n/g' \
    )

# Graph of CMake files to packages; combines package names and package targets.
package_deps_required=$( \
    echo "${cmake_files}" \
        | xargs grep -H 'find_package' \
        | egrep "\s+REQUIRED\s?" \
        | sed -nr \
            -e 's/([a-zA-Z0-9_\/]+)\/CMakeLists\.txt:\s*find_package\s*\(\s*([a-zA-Z0-9_-]+)\s.*\)/"\1"->"\2"/p' \
        | sort | uniq \
        | sed 's/\//\/\\n/g' \
    )

package_deps_optional=$( \
    echo "${cmake_files}" \
        | xargs grep -H 'find_package' \
        | egrep -v "\s+REQUIRED\s?" \
        | sed -nr \
            -e 's/^([a-zA-Z0-9_\/]+)\/CMakeLists\.txt:\s*find_package\s*\(\s*([a-zA-Z0-9_-]+)\s?.*\)/"\1"->"\2"/p' \
        | sort | uniq \
        | sed 's/\//\/\\n/g' \
    )

if [ -z "${package_deps_required}" -a -z "${package_deps_optional}" ]; then
    echo "Found NO dependencies for ${BASE}"
    exit -1
fi

echo > $NAME.dot
echo "digraph \"${NAME}\" {" >> $NAME.dot
echo "  root=\"Kommon\"; model=subset; epsilon=0.0001; splines=true; overlap=false; rankdir=LR; ordering=out;" >> $NAME.dot
echo "  node[shape=box];" >> $NAME.dot
echo "# project_names" >> $NAME.dot
for node in ${project_names}; do
    echo "  ${node} [style=invis,shape=box];" >> $NAME.dot
done
echo "# package_names" >> $NAME.dot
for node in ${package_names}; do
    echo "  ${node} [style=filled,color=lightgrey];" >> $NAME.dot
done
for node in ${project_names}; do
    echo "  ${node} [color=lightgreen];" >> $NAME.dot
done
echo "# package_targets" >> $NAME.dot
echo "subgraph cluster_targets {" >> $NAME.dot
echo "  style=filled; color=none;" >> $NAME.dot
for node in ${package_targets}; do
    color="lightpink"
    for proj in ${project_names}; do
        # color internal packages differently
        if [ "$node" == "$proj" ]; then
            color="lightblue"
        fi
    done
    echo "  ${node} [style=filled,color=${color}];" >> $NAME.dot
done
echo "}" >> $NAME.dot
echo "# package_deps_required" >> $NAME.dot
for edge in ${package_deps_required}; do
    echo "  ${edge} [style=solid,color=black];" >> $NAME.dot
done
echo "# package_deps_optional" >> $NAME.dot
for edge in ${package_deps_optional}; do
    echo "  ${edge} [style=dashed];" >> $NAME.dot
done
echo "}" >> $NAME.dot

# Make list of external package dependencies that can be sorted later
for edge in ${package_deps_required}; do
    name=$(echo -E ${edge} | sed 's/"\(.*\)"->"\(.*\)"/\2/g')
    for proj in ${project_names}; do
        if [ "\"${name}\"" == "${proj}" ]; then
            name=""
        fi
    done
    package_names_required+="${name}\n"
done
for edge in ${package_deps_optional}; do
    name=$(echo -E ${edge} | sed 's/"\(.*\)"->"\(.*\)"/\2/g')
    for proj in ${project_names}; do
        if [ "\"${name}\"" == "${proj}" ]; then
            name=""
        fi
    done
    package_names_optional+="${name}\n"
done

neato -Tpng -o$NAME.png $NAME.dot

echo "Dependency scan of ${BASE} completed:"
echo "  $(echo "${project_names}" | wc -l) CMake projects"
echo "  $(echo "${package_targets}" | wc -l) referenced packages"
echo "  $(echo "${package_deps_required}" | wc -l) required + $(echo "${package_deps_optional}" | wc -l) optional dependencies:"
echo "    Required: $(echo -e ${package_names_required} | sort -u | tr '\n' ' ')"
echo "    Optional: $(echo -e ${package_names_optional} | sort -u | tr '\n' ' ')"
echo "Output created in: ${NAME}.{dot,png}"
