#!/bin/bash

# This command runs clang-check on the given directories. The enabled filters
# are configured in this script.
# Please check the build and unit tests before committing any changes!

BUILD_PATH=""
SOURCE_PATH=""
if [ -f "${PWD}/../CMakeLists.txt" -a -d "${PWD}/../.git" ]; then
    printf "Script running from build directory.\n"
    BUILD_PATH="${PWD}"
    SOURCE_PATH="${PWD}/.."
elif [ -f "${PWD}/CMakeLists.txt" -a -d "${PWD}/.git" ]; then
    printf "Script running from source directory.\n"
    SOURCE_PATH="${PWD}"
else
    printf "Script needs to be run in a source or build directory! Exiting...\n"
    exit 1
fi

CHECK=$(which clang-check)

if [ ! -e ${CHECK} ]; then
    echo "Script ${CHECK} not found - please install clang-tidy or update path in this script"
    exit 1
fi

log_file="${PWD}/clang-check.log"
rm -f "${log_file}"

compile_commands="${BUILD_PATH}/compile_commands.json"
if [ -f "${compile_commands}" ]; then
    echo "Found a compilation database '${compile_commands}' to use with clang-tidy"
    TIDY+=" -p ${compile_commands}"
fi

EXTRA_HEADERS=""

complete_file="${SOURCE_PATH}/.clang_complete"
if [ -f "${complete_file}" ]; then
    echo "Found a completion file '${complete_file}' to provide extra headers"
    EXTRA_HEADERS+=$(cat "${complete_file}")
fi

if [ -e $(which root-config) ]; then
    EXTRA_HEADERS+=" -I$(root-config --incdir)"
fi
if [ -d /usr/include/vtk ]; then
    EXTRA_HEADERS+=" -I/usr/include/vtk -DVTK6"
fi

pushd "${SOURCE_PATH}"
for dir in $@; do
    echo "Processing: ${dir}"
    SOURCES=$(git ls-tree --full-tree -r --name-only HEAD -- "${dir}" \
            | egrep '\.([chi]pp|[chi]xx|[chi]{2}|h|[CHI])$' \
            | egrep -v '(std)?soap[[:alnum:]]*\.(cpp|h)' \
            | egrep -v '(opencl)\/(1.1|1.2)\/(Open)?CL\/\w*\.(hpp|cpp|h)$' \
            | egrep -v 'gtest(-all|_main)?\.(cc|h)$')

    if [ -z "${SOURCES}" ]; then
        echo "(... no source files)"
        continue
    fi

    for file in ${SOURCES}; do
        echo "  ${file}"
        cp -af "${file}" "${file}.clang-check.bak" \
            && $CHECK -analyze "${file}" \
            -- -x c++ -std=c++14 ${EXTRA_HEADERS} \
            | tee -a "${log_file}"
    done
done
popd
