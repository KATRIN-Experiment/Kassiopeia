#!/bin/bash

# This command runs clang-format on the given directories. The formatting style
# is configured in the `.clang-format` file located in the top-level directory
# or in one of its sub-directories. All files are edited in-place, and a backup
# file with edning `.clang-format.bak` is created before any changes are made.

if [ ! -f "${PWD}/CMakeLists.txt" -o ! -d "${PWD}/.git" ]; then
    printf "Script needs to be run in a source directory! Exiting...\n"
    exit 1
fi

FORMAT=$(which clang-format)

if [ ! -e ${FORMAT} ]; then
    echo "Script ${FORMAT} not found - please install clang-format or update path in this script"
    exit 1
fi

log_file="${PWD}/clang-format.log"
rm -f "${log_file}"

for dir in $@; do
    echo "Processing: ${dir}"
    SOURCES=$(git ls-tree --full-tree -r --name-only HEAD -- "${dir}" \
            | egrep '\.([chi]pp|[chi]xx|[chi]{2}|h|[CHI])$' \
            | egrep -v '(std)?soap\w*\.(hpp|cpp|h)$' \
            | egrep -v '(md5|miniz)\.(cc|hh)' \
            | egrep -v '(opencl)\/(1.1|1.2)\/(Open)?CL\/\w*\.(hpp|cpp|h)$' \
            | egrep -v 'gtest(-all|_main)?\.(cc|h)$')

    for file in ${SOURCES}; do
        echo "  ${file}"
        cp -af "${file}" "${file}.clang-format.bak" \
            && $FORMAT -style=file -i "${file}" \
            | tee -a "${log_file}"
    done
done
