#!/bin/sh

# A simple script to pull files from the CMake repository.

CMAKE_VERSION="v3.14.0"
GITHUB_URL="https://raw.githubusercontent.com/Kitware/CMake"

for name in $(find ! -name '*.bak' -a ! -name '*.in' -type f) ; do
    cp -af "${name}" "${name}.bak"
    wget -q "${GITHUB_URL}/${CMAKE_VERSION}/Modules/${name}" -O "${name}.update" && \
        mv "${name}.update" "${name}" && \
        echo "Updated ${name}" && \
        git add ${name}
   rm -f "${name}.update"
done
