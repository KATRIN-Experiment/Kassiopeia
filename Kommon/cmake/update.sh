#!/bin/sh

CMAKE_VERSION="v3.13.0"
GITHUB_URL="https://raw.githubusercontent.com/Kitware/CMake"

for name in *.cmake; do
    cp -af "${name}" "${name}.update.bak"
    wget -q "${GITHUB_URL}/${CMAKE_VERSION}/Modules/${name}" -O "${name}.update" && mv "${name}.update" "${name}" && echo "Updated ${name}"
done
