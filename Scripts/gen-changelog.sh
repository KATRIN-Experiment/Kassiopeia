#!/bin/bash

# This script uses `git log` to list all changes between two branches. The
# default is to compare `master` to `develop`. The output is in Markdown
# format so it can be added to `CHANGELOG.md`.

if [ ! -f "${PWD}/CMakeLists.txt" -o ! -d "${PWD}/.git" ]; then
    printf "Script needs to be run in a source directory! Exiting...\n"
    exit 1
fi

FROM=${1:-master}
TO=${2:-develop}
shift 2

PATHS=$@
if [ -z "$PATHS" ]; then
    PATHS=$(git ls-tree --full-tree -d --name-only ${TO} | egrep -v '^\.')
fi

FROM_DATE=$(git log ${FROM} --pretty=format:'%S (%cs)' -n 1)
TO_DATE=$(git log ${TO} --pretty=format:'%S (%cs)' -n 1)

GIT_WEB_URL="https://github.com/KATRIN-Experiment/Kassiopeia/commit/"

changes_file="Changelog/changes-from-${FROM}-to-${TO}.md"
rm -f "${changes_file}"

echo "# Kasper Changelog" >> "${changes_file}"
echo >> "${changes_file}"
echo "## Changes from ${FROM_DATE} to ${TO_DATE}" >> "${changes_file}"
for dir in $PATHS; do
    echo "### ${dir}" >> "${changes_file}"
    git log ${FROM}..${TO} --no-merges \
            --pretty=format:"- **%as:** %s [*(view commit)*](${GIT_WEB_URL}/%H)" \
            -- "${dir}" \
            | sort --key=1,10 --reverse \
            >> "${changes_file}"
done

echo "Created file ${changes_file} with $(wc -l "${changes_file}" | cut -d ' ' -f 1) lines."
