#!/bin/bash

# Build and install the default configuration of Kassiopeia
# 
# Usage:
# ./setup.sh
# 
# Respects the following optional environment variables, with defaults:
# KASSIOPEIA_BUILD_TYPE=RelWithDebInfo
# KASSIOPEIA_INSTALL_PREFIX=install
# KASSIOPEIA_BUILD_PREFIX=build
# KASSIOPEIA_MAKECMD=make
# KASSIOPEIA_CUSTOM_CMAKE_ARGS=""
# KASSIOPEIA_GIT_BRANCH=""
# KASSIOPEIA_GIT_COMMIT=""
# KASSIOPEIA_CPUS=$(($(nproc)-1))
# 
# Full command line examples:
# KASSIOPEIA_BUILD_TYPE="Release" ./setup.sh
# KASSIOPEIA_BUILD_TYPE="RelWithDebInfo" \
#     KASSIOPEIA_INSTALL_PREFIX="/path/to/install" \
#     KASSIOPEIA_BUILD_PREFIX="/path/to/build" \
#     ./setup.sh
# KASSIOPEIA_BUILD_TYPE="RelWithDebInfo" \
#     KASSIOPEIA_INSTALL_PREFIX="/path/to/install" \
#     KASSIOPEIA_BUILD_PREFIX="/path/to/build" \
#     KASSIOPEIA_MAKECMD="ninja" \
#     KASSIOPEIA_CUSTOM_CMAKE_ARGS="-GNinja" \
#     KASSIOPEIA_GIT_BRANCH="develop" \
#     KASSIOPEIA_GIT_COMMIT="6c9dbbf3e" \
#     KASSIOPEIA_CPUS=4 \
#     KASSIOPEIA_USE_OPENCL=ON \
#     ./setup.sh

if [ $# -ne 0 ] ; then
    echo "Since 2023-07, setup.sh uses environment variables instead of arguments. Documentation can be found at the beginning of setup.sh."
    exit -1
fi

KASSIOPEIA_BUILD_TYPE=${KASSIOPEIA_BUILD_TYPE:-"RelWithDebInfo"}
KASSIOPEIA_INSTALL_PREFIX=$(realpath -s ${KASSIOPEIA_INSTALL_PREFIX:-"install"})
KASSIOPEIA_BUILD_PREFIX=$(realpath -s ${KASSIOPEIA_BUILD_PREFIX:-"build"})

KASSIOPEIA_MAKECMD=${KASSIOPEIA_MAKECMD:-"make"}
KASSIOPEIA_CUSTOM_CMAKE_ARGS=${KASSIOPEIA_CUSTOM_CMAKE_ARGS:-""}

KASSIOPEIA_GIT_BRANCH=${KASSIOPEIA_GIT_BRANCH:-""}
KASSIOPEIA_GIT_COMMIT=${KASSIOPEIA_GIT_COMMIT:-""}

KASSIOPEIA_CPUS=${KASSIOPEIA_CPUS:-"$(($(nproc)-1))"}

KASSIOPEIA_USE_OPENCL=${KASSIOPEIA_USE_OPENCL:-"OFF"}

echo "Building KASPER $KASSIOPEIA_BUILD_TYPE for '$KASSIOPEIA_INSTALL_PREFIX' in '$KASSIOPEIA_BUILD_PREFIX'"

# Cause script to exit after a command failed
set -e

GIT_ARGS="-DKASPER_GIT_INFO_USERDEFINED=OFF"

if [[ ! -z $KASSIOPEIA_GIT_BRANCH ]]
then
    GIT_ARGS="-DKASPER_GIT_INFO_USERDEFINED=ON \
              -DKASPER_GIT_BRANCH=$KASSIOPEIA_GIT_BRANCH \
              -DKASPER_GIT_COMMIT=$KASSIOPEIA_GIT_COMMIT"

    echo "User-defined git branch $KASSIOPEIA_GIT_BRANCH and commit $KASSIOPEIA_GIT_COMMIT"
fi

# Get script location
# https://stackoverflow.com/a/246128
DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" >/dev/null 2>&1 && pwd )"

mkdir -p $KASSIOPEIA_BUILD_PREFIX
pushd $KASSIOPEIA_BUILD_PREFIX
cmake -DCMAKE_BUILD_TYPE=$KASSIOPEIA_BUILD_TYPE \
        -DCMAKE_INSTALL_PREFIX=$KASSIOPEIA_INSTALL_PREFIX \
        -DKASPER_USE_ROOT=ON \
        -DKASPER_USE_VTK=ON \
        -DKASPER_USE_GSL=ON \
        -DKEMField_USE_OPENCL=${KASSIOPEIA_USE_OPENCL} \
        -DBUILD_KASSIOPEIA=ON \
        -DBUILD_KEMFIELD=ON \
        -DBUILD_KGEOBAG=ON \
        -DBUILD_KOMMON=ON \
        -DBUILD_UNIT_TESTS=ON \
        $KASSIOPEIA_CUSTOM_CMAKE_ARGS \
        $GIT_ARGS \
    $DIR
if [[ ! -z $KASSIOPEIA_CPUS ]]
then
    $KASSIOPEIA_MAKECMD -j $KASSIOPEIA_CPUS
fi
$KASSIOPEIA_MAKECMD install
popd
