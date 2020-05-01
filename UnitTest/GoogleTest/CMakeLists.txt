########################################################################
# CMake build script for Google Test.

option(gtest_disable_pthreads
    "Disable uses of pthreads in gtest."
    OFF)

option(gtest_force_shared_crt
    "Use shared (DLL) run-time lib even when Google Test is built as static lib."
    OFF)

########################################################################
#
# Project-wide settings

# Name of the project.
#
# CMake files in this project can refer to the root source directory
# as ${gtest_SOURCE_DIR} and to the root binary directory as
# ${gtest_BINARY_DIR}.
# Language "C" is required for find_package(Threads).
# project(gtest CXX C)
# cmake_minimum_required(VERSION 2.6.2)

# Define helper functions and macros used by Google Test.
include(cmake/internal_utils.cmake)

config_compiler_and_linker()  # Defined in internal_utils.cmake.

# Where Google Test's .h files can be found.
set( gtest_SOURCE_DIR ${CMAKE_CURRENT_SOURCE_DIR} )

include_directories(
  ${gtest_SOURCE_DIR}/include
  ${gtest_SOURCE_DIR}
)

# Where Google Test's libraries can be found.
link_directories(${gtest_BINARY_DIR}/src)

########################################################################
#
# Defines the gtest & gtest_main libraries.  User tests should link
# with one of them.

# Google Test libraries.  We build them using more strict warnings than what
# are used for other targets, to ensure that gtest can be compiled by a user
# aggressive about warnings.
cxx_shared_library(gtest "-w" src/gtest-all.cc)
cxx_shared_library(gtest_main "-w" src/gtest_main.cc)
target_link_libraries( gtest_main gtest )
kasper_install_libraries( gtest_main gtest )
