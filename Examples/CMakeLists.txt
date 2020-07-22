cmake_minimum_required(VERSION 3.0)

# Name your project
project(Kassiopeia-Examples)

message("CMake prefix path: ${CMAKE_PREFIX_PATH}")

find_package(Boost REQUIRED COMPONENTS filesystem)
find_package(VTK)
if(VTK_FOUND)
    include(${VTK_USE_FILE})
endif(VTK_FOUND)

# Don't forget to set CMAKE_PREFIX_PATH to your Kasper installation folder
# before executing cmake. That way, CMake will be able to find the KaLi installation.
find_package(Kassiopeia REQUIRED)
find_package(KEMField REQUIRED)
find_package(KGeoBag REQUIRED)

# Set the default install prefix (you don't need to remember this)
if(CMAKE_INSTALL_PREFIX_INITIALIZED_TO_DEFAULT)
    get_filename_component(BUILD_PARENT_DIR ${CMAKE_BINARY_DIR} PATH)
    set(CMAKE_INSTALL_PREFIX "${BUILD_PARENT_DIR}/install" CACHE PATH "Install path prefix, prepended onto install directories." FORCE)
endif()

set( CMAKE_CXX_STANDARD 14 )
set( CMAKE_CXX_STANDARD_REQUIRED ON )
set( CMAKE_CXX_EXTENSIONS OFF )

# Define your include directories
include_directories (
    ${Kassiopeia_INCLUDE_DIRS}
    ${KEMField_INCLUDE_DIRS}
    ${KGeoBag_INCLUDE_DIRS}
)

SET(CMAKE_INSTALL_RPATH_USE_LINK_PATH TRUE)

#----------------------------------------
# Build libraries and ROOT dictionaries
#----------------------------------------

# not needed for this example

#----------------------------------------
# Build the executable
#----------------------------------------

set( EXECUTABLES
        ks-particles-example
)

foreach(EXECUTABLE ${EXECUTABLES})
    add_executable(${EXECUTABLE} ${EXECUTABLE}.cxx)
    target_link_libraries (${EXECUTABLE} ${Kassiopeia_LIBRARIES})
endforeach()

# Define a cmake install target
install(TARGETS ${EXECUTABLES} DESTINATION ${CMAKE_INSTALL_PREFIX})
