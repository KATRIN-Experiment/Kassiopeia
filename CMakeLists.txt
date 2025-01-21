set( CMAKE_MINIMUM_VERSION "3.14.0" )
cmake_minimum_required( VERSION ${CMAKE_MINIMUM_VERSION} )

# project version
set(KASPER_VERSION_MAJOR 4)
set(KASPER_VERSION_MINOR 1)
set(KASPER_VERSION_PATCH 0)
set(KASPER_VERSION "${KASPER_VERSION_MAJOR}.${KASPER_VERSION_MINOR}.${KASPER_VERSION_PATCH}")

#project( Kasper )
project( Kasper VERSION ${KASPER_VERSION}
                DESCRIPTION "The KATRIN Analysis and Simulations Package"
                LANGUAGES CXX C )

set( BUILD_SHARED_LIBS ON )
set( Boost_USE_STATIC_LIBS OFF )
set( Boost_USE_MULTITHREADED ON )
set( Boost_USE_STATIC_RUNTIME OFF )

set( CMAKE_CXX_STANDARD 14 CACHE STRING "The C++ standard whose features are requested to build this target" )
mark_as_advanced( FORCE CMAKE_CXX_STANDARD )
set( CMAKE_CXX_STANDARD_REQUIRED ON )
set( CMAKE_CXX_EXTENSIONS ON )
set( CMAKE_EXPORT_COMPILE_COMMANDS ON )

if(NOT CMAKE_CXX_STANDARD MATCHES "14|17|20")
    message(FATAL_ERROR "Unsupported C++ standard: ${CMAKE_CXX_STANDARD}")
endif()

# Additional compiler version checks, see https://en.cppreference.com/w/cpp/compiler_support
if( CMAKE_SYSTEM_NAME STREQUAL "Linux" )
    if( CMAKE_CXX_COMPILER_ID STREQUAL "GNU" )
        if( CMAKE_CXX_COMPILER_VERSION VERSION_LESS "6.1" )
            message(FATAL_ERROR "Your compiler is unsupported! You need at least GCC 6.1 to build Kasper.")
        endif()
    elseif( CMAKE_CXX_COMPILER_ID STREQUAL "Clang" )
        if( CMAKE_CXX_COMPILER_VERSION VERSION_LESS "3.4" )
            message(FATAL_ERROR "Your compiler is unsupported! You need at least Clang 3.4 to build Kasper.")
        endif()
    endif()
elseif( CMAKE_SYSTEM_NAME STREQUAL "Darwin" )
    if( CMAKE_CXX_COMPILER_ID STREQUAL "Clang" )
        if( CMAKE_CXX_COMPILER_VERSION VERSION_LESS "10.0" )
            message(FATAL_ERROR "Your compiler is unsupported! You need at least Clang 10.0 to build Kasper.")
        endif()
    endif()
else()
    message(WARNING "You're using an unknown compiler that may be unable to build Kasper. ")
endif()

set( CMAKE_MODULE_PATH ${PROJECT_SOURCE_DIR}/Kommon/cmake )
include( KasperDefaults )

set( KASPER_BUILD TRUE )
set( CMAKE_PREFIX_PATH ${CMAKE_BINARY_DIR} )

# use, i.e. don't skip the full RPATH for the build tree
set( CMAKE_SKIP_BUILD_RPATH FALSE )

# when building, don't use the install RPATH already
#( but later on when installing )
set( CMAKE_BUILD_WITH_INSTALL_RPATH FALSE )

# add the automatically determined parts of the RPATH
# which point to directories outside the build tree to the install RPATH
set( CMAKE_INSTALL_RPATH_USE_LINK_PATH TRUE )

# whether to create a position-independent target
include(CheckPIESupported)
check_pie_supported(LANGUAGES CXX)
if( CMAKE_CXX_LINK_PIE_SUPPORTED )
    set( CMAKE_POSITION_INDEPENDENT_CODE TRUE )
endif( CMAKE_CXX_LINK_PIE_SUPPORTED )

if( APPLE )
    #set( CMAKE_INSTALL_RPATH "@executable_path/../${CMAKE_INSTALL_LIBDIR};@loader_path" )
    set( CMAKE_MACOSX_RPATH TRUE )
elseif( UNIX )
    set( CMAKE_INSTALL_RPATH "$ORIGIN/../${CMAKE_INSTALL_LIBDIR}" )
endif( APPLE )


# Flags for all modules, every module should have a flag.

set( BUILD_KOMMON_DESC "Build Kommon" )
option( BUILD_KOMMON ${BUILD_KOMMON_DESC} ON )

set( BUILD_KGEOBAG_DESC "Build KGeoBag [geometry system]" )
option( BUILD_KGEOBAG ${BUILD_KGEOBAG_DESC} ON )

set( BUILD_KEMFIELD_DESC "Build KEMField [electromagnetic simulation system]" )
option( BUILD_KEMFIELD ${BUILD_KEMFIELD_DESC} ON )

set( BUILD_KASSIOPEIA_DESC "Build Kassiopeia [particle tracking simulation system]" )
option( BUILD_KASSIOPEIA ${BUILD_KASSIOPEIA_DESC} ON )

set( BUILD_UNIT_TESTS_DESC "Build Unit Tests [automatic testing of core functionality using GoogleTest]" )
option( BUILD_UNIT_TESTS ${BUILD_UNIT_TESTS_DESC} OFF )


# Set a default build type if none was specified
if( NOT CMAKE_BUILD_TYPE )
    message(STATUS "Setting build type to 'RelWithDebInfo' as none was specified.")
    set( CMAKE_BUILD_TYPE RelWithDebInfo CACHE STRING "" FORCE )
endif()

# We like the compiler to be more pedantic than usual :-P
add_compile_options( -Wall -Wextra -Werror )

# Only link in libraries that we actually need (but not for executables since it makes things slow)
if( NOT APPLE )
    set(_target_type "$<TARGET_PROPERTY:TYPE>")
    add_link_options( $<IF:$<STREQUAL:"${_target_type}","SHARED_LIBRARY">,LINKER:--as-needed,LINKER:--no-as-needed> )
endif()

# Enable some advanced options depending on build type
if (CMAKE_BUILD_TYPE MATCHES Rel)
    option( COMPILER_TUNE_OPTIONS "Enable additional tuning options [compiles for local CPU, use with care!]" OFF )
    mark_as_advanced(COMPILER_TUNE_OPTIONS)
    if( COMPILER_TUNE_OPTIONS )
        # These are advanced options for GCC which might improve performance ... or break everything - use with care!
        add_compile_options( -march=native -mfpmath=sse -funroll-loops )
    endif( COMPILER_TUNE_OPTIONS )
endif()

if( CMAKE_CXX_COMPILER_ID STREQUAL "GNU" OR CMAKE_COMPILER_IS_GNUCXX )
    option( GCC_REDUCE_MEMORY "Significantly reduce GCC memory usage by setting ftrack-macro-expansion=1 (default: 2)" ON )
    mark_as_advanced(GCC_REDUCE_MEMORY)
    if( GCC_REDUCE_MEMORY )
        add_compile_options( -ftrack-macro-expansion=1 )
    endif( GCC_REDUCE_MEMORY )

    option( GCC_ENABLE_PIPES "Use pipes to speed up compiling with GCC by adding the -pipe flag" ON )
    mark_as_advanced(GCC_ENABLE_PIPES)
    if( GCC_ENABLE_PIPES )
        add_compile_options( -pipe )
    endif( GCC_ENABLE_PIPES )

else( CMAKE_CXX_COMPILER_ID STREQUAL "Clang" )
    # Suppress warnings about unknown warning options. See https://bugs.llvm.org/show_bug.cgi?id=24979
    add_compile_options( -Wno-unknown-warning-option )

endif()

# Enable some advanced options (compiler-independent)
option( ENABLE_PROFILING "Enable profiling (gperftools-based, adds '-lprofile' flag to linker calls)" OFF )
mark_as_advanced(ENABLE_PROFILING)
if( ENABLE_PROFILING )
    find_library( PROFILER_LIBRARY profiler )
    if ( PROFILER_LIBRARY )
        add_link_options( -lprofiler )
    endif( PROFILER_LIBRARY )
endif( ENABLE_PROFILING )

# Check for memory errors using clang's AddressSanitizer. See https://github.com/google/sanitizers/wiki/AddressSanitizer
option( ENABLE_SANITIZE_ADDRESSES "Enable memory error detection (asan-based, adds '-fsanitize=address' flag)" OFF )
mark_as_advanced(ENABLE_SANITIZE_ADDRESSES)
if( ENABLE_SANITIZE_ADDRESSES )
    add_compile_options( -O1 -g -fno-omit-frame-pointer -fsanitize=address -fno-optimize-sibling-calls )
    add_link_options( -g -fno-omit-frame-pointer -fsanitize=address )
    if( CMAKE_CXX_COMPILER_ID STREQUAL "Clang" )
        # See https://github.com/google/sanitizers/issues/1017
        add_compile_options( -mllvm -asan-use-private-alias=1 )
    endif()
endif( ENABLE_SANITIZE_ADDRESSES )

# Force coloured compiler output (useful for Ninja builds)
if( CMAKE_CXX_COMPILER_ID STREQUAL "GNU" OR CMAKE_CXX_COMPILER_ID STREQUAL "Clang" )
    option( FORCE_COLOURED_OUTPUT "Always produce ANSI-colored output (GNU/Clang only)." OFF )
    if( NOT DEFINED FORCE_COLOURED_OUTPUT AND CMAKE_GENERATOR MATCHES "Ninja" )
        message(STATUS "Building with Ninja - enabling coloured compiler output.")
        set( FORCE_COLOURED_OUTPUT ON CACHE BOOL "" FORCE )
    endif()
    if( FORCE_COLOURED_OUTPUT )
        if( CMAKE_CXX_COMPILER_ID STREQUAL "GNU" )
           add_compile_options( -fdiagnostics-color=always )
        elseif( CMAKE_CXX_COMPILER_ID STREQUAL "Clang" )
           add_compile_options( -fcolor-diagnostics )
        endif()
    endif( FORCE_COLOURED_OUTPUT )
endif()

mark_as_advanced( CLEAR CMAKE_VERBOSE_MAKEFILE )

# Global dependency options
#find_package(Boost QUIET)
#option( KASPER_USE_BOOST "Build Boost dependent modules" ${Boost_FOUND} )
set(KASPER_USE_BOOST ON)

find_package(ROOT QUIET)
if(ROOT_FOUND AND NOT DEFINED KASPER_USE_ROOT)
    message("ROOT was found and enabled automatically. Set KASPER_USE_ROOT=OFF to change this.")
endif()
option( KASPER_USE_ROOT "Build ROOT dependent modules" ${ROOT_FOUND} )

find_package(GSL QUIET)
if(GSL_FOUND AND NOT DEFINED KASPER_USE_GSL)
    message("GSL was found and enabled automatically. Set KASPER_USE_GSL=OFF to change this.")
endif()
option( KASPER_USE_GSL "Build GSL dependent modules" ${GSL_FOUND} )

if(NOT DEFINED KASPER_USE_VTK)
    find_package(VTK QUIET COMPONENTS vtkCommonCore)
    find_package(X11 QUIET)
    find_package(GLEW QUIET)
    find_package(Freetype QUIET)
    find_package(JsonCpp QUIET)
    find_package(nlohmann_json QUIET) # Workaround for https://gitlab.kitware.com/vtk/vtk/-/issues/19588
    if(VTK_FOUND)
        message("VTK was found and enabled automatically. Set KASPER_USE_VTK=OFF to change this.")
    endif()
elseif(KASPER_USE_VTK)
    find_package(VTK REQUIRED COMPONENTS vtkCommonCore)
    find_package(X11 QUIET)
    find_package(GLEW QUIET)
    find_package(Freetype QUIET)
    find_package(JsonCpp QUIET)
    find_package(nlohmann_json QUIET) # Workaround for https://gitlab.kitware.com/vtk/vtk/-/issues/19588
endif()
option( KASPER_USE_VTK "Build VTK dependent modules" ${VTK_FOUND} )

option( KASPER_USE_XGBOOST "Build XGBOOST dependent modules" OFF )

# Make sure that version requirements are fulfilled
if( KASPER_USE_BOOST )
    set(BOOST_MINIMUM_VERSION "1.65")
    find_package(Boost ${BOOST_MINIMUM_VERSION} REQUIRED)
endif( KASPER_USE_BOOST )

if( KASPER_USE_ROOT )
    set(ROOT_MINIMUM_VERSION "6.16")
    find_package(ROOT ${ROOT_MINIMUM_VERSION} CONFIG REQUIRED)
endif( KASPER_USE_ROOT )

if( KASPER_USE_GSL )
    find_package(GSL REQUIRED)
endif( KASPER_USE_GSL )

# Make sure all dependencies are being built
# Every module must be listed before its dependencies and activate ONLY its
# direct dependencies.

if( BUILD_KASSIOPEIA )
    set( BUILD_KEMFIELD ON CACHE BOOL "${BUILD_KEMFIELD_DESC} (Required)" FORCE )
    set( BUILD_KGEOBAG ON CACHE BOOL "${BUILD_KGEOBAG_DESC} (Required)" FORCE )
    set( BUILD_KOMMON ON CACHE BOOL "${BUILD_KOMMON_DESC} (Required)" FORCE )

    if( NOT KASPER_USE_GSL )
        message("Kassiopeia requires GSL to be enabled in Kasper (activated automatically)" )
    endif()
    set( KASPER_USE_GSL ON CACHE BOOL "(Required)" FORCE )
endif( BUILD_KASSIOPEIA )

if( BUILD_KEMFIELD )
    set( BUILD_KGEOBAG ON CACHE BOOL "${BUILD_KGEOBAG_DESC} (Required)" FORCE )
    set( BUILD_KOMMON ON CACHE BOOL "${BUILD_KOMMON_DESC} (Required)" FORCE )
endif( BUILD_KEMFIELD )

if( BUILD_KGEOBAG )
    set( BUILD_KOMMON ON CACHE BOOL "${BUILD_KOMMON_DESC} (Required)" FORCE )

    if( NOT KASPER_USE_GSL )
        message("KGeoBag requires GSL to be enabled in Kasper (activated automatically)" )
    endif()
    set( KASPER_USE_GSL ON CACHE BOOL "(Required)" FORCE )
endif( BUILD_KGEOBAG )

# Build modules
# NOTE: The order matters! Please place dependent modules at the end.

if( BUILD_KOMMON )
    add_subdirectory( Kommon )
endif()

if( BUILD_KGEOBAG )
    # needs Kommon
    add_subdirectory( KGeoBag )
endif()

if( BUILD_KEMFIELD )
    # needs Kommon, KGeoBag
    add_subdirectory( KEMField )
endif()

if( BUILD_KASSIOPEIA )
    # needs Kommon, KGeoBag, KEMField
    add_subdirectory( Kassiopeia )
endif()

if( BUILD_UNIT_TESTS )
    add_subdirectory( UnitTest )
endif()

add_subdirectory( Documentation )

# Print some warnings

if( BUILD_KASSIOPEIA OR BUILD_KEMFIELD OR BUILD_KGEOBAG )
    if( NOT KASPER_USE_VTK )
        message(WARNING "*** Building without VTK support ***\n"
            "You won't be able to use 3D visualization tools. Set KASPER_USE_VTK=ON to change this.")
    endif( NOT KASPER_USE_VTK )
endif()

if( BUILD_KASSIOPEIA )
    if( NOT KASPER_USE_ROOT )
        message(WARNING "*** Building without ROOT support ***\n"
            "You won't be able to read/write ROOT files. Set KASPER_USE_ROOT=ON to change this.")
    endif( NOT KASPER_USE_ROOT )
endif()

get_filename_component(CMAKE_MAKE_PROGRAM_NAME ${CMAKE_MAKE_PROGRAM} NAME)
message(STATUS
    "Build was configured. Run `${CMAKE_MAKE_PROGRAM_NAME}` to start the build,\n"
    "   or `${CMAKE_MAKE_PROGRAM_NAME} install` to install to ${KASPER_INSTALL_DIR},\n"
    "   or `${CMAKE_MAKE_PROGRAM_NAME} doc` to generate the documentation.\n")

# Install common project files
configure_file( kasperenv.sh.in ${CMAKE_CURRENT_BINARY_DIR}/kasperenv.sh @ONLY )
configure_file( create_kasper_user_directory.sh.in ${CMAKE_CURRENT_BINARY_DIR}/create_kasper_user_directory.sh @ONLY )
install( PROGRAMS ${CMAKE_CURRENT_BINARY_DIR}/kasperenv.sh ${CMAKE_CURRENT_BINARY_DIR}/create_kasper_user_directory.sh DESTINATION ${BIN_INSTALL_DIR} )
