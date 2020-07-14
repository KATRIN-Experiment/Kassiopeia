set( CMAKE_MINIMUM_VERSION "3.13.0" )
cmake_minimum_required( VERSION ${CMAKE_MINIMUM_VERSION} )

# project version
set(KASPER_VERSION_MAJOR 3)
set(KASPER_VERSION_MINOR 7)
set(KASPER_VERSION_PATCH 5)
set(KASPER_VERSION "${KASPER_VERSION_MAJOR}.${KASPER_VERSION_MINOR}.${KASPER_VERSION_PATCH}")

#project( Kasper VERSION ${KASPER_VERSION} )
project( Kasper )

set( BUILD_SHARED_LIBS ON )

set( CMAKE_CXX_STANDARD 14 )
set( CMAKE_CXX_STANDARD_REQUIRED ON )
set( CMAKE_CXX_EXTENSIONS OFF )

# Additional compiler version checks, see https://en.cppreference.com/w/cpp/compiler_support
if( CMAKE_SYSTEM_NAME STREQUAL "Linux" )
    if( CMAKE_CXX_COMPILER_ID STREQUAL "GNU" )
        if( CMAKE_CXX_COMPILER_VERSION VERSION_LESS "5.0" )
            message(WARNING "Your compiler is unsupported! You need at least GCC 5.0 to build Kasper.")
        endif()
    elseif( CMAKE_CXX_COMPILER_ID STREQUAL "Clang" )
        if( CMAKE_CXX_COMPILER_VERSION VERSION_LESS "3.4" )
            message(WARNING "Your compiler is unsupported! You need at least Clang 3.4 to build Kasper.")
        endif()
    endif()
elseif( CMAKE_SYSTEM_NAME STREQUAL "Darwin" )
    if( CMAKE_CXX_COMPILER_ID STREQUAL "Clang" )
        if( CMAKE_CXX_COMPILER_VERSION VERSION_LESS "10.0" )
            message(WARNING "Your compiler is unsupported! You need at least Clang 10.0 to build Kasper.")
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
if( APPLE )
    set(CMAKE_INSTALL_RPATH "@executable_path/../${CMAKE_INSTALL_LIBDIR};@loader_path")
elseif( UNIX )
    set(CMAKE_INSTALL_RPATH "$ORIGIN/../${CMAKE_INSTALL_LIBDIR}")
endif( APPLE )

# the RPATH to be used when installing, but only if it's not a system directory
#LIST( FIND CMAKE_PLATFORM_IMPLICIT_LINK_DIRECTORIES "${LIB_INSTALL_DIR}" isSystemDir )
#IF( "${isSystemDir}" STREQUAL "-1" )
#   set( CMAKE_INSTALL_RPATH "${LIB_INSTALL_DIR}" )
#ENDIF()


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

# Enable some advanced options depending on build type
if (CMAKE_BUILD_TYPE MATCHES Rel)
    option( COMPILER_TUNE_OPTIONS "Enable additional tuning options [compiles for local CPU, use with care!]" OFF )
    if( COMPILER_TUNE_OPTIONS )
        # These are advanced options for GCC which might improve performance ... or break everything - use with care!
        add_compile_options( -march=native -mfpmath=sse -funroll-loops )
    endif( COMPILER_TUNE_OPTIONS )
endif()

if( "${CMAKE_CXX_COMPILER_ID}" STREQUAL "GNU" OR CMAKE_COMPILER_IS_GNUCXX )
    if( NOT CMAKE_CXX_COMPILER_VERSION VERSION_LESS "7.0" )
        # Use -faligned-new flag to trick gcc into using aligned allocators CL types.
        # This is only used by KEMField, but required here because of include files.
        add_compile_options($<$<COMPILE_LANGUAGE:CXX>:-faligned-new>)
    endif()

    # Enable some advanced options
    option( GCC_FORCE_LINKING "Fix linker errors with some GCC versions by adding the --no-as-needed flag" ON )
    if( GCC_FORCE_LINKING )
        add_link_options( -Wl,--no-as-needed )
    endif( GCC_FORCE_LINKING )

    option( GCC_REDUCE_MEMORY "Significantly reduce GCC memory usage by setting ftrack-macro-expansion=1 (default: 2)" ON )
    if( GCC_REDUCE_MEMORY )
        add_compile_options( -ftrack-macro-expansion=1 )
    endif( GCC_REDUCE_MEMORY )

    option( GCC_ENABLE_PIPES "Use pipes to speed up compiling with GCC by adding the -pipe flag" ON )
    if( GCC_ENABLE_PIPES )
        add_compile_options( -pipe )
    endif( GCC_ENABLE_PIPES )

else( "${CMAKE_CXX_COMPILER_ID}" STREQUAL "Clang" )
    # Suppress warnings about unknown warning options. See https://bugs.llvm.org/show_bug.cgi?id=24979
    add_compile_options( -Wno-unknown-warning-option )

endif()

# Enable some advanced options (compiler-independent)
option( ENABLE_PROFILING "Enable profiling (gperftools-based, adds '-lprofile' flag to linker calls)" OFF )
if( ENABLE_PROFILING )
    find_library( PROFILER_LIBRARY profiler )
    if ( PROFILER_LIBRARY )
        add_link_options( -lprofiler )
    endif( PROFILER_LIBRARY )
endif( ENABLE_PROFILING )

# Force coloured compiler output (useful for Ninja builds)
if( "${CMAKE_CXX_COMPILER_ID}" STREQUAL "GNU" OR "${CMAKE_CXX_COMPILER_ID}" STREQUAL "Clang" )
    option( FORCE_COLOURED_OUTPUT "Always produce ANSI-colored output (GNU/Clang only)." OFF )
    if( NOT DEFINED FORCE_COLOURED_OUTPUT AND "${CMAKE_GENERATOR}" MATCHES "Ninja" )
        message(STATUS "Building with Ninja - enabling coloured compiler output.")
        set( FORCE_COLOURED_OUTPUT ON CACHE BOOL "" FORCE )
    endif()
    if( FORCE_COLOURED_OUTPUT )
        if( "${CMAKE_CXX_COMPILER_ID}" STREQUAL "GNU" )
           add_compile_options( -fdiagnostics-color=always )
        elseif( "${CMAKE_CXX_COMPILER_ID}" STREQUAL "Clang" )
           add_compile_options( -fcolor-diagnostics )
        endif()
    endif( FORCE_COLOURED_OUTPUT )
endif()

mark_as_advanced( CLEAR CMAKE_VERBOSE_MAKEFILE )


# Global dependency options
option( KASPER_USE_BOOST "Build Boost dependent modules" ON )
option( KASPER_USE_ROOT "Build ROOT dependent modules" ON )
option( KASPER_USE_GSL "Build GSL dependent modules" OFF )
option( KASPER_USE_VTK "Build VTK dependent modules" OFF )

# Advanced options mainly for testing
option( KASPER_EXPERIMENTAL "Enable some experimental features" OFF )
if( KASPER_EXPERIMENTAL )
    option( KASPER_USE_TBB "Build TBB dependent modules" OFF )
    if( KASPER_USE_TBB )
        message( "WARNING: Enabling TBB can produce inconsistent or wrong results in KaFit. It is strongly recommended to leave this option disabled until further investigation." )
    endif()
endif( KASPER_EXPERIMENTAL )

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
    set( KASPER_USE_BOOST ON CACHE BOOL "(Required)" FORCE )
endif( BUILD_KGEOBAG )

# Build modules
# The order seems to matter! Please place dependent modules at the end.

if( BUILD_KOMMON )
    add_subdirectory( Kommon )
endif()

if( BUILD_KGEOBAG )
    add_subdirectory( KGeoBag )
endif()

if( BUILD_KEMFIELD )
    add_subdirectory( KEMField )
endif()

if( BUILD_KASSIOPEIA )
    add_subdirectory( Kassiopeia )
endif()

if( BUILD_UNIT_TESTS )
    add_subdirectory( UnitTest )
endif()

#add_subdirectory( Documentation )

install( EXPORT KasperTargets DESTINATION ${CMAKE_INSTALL_DIR}/{PROJECT_NAME} )

configure_file( kasperenv.sh.in ${CMAKE_CURRENT_BINARY_DIR}/kasperenv.sh @ONLY )
configure_file( create_kasper_user_directory.sh.in ${CMAKE_CURRENT_BINARY_DIR}/create_kasper_user_directory.sh @ONLY )
install( PROGRAMS ${CMAKE_CURRENT_BINARY_DIR}/kasperenv.sh ${CMAKE_CURRENT_BINARY_DIR}/create_kasper_user_directory.sh DESTINATION ${BIN_INSTALL_DIR} )
