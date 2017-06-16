cmake_minimum_required( VERSION 2.8.6 )

project( Kasper )

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

set( CMAKE_INSTALL_RPATH "${CMAKE_INSTALL_PREFIX}/lib" )

# the RPATH to be used when installing, but only if it's not a system directory
LIST( FIND CMAKE_PLATFORM_IMPLICIT_LINK_DIRECTORIES "${CMAKE_INSTALL_PREFIX}/lib" isSystemDir )
IF( "${isSystemDir}" STREQUAL "-1" )
   set( CMAKE_INSTALL_RPATH "${CMAKE_INSTALL_PREFIX}/lib" )
ENDIF( "${isSystemDir}" STREQUAL "-1" )


# Flags for all modules, every module should have a flag.

set( BUILD_KOMMON_DESC "Build Kommon" )
option( BUILD_KOMMON ${BUILD_KOMMON_DESC} ON )

set( BUILD_KGEOBAG_DESC "Build KGeoBag [geometry system]" )
option( BUILD_KGEOBAG ${BUILD_KGEOBAG_DESC} ON )

set( BUILD_KEMFIELD_DESC "Build KEMField [electromagnetic simulation system]" )
option( BUILD_KEMFIELD ${BUILD_KEMFIELD_DESC} ON )

set( BUILD_KASSIOPEIA_DESC "Build Kassiopeia [particle tracking simulation system]" )
option( BUILD_KASSIOPEIA ${BUILD_KASSIOPEIA_DESC} ON )

#set( CMAKE_C_FLAGS ${CMAKE_C_FLAGS_INIT} )
#set( CMAKE_CXX_FLAGS ${CMAKE_CXX_FLAGS_INIT} )
#set( CMAKE_EXE_LINKER_FLAG ${CMAKE_EXE_LINKER_FLAGS_INIT} )

if (CMAKE_BUILD_TYPE MATCHES Rel)
    option( COMPILER_TUNE_OPTIONS "Enable additional tuning options [compiles for local CPU, use with care!]" OFF )
    if( COMPILER_TUNE_OPTIONS )
        # these are advanced options for GCC which might improve performance ... or break everything - use with care!
        set( TUNING_OPTIONS "-march=native -mfpmath=sse -funroll-loops" )
        set( CMAKE_C_FLAGS "${CMAKE_C_FLAGS} ${TUNING_OPTIONS}" )
        set( CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${TUNING_OPTIONS}" )
    endif()
endif()

option( COMPILER_ENABLE_PROFILING "Enable profiling( gprof-based, adds -pg flag to GCC calls )" OFF )
if( COMPILER_ENABLE_PROFILING )
    set( CMAKE_C_FLAGS "${CMAKE_C_FLAGS} -g -pg" )
    set( CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -g -pg" )
    set( CMAKE_EXE_LINKER_FLAGS" ${CMAKE_EXE_LINKER_FLAGS} -g -pg" )
endif()

if (CMAKE_COMPILER_IS_GNUCXX)
    option( GCC_FORCE_LINKING "Fix linker errors with some GCC versions by adding the --no-as-needed flag" ON )
    if( GCC_FORCE_LINKING )
        set( CMAKE_EXE_LINKER_FLAGS "${CMAKE_EXE_LINKER_FLAGS} -Wl,--no-as-needed" )
    endif( )

    option( GCC_REDUCE_MEMORY "Significantly reduce GCC memory usage by setting ftrack-macro-expansion=1 (default: 2)" OFF )
    if( GCC_REDUCE_MEMORY )
        set( CMAKE_C_FLAGS "${CMAKE_C_FLAGS} -ftrack-macro-expansion=1" )
        set( CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -ftrack-macro-expansion=1" )
    endif( GCC_REDUCE_MEMORY )

    option( GCC_ENABLE_PIPES "Use pipes to speed up compiling with GCC by adding the -pipe flag" OFF )
    if( GCC_ENABLE_PIPES )
        set( CMAKE_C_FLAGS "${CMAKE_C_FLAGS} -pipe" )
        set( CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -pipe" )
    endif( GCC_ENABLE_PIPES )
endif()

mark_as_advanced( CLEAR CMAKE_VERBOSE_MAKEFILE )


# Global dependency options
option( KASPER_USE_BOOST "Build Boost dependent modules" OFF )
option( KASPER_USE_ROOT "Build ROOT dependent modules" ON )
option( KASPER_USE_GSL "Build GSL dependent modules" OFF )
option( KASPER_USE_VTK "Build VTK dependent modules" OFF )
option( KASPER_USE_TBB "Build TBB dependent modules" OFF )

# Make sure all dependencies are being built
# Every module must be listed before its dependencies and activate ONLY its
# direct dependencies.

if( BUILD_KASSIOPEIA )
    set( BUILD_KOMMON ON CACHE BOOL "${BUILD_KOMMON_DESC} (Required)" FORCE )
    set( BUILD_KGEOBAG ON CACHE BOOL "${BUILD_KGEOBAG_DESC} (Required)" FORCE )
    set( BUILD_KEMFIELD ON CACHE BOOL "${BUILD_KEMFIELD_DESC} (Required)" FORCE )

    set( KASPER_USE_GSL ON CACHE BOOL "(Required)" FORCE )
endif()

if( BUILD_KGEOBAG )
    set( BUILD_KOMMON ON CACHE BOOL "${BUILD_KOMMON_DESC} (Required)" FORCE )
    set( KASPER_USE_GSL ON CACHE BOOL "(Required)" FORCE )
endif()

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

if( BUILD_KTEMPLATE )
    add_subdirectory( KTemplate )
endif()

#add_subdirectory( Documentation )

install( EXPORT KasperTargets DESTINATION ${CMAKE_INSTALL_DIR} )

configure_file( Kommon/kasperenv.sh ${CMAKE_CURRENT_BINARY_DIR}/kasperenv.sh @ONLY )
configure_file( Kommon/create_kasper_user_directory.sh ${CMAKE_CURRENT_BINARY_DIR}/create_kasper_user_directory.sh @ONLY )
install( PROGRAMS ${CMAKE_CURRENT_BINARY_DIR}/kasperenv.sh ${CMAKE_CURRENT_BINARY_DIR}/create_kasper_user_directory.sh DESTINATION ${BIN_INSTALL_DIR} )
