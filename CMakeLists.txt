cmake_minimum_required( VERSION 2.6 )
project( KassiopeiaMC )

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

set( CMAKE_C_FLAGS ${CMAKE_C_FLAGS_INIT} )
set( CMAKE_CXX_FLAGS ${CMAKE_CXX_FLAGS_INIT} )
set( CMAKE_EXE_LINKER_FLAG ${CMAKE_EXE_LINKER_FLAGS_INIT} )

option( GCC_FIX_LINKER_ERROR "Fix linker errors with some GCC versions by adding the --no-as-needed flag" OFF )
if( GCC_FIX_LINKER_ERROR )
    set( CMAKE_EXE_LINKER_FLAGS "${CMAKE_EXE_LINKER_FLAGS} -Wl,--no-as-needed" )
endif( GCC_FIX_LINKER_ERROR )

option( GCC_ENABLE_PIPES "Use pipes to speed up compiling with GCC by adding the -pipe flag" OFF )
if( GCC_ENABLE_PIPES )
    set( CMAKE_C_FLAGS "${CMAKE_CXX_FLAGS} -pipe" )
    set( CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -pipe" )
endif( GCC_ENABLE_PIPES )

option( GCC_ENABLE_CPP11 "Enable C++11 support in GCC by adding the --std=c++0x flag" OFF )
if( GCC_ENABLE_CPP11 )
    set( CMAKE_C_FLAGS "${CMAKE_C_FLAGS} --std=c++0x" )
    set( CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} --std=c++0x" )
endif( GCC_ENABLE_CPP11 )

option( GCC_TUNE_OPTIONS "Tune some tuning options to GCC [compiles for local CPU, use with care!]" OFF )
if( GCC_TUNE_OPTIONS )
    # these are advanced options for GCC which might improve performance ... or break everything - use with care!
    set( GCC_TUNING_OPTIONS "-O3 -funroll-loops -march=native -mtune=native -mfpmath=sse" )
    set( CMAKE_C_FLAGS "${CMAKE_C_FLAGS} ${GCC_TUNING_OPTIONS}" )
    set( CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${GCC_TUNING_OPTIONS}" )
endif( GCC_TUNE_OPTIONS )

option( GCC_ENABLE_PROFILING "Enable profiling( gprof-based, adds -pg flag to GCC calls )" OFF )
if( GCC_ENABLE_PROFILING )
    set( CMAKE_C_FLAGS "${CMAKE_C_FLAGS} -g -pg" )
    set( CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -g -pg" )
    set( CMAKE_EXE_LINKER_FLAGS" ${CMAKE_EXE_LINKER_FLAGS} -g -pg" )
endif( GCC_ENABLE_PROFILING )

mark_as_advanced( CLEAR CMAKE_VERBOSE_MAKEFILE )

# Make sure all dependencies are being built
# Every module must be listed before its dependencies and activate ONLY its
# direct depencies.

if(BUILD_DESASTER)
    set(BUILD_TABREE ON CACHE BOOL "${BUILD_TABREE_DESC} (Required)" FORCE)
    set(BUILD_KEBAP ON CACHE BOOL "${BUILD_KEBAP_DESC} (Required)" FORCE)
    set(BUILD_KALI ON CACHE BOOL "${BUILD_KALI_DESC} (Required)" FORCE)
    set(BUILD_KASSIOPEIA ON CACHE BOOL "${BUILD_KASSIOPEIA_DESC} (Required)" FORCE)
    set(BUILD_KSC ON CACHE BOOL "${BUILD_KSC_DESC} (Required)" FORCE)
    set(KSC_USE_KALI ON CACHE BOOL "${BUILD_KSC_DESC} (Required)" FORCE)
endif()

if(BUILD_KAFIT)
    set(BUILD_KOMMON ON CACHE BOOL "${BUILD_KOMMON_DESC} (Required)" FORCE)
    set( Kommon_USE_GSL ON CACHE BOOL "(Required)" FORCE )
    set( Kommon_USE_BOOST ON CACHE BOOL "(Required)" FORCE )
    
    set(BUILD_KALI ON CACHE BOOL "${BUILD_KALI_DESC} (Required)" FORCE)
    set(BUILD_SSC ON CACHE BOOL "${BUILD_SSC_DESC} (Required)" FORCE)
endif()

if( BUILD_KSC )
	set( BUILD_KGEOBAG ON CACHE BOOL "${BUILD_KGEOBAG_DESC} (Required)" FORCE )
	set( BUILD_KEMFIELD ON CACHE BOOL "${BUILD_KEMFIELD_DESC} (Required)" FORCE )
    if(KSC_USE_KASSIOPEIA)
	    set( BUILD_KASSIOPEIA ON CACHE BOOL "${BUILD_KASSIOPEIA_DESC} (Required)" FORCE )
    endif(KSC_USE_KASSIOPEIA)
    if( KSC_USE_SSC )
		set( BUILD_SSC ON CACHE BOOL "${BUILD_SSC_DESC} (Required)" FORCE )
	endif( KSC_USE_SSC )
	set( Kassiopeia_EXECUTABLES OFF CACHE BOOL "(Required)" FORCE )
    if(KSC_USE_KALI)
	    set( BUILD_KALI ON CACHE BOOL "${BUILD_KALI_DESC} (Required)" FORCE )
    endif(KSC_USE_KALI)
endif()

if( BUILD_SSC )
	set( BUILD_KOMMON ON CACHE BOOL "${BUILD_KOMMON_DESC} (Required)" FORCE )
	set( Kommon_USE_BOOST ON CACHE BOOL "(Required)" FORCE )
    set( Kommon_USE_ROOT ON CACHE BOOL "(Required)" FORCE )
    set( BUILD_KALI ON CACHE BOOL "${BUILD_KALI_DESC} (Required)" FORCE )
	set( BUILD_KASSIOPEIA ON CACHE BOOL "${BUILD_KASSIOPEIA_DESC} (Required)" FORCE )
	set( BUILD_KGEOBAG ON CACHE BOOL "${BUILD_KGEOBAG_DESC} (Required)" FORCE )
	set( BUILD_TABREE ON CACHE BOOL "${BUILD_TABREE_DESC} (Required)" FORCE )
endif()

if( BUILD_KASSIOPEIA )
    set( BUILD_KOMMON ON CACHE BOOL "${BUILD_KOMMON_DESC} (Required)" FORCE )
    set( BUILD_KGEOBAG ON CACHE BOOL "${BUILD_KGEOBAG_DESC} (Required)" FORCE )
    set( BUILD_KEMFIELD ON CACHE BOOL "${BUILD_KEMFIELD_DESC} (Required)" FORCE )

    set( Kommon_USE_GSL ON CACHE BOOL "(Required)" FORCE )
    if( Kassiopeia_USE_ROOT )
	    set( Kommon_USE_ROOT ON CACHE BOOL "(Required)" FORCE )
	    set( KGeoBag_USE_ROOT ON CACHE BOOL "(Required)" FORCE )
	endif( Kassiopeia_USE_ROOT )

    if( Kassiopeia_USE_VTK )
    	set( Kommon_USE_VTK ON CACHE BOOL "(Required)" FORCE )
		set( KGeoBag_USE_VTK ON CACHE BOOL "(Required)" FORCE )
	endif( Kassiopeia_USE_VTK )
endif()

if( BUILD_KGEOBAG )
    set( BUILD_KOMMON ON CACHE BOOL "${BUILD_KOMMON_DESC} (Required)" FORCE )

    set( Kommon_USE_GSL ON CACHE BOOL "(Required)" FORCE )

    if( KGeoBag_USE_ROOT )
    	set( Kommon_USE_ROOT ON CACHE BOOL "(Required)" FORCE )
	endif( KGeoBag_USE_ROOT )

    if( KGeoBag_USE_VTK )
    	set( Kommon_USE_VTK ON CACHE BOOL "(Required)" FORCE )
	endif( KGeoBag_USE_VTK )
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

#add_subdirectory( Documentation )

install( EXPORT KasperTargets DESTINATION ${CMAKE_INSTALL_DIR} )

configure_file( Kommon/kasperenv.sh ${CMAKE_CURRENT_BINARY_DIR}/kasperenv.sh @ONLY )
install( PROGRAMS ${CMAKE_CURRENT_BINARY_DIR}/kasperenv.sh DESTINATION ${BIN_INSTALL_DIR} )
