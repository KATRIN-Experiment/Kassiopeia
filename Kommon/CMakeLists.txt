cmake_minimum_required( VERSION ${CMAKE_MINIMUM_VERSION} )

# Kommon version
set( MODULE_VERSION_MAJOR 1 )
set( MODULE_VERSION_MINOR 6 )
set( MODULE_VERSION_PATCH 0 )
set( MODULE_VERSION "${MODULE_VERSION_MAJOR}.${MODULE_VERSION_MINOR}.${MODULE_VERSION_PATCH}" )

#project( Kommon VERSION ${MODULE_VERSION} )
project( Kommon )

include( KasperDefaults )

# paths
kasper_module_paths( Kommon )

# debugging
kasper_module_debug()

# # to be populated by subdirectories
# set(KOMMON_HEADER_FILES)
# set(KOMMON_SOURCE_FILES)

set( KASPER_USE_TBB OFF CACHE BOOL "Use the Intel TBB library in Kommon")
mark_as_advanced( FORCE KASPER_USE_TBB )

# subdirectories
add_subdirectory( cmake )
add_subdirectory( Documentation )

add_subdirectory( Base )

if( KASPER_USE_BOOST )
    add_subdirectory( Boost )
endif( KASPER_USE_BOOST )

add_subdirectory( Core )
if( KASPER_USE_TBB )
    # NOTE: This is not a separate library, but part of KommonCore. It *must* come after `Core`!
    add_subdirectory( TBB )
endif( KASPER_USE_TBB )

add_subdirectory( Binning )
add_subdirectory( Math )

if( KASPER_USE_GSL )
    add_subdirectory( Gsl )
endif( KASPER_USE_GSL )

if( KASPER_USE_ROOT )
    add_subdirectory( Root )
endif( KASPER_USE_ROOT )

if( KASPER_USE_VTK )
        add_subdirectory( Vtk )
endif()

add_subdirectory( Applications )

kasper_install_module()

# install extra cmake files needed by external software
install(DIRECTORY cmake/
    DESTINATION ${CMAKE_INSTALL_DIR}
    FILES_MATCHING PATTERN "*.cmake" PATTERN "*.c" PATTERN "*.h"
)
