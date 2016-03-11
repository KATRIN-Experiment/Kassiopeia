cmake_minimum_required( VERSION 2.8.6 )
if(POLICY CMP0048)
    cmake_policy(SET CMP0048 OLD)
endif()

project( Kommon )
set( CMAKE_MODULE_PATH ${PROJECT_SOURCE_DIR}/cmake ${PROJECT_SOURCE_DIR}/GoogleTest/cmake )
include( KasperDefaults )

# version
set( MODULE_VERSION_MAJOR 1 )
set( MODULE_VERSION_MINOR 1 )
set( MODULE_VERSION_PATCH 0 )
set( MODULE_VERSION "${MODULE_VERSION_MAJOR}.${MODULE_VERSION_MINOR}.${MODULE_VERSION_PATCH}" )

# build options
option( Kommon_USE_BOOST "Build Boost dependent Kommon modules" OFF )
option( Kommon_USE_ROOT "Build ROOT dependent Kommon modules" OFF )
option( Kommon_USE_GSL "Build GSL dependent Kommon modules" OFF )
option( Kommon_USE_VTK "Build VTK dependent Kommon modules" OFF )
option( Kommon_USE_TBB "Build TBB dependent Kommon modules" OFF )

# compiler requirements
kasper_require_cpp11()
kasper_require_gcc_version(4.6)
kasper_require_clang_version(3.0)

# paths
kasper_module_paths( Kommon )

# debugging
kasper_module_debug()
if( Kommon_ENABLE_DEBUG )
	add_cflag (Kommon_ENABLE_DEBUG)
endif()

# subdirectories
add_subdirectory( cmake )
add_subdirectory( Core )
add_subdirectory( Documentation )

if( Kommon_USE_BOOST )
	add_subdirectory( Extensions/Boost )
endif( Kommon_USE_BOOST )

if( Kommon_USE_ROOT )
	add_subdirectory( Extensions/Root )
	add_cflag (Kommon_USE_ROOT)
endif( Kommon_USE_ROOT )

if( Kommon_USE_GSL )
	add_subdirectory( Extensions/Gsl )
endif( Kommon_USE_GSL )

if( Kommon_USE_VTK )
	add_subdirectory( Extensions/Vtk )
endif( Kommon_USE_VTK )

if( Kommon_USE_TBB )
    add_subdirectory( Extensions/TBB )
endif( Kommon_USE_TBB )

# testing
kasper_module_test( Test )

kasper_install_module()
