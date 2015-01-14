# attention:
# i spent some time making this file neat and organized.
# this is important as it allows changes to be performed quickly.
# please be considerate and make additions as neatly as you can.
# thank you! -dan.

cmake_minimum_required( VERSION 2.6 )
project( Kassiopeia )
include( KasperDefaults )

# module version
set( MODULE_VERSION_MAJOR 3 )
set( MODULE_VERSION_MINOR 1 )
set( MODULE_VERSION_PATCH 0 )
set( MODULE_VERSION "${MODULE_VERSION_MAJOR}.${MODULE_VERSION_MINOR}.${MODULE_VERSION_PATCH}" )

# module options
option( Kassiopeia_USE_ROOT "build in root functionality [writers, visualization]" ON )
option( Kassiopeia_USE_VTK "build in vtk functionality [visualization]" ON )
option( Kassiopeia_EXECUTABLES "build kassiopeia executable" ON )

# module paths
kasper_module_paths( Kassiopeia )

# module debugging
kasper_module_debug()
if( Kassiopeia_ENABLE_DEBUG )
	add_cflag( Kassiopeia_ENABLE_DEBUG )
endif()


# external dependencies: root
if( Kassiopeia_USE_ROOT )
	find_package( ROOT 5.24.0 REQUIRED )
	kasper_external_include_directories( ${ROOT_INCLUDE_DIRS} )
	add_cflag( Kassiopeia_USE_ROOT )
endif( Kassiopeia_USE_ROOT )

# external dependencies: vtk
if( Kassiopeia_USE_VTK )
	find_package( VTK REQUIRED )
	include( ${VTK_USE_FILE} )
    kasper_external_include_directories( ${VTK_INCLUDE_DIRS} )
	add_cflag( Kassiopeia_USE_VTK )
endif( Kassiopeia_USE_VTK )


# internal dependencies: kommon
kasper_find_module( Kommon )
foreach( FLAG ${Kommon_CFLAGS} )
	add_definitions( -D${FLAG} )
endforeach( FLAG )

# internal dependencies: kgeobag
kasper_find_module( KGeoBag )
foreach( FLAG ${KGeoBag_CFLAGS} )
	add_definitions( -D${FLAG} )
endforeach( FLAG )

# internal dependencies: kemfield
kasper_find_module( KEMField )
if( KEMField_USE_MPI )
  message( FATAL_ERROR "In order to compile Kassiopeia please rebuild KEMField with MPI disabled." )
endif()

foreach( FLAG ${KEMField_CFLAGS} )
	add_definitions( -D${FLAG} )
endforeach( FLAG )

# subdirectories
add_subdirectory( Utility )
add_subdirectory( Math )
add_subdirectory( Objects )
add_subdirectory( Operators )
add_subdirectory( Fields )
add_subdirectory( Geometry )
add_subdirectory( Generators )
add_subdirectory( Trajectories )
add_subdirectory( Navigators )
add_subdirectory( Interactions )
add_subdirectory( Terminators )
add_subdirectory( Modifiers )
add_subdirectory( Writers )
add_subdirectory( Readers )
add_subdirectory( XML )
add_subdirectory( Visualization )
add_subdirectory( Simulation )
add_subdirectory( Bindings )

if( Kassiopeia_EXECUTABLES )
	add_subdirectory( Applications/Simulation )
endif( Kassiopeia_EXECUTABLES )

add_subdirectory( Applications/Validation )
if( Kassiopeia_USE_ROOT )
	add_subdirectory( Applications/Examples )
	add_subdirectory( Applications/Other )
endif( Kassiopeia_USE_ROOT )

kasper_install_module()
