# attention:
# i spent some time making this file neat and organized.
# this is important as it allows changes to be performed quickly.
# please be considerate and make additions as neatly as you can.
# thank you! -dan.

cmake_minimum_required( VERSION ${CMAKE_MINIMUM_VERSION} )

if(POLICY CMP0053)
    cmake_policy(SET CMP0053 NEW)
endif()

# Kassiopeia version
set( MODULE_VERSION_MAJOR 4 )
set( MODULE_VERSION_MINOR 1 )
set( MODULE_VERSION_PATCH 0 )
set( MODULE_VERSION "${MODULE_VERSION_MAJOR}.${MODULE_VERSION_MINOR}.${MODULE_VERSION_PATCH}" )

#project( Kassiopeia VERSION ${MODULE_VERSION} )
project( Kassiopeia )

include( KasperDefaults )

# module options
# build in boost functionality [utilities]
SET( Kassiopeia_USE_BOOST ${KASPER_USE_BOOST} )
# build in root functionality [writers, visualization]
SET( Kassiopeia_USE_ROOT ${KASPER_USE_ROOT} )
# build in vtk functionality [visualization]
SET( Kassiopeia_USE_VTK ${KASPER_USE_VTK} )

# module paths
kasper_module_paths( Kassiopeia )

# module debugging
kasper_module_debug()

# subdirectories
add_subdirectory( Documentation )
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
add_subdirectory( Simulation )
add_subdirectory( Readers )
add_subdirectory( Visualization )
add_subdirectory( Bindings )
add_subdirectory( Python )
add_subdirectory( Applications/Simulation )
add_subdirectory( Applications/FieldCalculation )
add_subdirectory( Applications/Validation )
add_subdirectory( Applications/Examples )
add_subdirectory( Applications/Other )
add_subdirectory( Applications/Visualization )
add_subdirectory( XML )

kasper_install_module()

# Install code examples
install(DIRECTORY AnalysisExamples/ DESTINATION ${KASPER_INSTALL_DIR}/examples/Kassiopeia)
