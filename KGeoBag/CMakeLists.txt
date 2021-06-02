cmake_minimum_required( VERSION ${CMAKE_MINIMUM_VERSION} )

# KGeoBag version
set( MODULE_VERSION_MAJOR 3 )
set( MODULE_VERSION_MINOR 7 )
set( MODULE_VERSION_PATCH 8 )
set( MODULE_VERSION "${MODULE_VERSION_MAJOR}.${MODULE_VERSION_MINOR}.${MODULE_VERSION_PATCH}" )

#project( KGeoBag VERSION ${MODULE_VERSION})
project( KGeoBag )

include( KasperDefaults )

#option( KGeoBag_USE_VTK "Enable visualization with VTK" OFF )
#option( KGeoBag_USE_ROOT "Enable visualization with ROOT" OFF )
set( KGeoBag_USE_BOOST ${KASPER_USE_BOOST} )
set( KGeoBag_USE_VTK ${KASPER_USE_VTK} )
set( KGeoBag_USE_ROOT ${KASPER_USE_ROOT} )
set( KGeoBag_USE_GSL ${KASPER_USE_GSL} )

if( KGeoBag_USE_GSL )
    find_package( GSL REQUIRED )
endif( KGeoBag_USE_GSL )

if( KGeoBag_USE_BOOST )
    find_package( Boost ${BOOST_MINIMUM_VERSION} REQUIRED )
endif()

if( KGeoBag_USE_VTK )
    find_package(VTK CONFIG REQUIRED)
endif()

if( KGeoBag_USE_ROOT )
    find_package( ROOT ${ROOT_MINIMUM_VERSION} CONFIG REQUIRED )
endif()

# installing
kasper_module_paths( TheBag )

# debugging
kasper_module_debug()

## testing
kasper_module_test( Source/Test )

# recursively build in these subdirectories
add_subdirectory( Documentation )
add_subdirectory( Source/Math )
add_subdirectory( Source/Core )
add_subdirectory( Source/Math/LinearAlgebra )
add_subdirectory( Source/Math/SpaceTree )
add_subdirectory( Source/Math/2DUtility )
add_subdirectory( Source/Shapes )
add_subdirectory( Source/Extensions/Mesh )
add_subdirectory( Source/Extensions/AxialMesh )
add_subdirectory( Source/Extensions/DiscreteRotationalMesh )
add_subdirectory( Source/Extensions/Appearance )
add_subdirectory( Source/Extensions/Metrics )
add_subdirectory( Source/Extensions/Random )
add_subdirectory( Source/Extensions/Intersection )
add_subdirectory( Source/Extensions/Deformation )
add_subdirectory( Source/Visualization )
add_subdirectory( Source/Bindings )
add_subdirectory( Source/Application )
add_subdirectory( Source/Test/MeshNavigation )
add_subdirectory( Source/XML )


# install KGeoBag as a Kasper module
kasper_install_module()
