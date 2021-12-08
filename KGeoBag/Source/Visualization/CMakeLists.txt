#basic visual

# headers
set( BASIC_VISUALIZATION_HEADER_FILES
    Basic/Include/KGVisualizationMessage.hh
    Basic/Include/KGGeometryPrinter.hh
)

# source
set( BASIC_VISUALIZATION_SOURCE_FILES
    Basic/Source/KGVisualizationMessage.cc
    Basic/Source/KGGeometryPrinter.cc
)

# target
add_library( KGeoBagBasicVisualization
    SHARED ${BASIC_VISUALIZATION_SOURCE_FILES} ${BASIC_VISUALIZATION_HEADER_FILES}
)
target_include_directories( KGeoBagBasicVisualization
    PUBLIC $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/Basic/Include> $<INSTALL_INTERFACE:include>
)
target_link_libraries( KGeoBagBasicVisualization
    PUBLIC
        KGeoBagAppearance
        KGeoBagMesh
        KGeoBagAxialMesh
)


# install
kasper_install_headers( ${BASIC_VISUALIZATION_HEADER_FILES} )
kasper_install_libraries( KGeoBagBasicVisualization )

#######
# VTK #
#######

if( KGeoBag_USE_VTK )
    # headers
    set( VTK_VISUALIZATION_HEADER_FILES
        Vtk/Include/KGVTKGeometryPainter.hh
        Vtk/Include/KGVTKMeshPainter.hh
        Vtk/Include/KGVTKAxialMeshPainter.hh
        Vtk/Include/KGVTKPointTester.hh
        Vtk/Include/KGVTKNormalTester.hh
        Vtk/Include/KGVTKDistanceTester.hh
        Vtk/Include/KGVTKOutsideTester.hh
        Vtk/Include/KGVTKRandomPointTester.hh
        Vtk/Include/KGVTKMeshIntersectionTester.hh
    )

    # source
    set( VTK_VISUALIZATION_SOURCE_FILES
        Vtk/Source/KGVTKGeometryPainter.cc
        Vtk/Source/KGVTKMeshPainter.cc
        Vtk/Source/KGVTKAxialMeshPainter.cc
        Vtk/Source/KGVTKPointTester.cc
        Vtk/Source/KGVTKNormalTester.cc
        Vtk/Source/KGVTKDistanceTester.cc
        Vtk/Source/KGVTKOutsideTester.cc
        Vtk/Source/KGVTKRandomPointTester.cc
        Vtk/Source/KGVTKMeshIntersectionTester.cc
    )

    # target
    add_library( KGeoBagVtkVisualization SHARED
        ${VTK_VISUALIZATION_SOURCE_FILES} ${VTK_VISUALIZATION_HEADER_FILES}
    )
    target_include_directories( KGeoBagVtkVisualization
        PUBLIC $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/Vtk/Include> $<INSTALL_INTERFACE:include>
    )
    target_link_libraries( KGeoBagVtkVisualization
        PUBLIC
            KommonVtk
            KGeoBagBasicVisualization
    )

    target_compile_definitions( KGeoBagVtkVisualization PUBLIC KGeoBag_USE_VTK )
    if( VTK_MAJOR_VERSION GREATER 5 )
        target_compile_definitions( KGeoBagVtkVisualization PRIVATE VTK6 )
    endif()

    # install
    kasper_install_headers( ${VTK_VISUALIZATION_HEADER_FILES} )
    kasper_install_libraries( KGeoBagVtkVisualization )

endif( KGeoBag_USE_VTK )

########
# ROOT #
########

if( KGeoBag_USE_ROOT )

    # headers
    set( ROOT_VISUALIZATION_HEADER_FILES
        Root/Include/KGROOTGeometryPainter.hh
    )

    # source
    set( ROOT_VISUALIZATION_SOURCE_FILES
        Root/Source/KGROOTGeometryPainter.cc
    )

    # target
    add_library( KGeoBagROOTVisualization SHARED
        ${ROOT_VISUALIZATION_SOURCE_FILES} ${ROOT_VISUALIZATION_HEADER_FILES}
    )
    target_include_directories( KGeoBagROOTVisualization
        PUBLIC $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/Root/Include> $<INSTALL_INTERFACE:include>
    )
    target_link_libraries( KGeoBagROOTVisualization
        PUBLIC
            KommonRoot
            KGeoBagBasicVisualization
            ROOT::Core
    )

    target_compile_definitions( KGeoBagROOTVisualization PUBLIC KGeoBag_USE_ROOT )

    # install
    kasper_install_headers( ${ROOT_VISUALIZATION_HEADER_FILES} )
    kasper_install_libraries( KGeoBagROOTVisualization )

endif( KGeoBag_USE_ROOT )
