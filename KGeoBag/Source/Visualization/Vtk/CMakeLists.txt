
# headers
set( VTK_VISUALIZATION_HEADER_FILES
    Include/KGVTKGeometryPainter.hh
    Include/KGVTKMeshPainter.hh
    Include/KGVTKAxialMeshPainter.hh
    Include/KGVTKPointTester.hh
    Include/KGVTKNormalTester.hh
    Include/KGVTKDistanceTester.hh
    Include/KGVTKOutsideTester.hh
    Include/KGVTKRandomPointTester.hh
)

# source
set( VTK_VISUALIZATION_SOURCE_FILES
    Source/KGVTKGeometryPainter.cc
    Source/KGVTKMeshPainter.cc
    Source/KGVTKAxialMeshPainter.cc
    Source/KGVTKPointTester.cc
    Source/KGVTKNormalTester.cc
    Source/KGVTKDistanceTester.cc
    Source/KGVTKOutsideTester.cc
    Source/KGVTKRandomPointTester.cc
)

if( VTK_MAJOR_VERSION GREATER 5 )
	set_property( SOURCE ${VTK_VISUALIZATION_SOURCE_FILES} APPEND PROPERTY COMPILE_DEFINITIONS VTK6 )
endif()

# internal
kasper_internal_include_directories(
	${CMAKE_CURRENT_SOURCE_DIR}/Include
)

# target
add_library( KGeoBagVtkVisualization SHARED ${VTK_VISUALIZATION_SOURCE_FILES} ${VTK_VISUALIZATION_HEADER_FILES} )
target_link_libraries( KGeoBagVtkVisualization
	${VTK_LIBRARIES}
	${Kommon_LIBRARIES}
	KGeoBagBasicVisualization
	KGeoBagCore
	KGeoBagShapes
	KGeoBagAppearance
	KGeoBagMesh
	KGeoBagAxialMesh
)

# install
kasper_install_headers( ${VTK_VISUALIZATION_HEADER_FILES} )
kasper_install_libraries( KGeoBagVtkVisualization )
    

