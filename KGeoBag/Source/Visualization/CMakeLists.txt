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


# internal
kasper_internal_include_directories(
	${CMAKE_CURRENT_SOURCE_DIR}/Basic/Include
)

# target
add_library( KGeoBagBasicVisualization SHARED ${BASIC_VISUALIZATION_SOURCE_FILES} ${BASIC_VISUALIZATION_HEADER_FILES} )
target_link_libraries( KGeoBagBasicVisualization
	${Kommon_LIBRARIES}
	KGeoBagCore
	KGeoBagShapes
	KGeoBagAppearance
	KGeoBagMesh
	KGeoBagAxialMesh
)

# install
kasper_install_headers( ${BASIC_VISUALIZATION_HEADER_FILES} )
kasper_install_libraries( KGeoBagBasicVisualization )


#VTK Stuff
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

	# external
	kasper_find_vtk()
	if( VTK_MAJOR_VERSION GREATER 5 )
		set_property( SOURCE ${VTK_VISUALIZATION_SOURCE_FILES} ${VTK_VISUALIZATION_HEADER_FILES} APPEND PROPERTY COMPILE_DEFINITIONS VTK6 )
	endif()

	# internal
	kasper_internal_include_directories(
		${CMAKE_CURRENT_SOURCE_DIR}/Vtk/Include
	)

	# target
	add_library( KGeoBagVtkVisualization SHARED ${VTK_VISUALIZATION_SOURCE_FILES} ${VTK_VISUALIZATION_HEADER_FILES} )
	target_link_libraries( KGeoBagVtkVisualization
		KGeoBagBasicVisualization
		KGeoBagCore
		KGeoBagShapes
		KGeoBagAppearance
		KGeoBagMesh
		KGeoBagAxialMesh
        ${Kommon_LIBRARIES}
        ${Kommon_Vtk_LIBRARIES}
        ${VTK_LIBRARIES}
	)

	# install
	kasper_install_headers( ${VTK_VISUALIZATION_HEADER_FILES} )
	kasper_install_libraries( KGeoBagVtkVisualization )

endif( KGeoBag_USE_VTK )

#ROOT Stuff
if( KGeoBag_USE_ROOT )

	# headers
	set( ROOT_VISUALIZATION_HEADER_FILES
		Root/Include/KGROOTGeometryPainter.hh
	)

	# source
	set( ROOT_VISUALIZATION_SOURCE_FILES
	    Root/Source/KGROOTGeometryPainter.cc
	)

	# external
#	find_package( ROOT 5.24.0 REQUIRED )
	kasper_external_include_directories(
		${ROOT_INCLUDE_DIR}
	)

	# internal
	kasper_internal_include_directories(
		${CMAKE_CURRENT_SOURCE_DIR}/Root/Include
	)

	# target
	add_library( KGeoBagROOTVisualization SHARED ${ROOT_VISUALIZATION_SOURCE_FILES} ${ROOT_VISUALIZATION_HEADER_FILES} )
	target_link_libraries( KGeoBagROOTVisualization
		${ROOT_LIBRARIES}
		${Kommon_LIBRARIES}
		KGeoBagBasicVisualization
		KGeoBagCore
		KGeoBagShapes
		KGeoBagAppearance
		KGeoBagMesh
		KGeoBagAxialMesh
	)

	# install
	kasper_install_headers( ${ROOT_VISUALIZATION_HEADER_FILES} )
	kasper_install_libraries( KGeoBagROOTVisualization )

endif( KGeoBag_USE_ROOT )
