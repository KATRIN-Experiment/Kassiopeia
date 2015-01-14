
# headers
set( BASIC_VISUALIZATION_HEADER_FILES
    Include/KGVisualizationMessage.hh
)

# source
set( BASIC_VISUALIZATION_SOURCE_FILES
    Source/KGVisualizationMessage.cc
)


# internal
kasper_internal_include_directories(
	${CMAKE_CURRENT_SOURCE_DIR}/Include
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
    

