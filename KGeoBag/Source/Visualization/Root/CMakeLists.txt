
# headers
set( ROOT_VISUALIZATION_HEADER_FILES
	Include/KGROOTGeometryPainter.hh
)

# source
set( ROOT_VISUALIZATION_SOURCE_FILES
    Source/KGROOTGeometryPainter.cc
)

# external
find_package( ROOT 5.24.0 REQUIRED )
kasper_external_include_directories(
	${ROOT_INCLUDE_DIR}
)

# internal
kasper_internal_include_directories(
	${CMAKE_CURRENT_SOURCE_DIR}/Include
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

