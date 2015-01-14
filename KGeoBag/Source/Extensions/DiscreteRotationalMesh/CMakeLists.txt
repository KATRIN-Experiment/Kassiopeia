# headers
set( DISCRETEROTATIONALMESH_HEADER_FILES
	Include/KGDiscreteRotationalMeshMessage.hh
	Include/KGDiscreteRotationalMeshElement.hh
	Include/KGDiscreteRotationalMesh.hh
	Include/KGDiscreteRotationalMesherBase.hh
	Include/KGDiscreteRotationalMesher.hh

	Complex/Include/KGConicalWireArrayDiscreteRotationalMesher.hh
)

# sources
set( DISCRETEROTATIONALMESH_SOURCE_FILES
	Source/KGDiscreteRotationalMeshMessage.cc
	Source/KGDiscreteRotationalMeshElement.cc
	Source/KGDiscreteRotationalMesh.cc
	Source/KGDiscreteRotationalMesherBase.cc
	Source/KGDiscreteRotationalMesher.cc

	Complex/Source/KGConicalWireArrayDiscreteRotationalMesher.cc
)

# internal
kasper_internal_include_directories(
	${CMAKE_CURRENT_SOURCE_DIR}/Include
	${CMAKE_CURRENT_SOURCE_DIR}/Complex/Include
)

# target
add_library( KGeoBagDiscreteRotationalMesh SHARED ${DISCRETEROTATIONALMESH_SOURCE_FILES} ${DISCRETEROTATIONALMESH_HEADER_FILES} )
target_link_libraries( KGeoBagDiscreteRotationalMesh
    ${Kommon_LIBRARIES}
    KGeoBagCore
    KGeoBagShapes
    KGeoBagMesh
    KGeoBagAxialMesh
)

# install
kasper_install_headers( ${DISCRETEROTATIONALMESH_HEADER_FILES} )
kasper_install_libraries( KGeoBagDiscreteRotationalMesh )


