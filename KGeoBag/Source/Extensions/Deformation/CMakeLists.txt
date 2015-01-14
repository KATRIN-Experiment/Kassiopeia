# headers
set( DEFORMATION_HEADER_FILES
    Include/KGDeformation.hh
    Include/KGAffineDeformation.hh
    Include/KGRadialDeformation.hh
    Include/KGDeformed.hh
    Include/KGMeshDeformer.hh
)

# sources
set( DEFORMATION_SOURCE_FILES
    Source/KGAffineDeformation.cc
    Source/KGRadialDeformation.cc
    Source/KGDeformed.cc
    Source/KGMeshDeformer.cc
)

# internal
kasper_internal_include_directories(
	${CMAKE_CURRENT_SOURCE_DIR}/Include
)

# target
add_library( KGeoBagDeformation SHARED
	${DEFORMATION_SOURCE_FILES}
)
target_link_libraries( KGeoBagDeformation
    KGeoBagMath
    KGeoBagCore
    KGeoBagShapes
    KGeoBagMesh
    ${Kommon_LIBRARIES}
)

# install
kasper_install_headers( ${DEFORMATION_HEADER_FILES} )
kasper_install_libraries( KGeoBagDeformation )


