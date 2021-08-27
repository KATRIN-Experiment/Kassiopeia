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

# target
add_library( KGeoBagDeformation SHARED
    ${DEFORMATION_SOURCE_FILES} ${DEFORMATION_HEADER_FILES})
target_include_directories( KGeoBagDeformation
    PUBLIC $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/Include> $<INSTALL_INTERFACE:include>)
target_link_libraries( KGeoBagDeformation
    PUBLIC
        KGeoBagShapes
        KGeoBagMesh
)

# install
kasper_install_headers( ${DEFORMATION_HEADER_FILES} )
kasper_install_libraries( KGeoBagDeformation )
