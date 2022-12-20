# headers
set( REFINEMENT_HEADER_FILES
    Include/KGMeshRefiner.hh
)

# sources
set( REFINEMENT_SOURCE_FILES
    Source/KGMeshRefiner.cc
)

# target
add_library( KGeoBagMeshRefinement SHARED
    ${REFINEMENT_SOURCE_FILES} ${REFINEMENT_HEADER_FILES})
target_include_directories( KGeoBagMeshRefinement
    PUBLIC $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/Include> $<INSTALL_INTERFACE:include>)
target_link_libraries( KGeoBagMeshRefinement
    PUBLIC
        KGeoBagShapes
        KGeoBagMesh
)

# install
kasper_install_headers( ${REFINEMENT_HEADER_FILES} )
kasper_install_libraries( KGeoBagMeshRefinement )
