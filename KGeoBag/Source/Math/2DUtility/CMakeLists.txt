# header
set( 2DUTILITY_HEADER_FILES
    Include/KG2DArc.hh
    Include/KG2DArea.hh
    Include/KG2DLineSegment.hh
    Include/KG2DPolyLine.hh
    Include/KG2DPolyLineWithArcs.hh
    Include/KG2DPolygon.hh
    Include/KG2DPolygonWithArcs.hh
    Include/KG2DShape.hh
    Include/KGInfinitePlane.hh
    Include/KGVertexSideDescriptor.hh
)

# sources
set( 2DUTILITY_SOURCE_FILES
    Source/KG2DArc.cc
    Source/KG2DLineSegment.cc
    Source/KG2DPolyLine.cc
    Source/KG2DPolyLineWithArcs.cc
    Source/KG2DPolygon.cc
    Source/KG2DPolygonWithArcs.cc
    Source/KGInfinitePlane.cc
)

# target
add_library( KGeoBag2DUtility SHARED
    ${2DUTILITY_SOURCE_FILES} ${2DUTILITY_HEADER_FILES})
target_include_directories( KGeoBag2DUtility
    PUBLIC $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/Include> $<INSTALL_INTERFACE:include>)
target_link_libraries( KGeoBag2DUtility
    PUBLIC
       KGeoBagMath
)

# install
kasper_install_headers( ${2DUTILITY_HEADER_FILES} )
kasper_install_libraries( KGeoBag2DUtility )
