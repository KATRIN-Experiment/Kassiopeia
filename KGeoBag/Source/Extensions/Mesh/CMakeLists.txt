# headers
set( MESH_HEADER_FILES
    Include/KGMeshMessage.hh

    Include/KGMeshElement.hh
    Include/KGMeshRectangle.hh
    Include/KGMeshTriangle.hh
    Include/KGMeshWire.hh

    Include/KGMesh.hh
    Include/KGMesherBase.hh
    Include/KGMesher.hh

    External/Include/KGStlFileSurfaceMesher.hh
    External/Include/KGStlFileSpaceMesher.hh

    Complex/Include/KGComplexMesher.hh
    Complex/Include/KGBoxMesher.hh
    Complex/Include/KGCylinderMesher.hh
    Complex/Include/KGRotatedSurfaceMesher.hh
    Complex/Include/KGExtrudedSurfaceMesher.hh
    Complex/Include/KGConicalWireArrayMesher.hh
    Complex/Include/KGPortHousingSurfaceMesher.hh
    Complex/Include/KGConicSectPortHousingSurfaceMesher.hh
    Complex/Include/KGRodSurfaceMesher.hh
    Complex/Include/KGBeamSurfaceMesher.hh
    Complex/Include/KGLinearWireGridMesher.hh
    Complex/Include/KGQuadraticWireGridMesher.hh
    Complex/Include/KGCircleWireMesher.hh
    Complex/Include/KGCircularWirePinsMesher.hh

    Simple/Include/KGSimpleMesher.hh
    Simple/Include/KGFlattenedCircleSurfaceMesher.hh
    Simple/Include/KGFlattenedCircleSurfaceMesher.hh
    Simple/Include/KGFlattenedPolyLoopSurfaceMesher.hh
    Simple/Include/KGRotatedLineSegmentSurfaceMesher.hh
    Simple/Include/KGRotatedArcSegmentSurfaceMesher.hh
    Simple/Include/KGRotatedPolyLineSurfaceMesher.hh
    Simple/Include/KGRotatedCircleSurfaceMesher.hh
    Simple/Include/KGRotatedPolyLoopSurfaceMesher.hh
    Simple/Include/KGShellLineSegmentSurfaceMesher.hh
    Simple/Include/KGShellArcSegmentSurfaceMesher.hh
    Simple/Include/KGShellPolyLineSurfaceMesher.hh
    Simple/Include/KGShellPolyLoopSurfaceMesher.hh
    Simple/Include/KGShellCircleSurfaceMesher.hh
    Simple/Include/KGExtrudedLineSegmentSurfaceMesher.hh
    Simple/Include/KGExtrudedArcSegmentSurfaceMesher.hh
    Simple/Include/KGExtrudedPolyLineSurfaceMesher.hh
    Simple/Include/KGExtrudedCircleSurfaceMesher.hh
    Simple/Include/KGExtrudedPolyLoopSurfaceMesher.hh
    Simple/Include/KGRotatedLineSegmentSpaceMesher.hh
    Simple/Include/KGRotatedArcSegmentSpaceMesher.hh
    Simple/Include/KGRotatedPolyLineSpaceMesher.hh
    Simple/Include/KGRotatedCircleSpaceMesher.hh
    Simple/Include/KGRotatedPolyLoopSpaceMesher.hh
    Simple/Include/KGExtrudedCircleSpaceMesher.hh
    Simple/Include/KGExtrudedPolyLoopSpaceMesher.hh

    Navigation/Include/KGNavigableMeshElement.hh
    Navigation/Include/KGMeshNavigationNode.hh
    Navigation/Include/KGMeshElementCollector.hh
    Navigation/Include/KGNavigableMeshElementContainer.hh
    Navigation/Include/KGInsertionCondition.hh
    Navigation/Include/KGNavigableMeshElementSorter.hh
    Navigation/Include/KGSubdivisionCondition.hh
    Navigation/Include/KGNavigableMeshTree.hh
    Navigation/Include/KGNavigableMeshTreeBuilder.hh
    Navigation/Include/KGNavigableMeshTreeInformationExtractor.hh
    Navigation/Include/KGNavigableMeshFirstIntersectionFinder.hh
    Navigation/Include/KGNavigableMeshIntersectionFinder.hh
    Navigation/Include/KGNavigableMeshProximityCheck.hh
)

# sources
set( MESH_SOURCE_FILES
    Source/KGMeshMessage.cc

    Source/KGMeshElement.cc
    Source/KGMeshRectangle.cc
    Source/KGMeshTriangle.cc
    Source/KGMeshWire.cc

    Source/KGMesh.cc
    Source/KGMesherBase.cc
    Source/KGMesher.cc

    External/Source/KGStlFileSurfaceMesher.cc
    External/Source/KGStlFileSpaceMesher.cc

    Complex/Source/KGComplexMesher.cc
    Complex/Source/KGBoxMesher.cc
    Complex/Source/KGCylinderMesher.cc
    Complex/Source/KGRotatedSurfaceMesher.cc
    Complex/Source/KGExtrudedSurfaceMesher.cc
    Complex/Source/KGPortHousingSurfaceMesher.cc
    Complex/Source/KGConicalWireArrayMesher.cc
    Complex/Source/KGConicSectPortHousingSurfaceMesher.cc
    Complex/Source/KGRodSurfaceMesher.cc
    Complex/Source/KGBeamSurfaceMesher.cc
    Complex/Source/KGLinearWireGridMesher.cc
    Complex/Source/KGQuadraticWireGridMesher.cc
    Complex/Source/KGCircleWireMesher.cc
    Complex/Source/KGCircularWirePinsMesher.cc

    Simple/Source/KGSimpleMesher.cc
    Simple/Source/KGFlattenedCircleSurfaceMesher.cc
    Simple/Source/KGFlattenedCircleSurfaceMesher.cc
    Simple/Source/KGFlattenedPolyLoopSurfaceMesher.cc
    Simple/Source/KGRotatedLineSegmentSurfaceMesher.cc
    Simple/Source/KGRotatedArcSegmentSurfaceMesher.cc
    Simple/Source/KGRotatedPolyLineSurfaceMesher.cc
    Simple/Source/KGRotatedCircleSurfaceMesher.cc
    Simple/Source/KGRotatedPolyLoopSurfaceMesher.cc
    Simple/Source/KGShellLineSegmentSurfaceMesher.cc
    Simple/Source/KGShellArcSegmentSurfaceMesher.cc
    Simple/Source/KGShellPolyLineSurfaceMesher.cc
    Simple/Source/KGShellPolyLoopSurfaceMesher.cc
    Simple/Source/KGShellCircleSurfaceMesher.cc
    Simple/Source/KGExtrudedLineSegmentSurfaceMesher.cc
    Simple/Source/KGExtrudedArcSegmentSurfaceMesher.cc
    Simple/Source/KGExtrudedPolyLineSurfaceMesher.cc
    Simple/Source/KGExtrudedCircleSurfaceMesher.cc
    Simple/Source/KGExtrudedPolyLoopSurfaceMesher.cc
    Simple/Source/KGRotatedLineSegmentSpaceMesher.cc
    Simple/Source/KGRotatedArcSegmentSpaceMesher.cc
    Simple/Source/KGRotatedPolyLineSpaceMesher.cc
    Simple/Source/KGRotatedCircleSpaceMesher.cc
    Simple/Source/KGRotatedPolyLoopSpaceMesher.cc
    Simple/Source/KGExtrudedCircleSpaceMesher.cc
    Simple/Source/KGExtrudedPolyLoopSpaceMesher.cc

    Navigation/Source/KGNavigableMeshElement.cc
    Navigation/Source/KGMeshElementCollector.cc
    Navigation/Source/KGNavigableMeshElementContainer.cc
    Navigation/Source/KGInsertionCondition.cc
    Navigation/Source/KGNavigableMeshElementSorter.cc
    Navigation/Source/KGSubdivisionCondition.cc
    Navigation/Source/KGNavigableMeshTreeBuilder.cc
    Navigation/Source/KGNavigableMeshFirstIntersectionFinder.cc
    Navigation/Source/KGNavigableMeshIntersectionFinder.cc
    Navigation/Source/KGNavigableMeshProximityCheck.cc
)

# target
add_library( KGeoBagMesh SHARED
    ${MESH_SOURCE_FILES} ${MESH_HEADER_FILES}
)
target_include_directories( KGeoBagMesh
    PUBLIC
        $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/Include>
        $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/External/Include>
        $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/Complex/Include>
        $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/Simple/Include>
        $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/Navigation/Include>
        $<INSTALL_INTERFACE:include>
)
target_link_libraries( KGeoBagMesh
    PUBLIC
        KGeoBagShapes
        KGeoBagMathSpaceTree
)

# install
kasper_install_headers( ${MESH_HEADER_FILES} )
kasper_install_libraries( KGeoBagMesh )
