# headers
set( MATH_SPACETREE_HEADER_FILES
    Include/KGPoint.hh
    Include/KGBall.hh
    Include/KGAxisAlignedBox.hh
    Include/KGCube.hh
    Include/KGPointCloud.hh
    Include/KGBallCloud.hh
    Include/KGAxisAlignedBoxSupportSet.hh
    Include/KGBallSupportSet.hh
    Include/KGPointCloudToBoundingBallConverter.hh
    Include/KGBoundaryCalculator.hh
    Include/KGNode.hh
    Include/KGObjectHolder.hh
    Include/KGObjectCollection.hh
    Include/KGObjectRetriever.hh
    Include/KGNodeActor.hh
    Include/KGRecursiveActor.hh
    Include/KGCorecursiveActor.hh
    Include/KGCompoundActor.hh
    Include/KGInspectingActor.hh
    Include/KGConditionalActor.hh
    Include/KGTree.hh
    Include/KGObjectContainer.hh
    Include/KGIdentitySet.hh
    Include/KGNodeObjectRemover.hh
    Include/KGNodeObjectNullifier.hh
    Include/KGIdentitySetSorter.hh
)

# source
set( MATH_SPACETREE_SOURCE_FILES
    Source/KGIdentitySet.cc
)

# internal
kasper_internal_include_directories(
	${CMAKE_CURRENT_SOURCE_DIR}/Include
)

# target
add_library( KGeoBagMathSpaceTree SHARED
	${MATH_SPACETREE_SOURCE_FILES}
)
target_link_libraries( KGeoBagMathSpaceTree
    ${Kommon_LIBRARIES}
    ${GSL_LIBRARIES}
)

# install
kasper_install_headers( ${MATH_SPACETREE_HEADER_FILES} )
kasper_install_libraries( KGeoBagMathSpaceTree )
