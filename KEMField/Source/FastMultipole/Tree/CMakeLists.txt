# CMakeLists for KEMField/FastMultipole/Tree
# Author: J. P. Barrett

set(KEMField_MULTIPOLE_BUFFER "32" CACHE STRING "Buffer size (MB) used for batched calculation of multipole moments.")
mark_as_advanced(FORCE KEMField_MULTIPOLE_BUFFER)
add_cflag(KEMFIELD_MULTIPOLE_BUFFER_SIZE_MB=${KEMField_MULTIPOLE_BUFFER})

kasper_internal_include_directories(${CMAKE_CURRENT_SOURCE_DIR}/include)

set (TREE_HEADERFILES
    ${CMAKE_CURRENT_SOURCE_DIR}/include/KFMSubdivisionStrategy.hh
    ${CMAKE_CURRENT_SOURCE_DIR}/include/KFMCubicSpaceTree.hh
    ${CMAKE_CURRENT_SOURCE_DIR}/include/KFMCubicSpaceTreeProperties.hh
    ${CMAKE_CURRENT_SOURCE_DIR}/include/KFMCubicSpaceNodeNeighborFinder.hh
    ${CMAKE_CURRENT_SOURCE_DIR}/include/KFMCubicSpaceBallSorter.hh
    ${CMAKE_CURRENT_SOURCE_DIR}/include/KFMInsertionCondition.hh
    ${CMAKE_CURRENT_SOURCE_DIR}/include/KFMSubdivisionCondition.hh
    ${CMAKE_CURRENT_SOURCE_DIR}/include/KFMSubdivisionConditionAggressive.hh
    ${CMAKE_CURRENT_SOURCE_DIR}/include/KFMSubdivisionConditionBalanced.hh
    ${CMAKE_CURRENT_SOURCE_DIR}/include/KFMSubdivisionConditionGuided.hh
    ${CMAKE_CURRENT_SOURCE_DIR}/include/KFMCubicSpaceNodeProgenitor.hh
    ${CMAKE_CURRENT_SOURCE_DIR}/include/KFMElementNodeAssociator.hh
    ${CMAKE_CURRENT_SOURCE_DIR}/include/KFMNearbyElementCounter.hh
    ${CMAKE_CURRENT_SOURCE_DIR}/include/KFMElementMomentBatchCalculator.hh
    ${CMAKE_CURRENT_SOURCE_DIR}/include/KFMElementScalarMomentDistributor.hh
    ${CMAKE_CURRENT_SOURCE_DIR}/include/KFMScalarMomentInitializer.hh
    ${CMAKE_CURRENT_SOURCE_DIR}/include/KFMScalarMomentResetter.hh
    ${CMAKE_CURRENT_SOURCE_DIR}/include/KFMScalarMomentCollector.hh
    ${CMAKE_CURRENT_SOURCE_DIR}/include/KFMReducedScalarMomentCollector.hh
    ${CMAKE_CURRENT_SOURCE_DIR}/include/KFMScalarMomentDistributor.hh
    ${CMAKE_CURRENT_SOURCE_DIR}/include/KFMScalarMomentRemoteToLocalConverter.hh
    ${CMAKE_CURRENT_SOURCE_DIR}/include/KFMReducedScalarMomentRemoteToLocalConverter.hh
    ${CMAKE_CURRENT_SOURCE_DIR}/include/KFMScalarMomentRemoteToRemoteConverter.hh
    ${CMAKE_CURRENT_SOURCE_DIR}/include/KFMScalarMomentLocalToLocalConverter.hh
    ${CMAKE_CURRENT_SOURCE_DIR}/include/KFMRemoteToLocalConverterInterface.hh
    ${CMAKE_CURRENT_SOURCE_DIR}/include/KFMCubicSpaceTreeNavigator.hh
    ${CMAKE_CURRENT_SOURCE_DIR}/include/KFMCubicSpaceNodeAdjacencyProgenitor.hh
    ${CMAKE_CURRENT_SOURCE_DIR}/include/KFMIdentitySetMerger.hh
    ${CMAKE_CURRENT_SOURCE_DIR}/include/KFMIdentitySetCollector.hh
    ${CMAKE_CURRENT_SOURCE_DIR}/include/KFMIdentitySetListCreator.hh
    ${CMAKE_CURRENT_SOURCE_DIR}/include/KFMCollocationPointIdentitySetCreator.hh
    ${CMAKE_CURRENT_SOURCE_DIR}/include/KFMDenseBlockSparseMatrixGenerator.hh
    ${CMAKE_CURRENT_SOURCE_DIR}/include/KFMDenseBlockSparseMatrixStructureGenerator.hh
    ${CMAKE_CURRENT_SOURCE_DIR}/include/KFMElementLocator.hh
    ${CMAKE_CURRENT_SOURCE_DIR}/include/KFMNodeCollector.hh
    ${CMAKE_CURRENT_SOURCE_DIR}/include/KFMSpecialNodeSet.hh
    ${CMAKE_CURRENT_SOURCE_DIR}/include/KFMSpecialNodeSetCreator.hh
    ${CMAKE_CURRENT_SOURCE_DIR}/include/KFMDirectCallCounter.hh
    ${CMAKE_CURRENT_SOURCE_DIR}/include/KFMWorkLoadBalanceWeights.hh
)

if (${PROJECT_NAME}_USE_MPI)
  list (APPEND TREE_HEADERFILES
    ${CMAKE_CURRENT_SOURCE_DIR}/include/KFMCubicSpaceTreeStaticLoadBalancer.hh
    )
endif (${PROJECT_NAME}_USE_MPI)


set (TREE_SOURCEFILES
    ${CMAKE_CURRENT_SOURCE_DIR}/src/KFMSubdivisionStrategy.cc
    ${CMAKE_CURRENT_SOURCE_DIR}/src/KFMElementMomentBatchCalculator.cc
)

##################################################

add_library (KFMTree SHARED ${TREE_SOURCEFILES} ${TREE_HEADERFILES})
target_link_libraries (KFMTree KFMCore KFMMath KFMMathUtilities KFMKernel KEMStructuredASCII)

kasper_install_headers (${TREE_HEADERFILES})
kasper_install_libraries (KFMTree)
