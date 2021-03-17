# CMakeLists for KEMField/FastMultipole/Core
# Author: J. P. Barrett

kasper_internal_include_directories(${CMAKE_CURRENT_SOURCE_DIR}/include)

set (KFMCORE_HEADERFILES
    ${CMAKE_CURRENT_SOURCE_DIR}/include/KFMObjectHolder.hh
    ${CMAKE_CURRENT_SOURCE_DIR}/include/KFMObjectCollection.hh
    ${CMAKE_CURRENT_SOURCE_DIR}/include/KFMNode.hh
    ${CMAKE_CURRENT_SOURCE_DIR}/include/KFMNodeData.hh
    ${CMAKE_CURRENT_SOURCE_DIR}/include/KFMObjectRetriever.hh
    ${CMAKE_CURRENT_SOURCE_DIR}/include/KFMNodeActor.hh
    ${CMAKE_CURRENT_SOURCE_DIR}/include/KFMNodeFinder.hh
    ${CMAKE_CURRENT_SOURCE_DIR}/include/KFMRecursiveActor.hh
    ${CMAKE_CURRENT_SOURCE_DIR}/include/KFMCorecursiveActor.hh
    ${CMAKE_CURRENT_SOURCE_DIR}/include/KFMCompoundActor.hh
    ${CMAKE_CURRENT_SOURCE_DIR}/include/KFMInspectingActor.hh
    ${CMAKE_CURRENT_SOURCE_DIR}/include/KFMCompoundInspectingActor.hh
    ${CMAKE_CURRENT_SOURCE_DIR}/include/KFMConditionalActor.hh
    ${CMAKE_CURRENT_SOURCE_DIR}/include/KFMTree.hh
    ${CMAKE_CURRENT_SOURCE_DIR}/include/KFMObjectContainer.hh
    ${CMAKE_CURRENT_SOURCE_DIR}/include/KFMIdentityPair.hh
    ${CMAKE_CURRENT_SOURCE_DIR}/include/KFMBasisData.hh
    ${CMAKE_CURRENT_SOURCE_DIR}/include/KFMIdentitySet.hh
    ${CMAKE_CURRENT_SOURCE_DIR}/include/KFMIdentitySetList.hh
    ${CMAKE_CURRENT_SOURCE_DIR}/include/KFMElementLocalInfluenceRange.hh
    ${CMAKE_CURRENT_SOURCE_DIR}/include/KFMNodeObjectRemover.hh
    ${CMAKE_CURRENT_SOURCE_DIR}/include/KFMNodeObjectNullifier.hh
    ${CMAKE_CURRENT_SOURCE_DIR}/include/KFMIdentitySetSorter.hh
    ${CMAKE_CURRENT_SOURCE_DIR}/include/KFMCollocationPointIdentitySet.hh
    ${CMAKE_CURRENT_SOURCE_DIR}/include/KFMTreeStructureExtractor.hh
    ${CMAKE_CURRENT_SOURCE_DIR}/include/KFMNodeIdentityListRange.hh
    ${CMAKE_CURRENT_SOURCE_DIR}/include/KFMNodeIdentityListCreator.hh
    ${CMAKE_CURRENT_SOURCE_DIR}/include/KFMNodeIdentityListRangeAssociator.hh
    ${CMAKE_CURRENT_SOURCE_DIR}/include/KFMNodeFlags.hh
    ${CMAKE_CURRENT_SOURCE_DIR}/include/KFMNodeFlagInitializer.hh
    ${CMAKE_CURRENT_SOURCE_DIR}/include/KFMNodeFlagValueInitializer.hh
    ${CMAKE_CURRENT_SOURCE_DIR}/include/KFMNodeFlagValueInspector.hh
    ${CMAKE_CURRENT_SOURCE_DIR}/include/KFMNonEmptyIdentitySetActor.hh
    ${CMAKE_CURRENT_SOURCE_DIR}/include/KFMNodeChildToParentFlagValueInitializer.hh
    ${CMAKE_CURRENT_SOURCE_DIR}/include/KFMNodeParentToChildFlagValueInspector.hh
    ${CMAKE_CURRENT_SOURCE_DIR}/include/KFMLeafConditionActor.hh
    ${CMAKE_CURRENT_SOURCE_DIR}/include/KFMLevelConditionActor.hh
)

set (KFMCORE_SOURCEFILES
    ${CMAKE_CURRENT_SOURCE_DIR}/src/KFMIdentitySet.cc
    ${CMAKE_CURRENT_SOURCE_DIR}/src/KFMIdentitySetList.cc
    ${CMAKE_CURRENT_SOURCE_DIR}/src/KFMElementLocalInfluenceRange.cc
)

##################################################

add_library (KFMCore SHARED ${KFMCORE_SOURCEFILES} ${KFMCORE_HEADERFILES})
target_link_libraries (KFMCore KEMCore)

kasper_install_headers (${KFMCORE_HEADERFILES})
kasper_install_libraries (KFMCore)
