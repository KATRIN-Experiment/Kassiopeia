# CMakeLists for KEMField/FastMultipole/Electrostatics
# Author: J. P. Barrett

set (KFMELECTROSTATICS_HEADERFILES
    ${CMAKE_CURRENT_SOURCE_DIR}/include/KFMElectrostaticParameters.hh
    ${CMAKE_CURRENT_SOURCE_DIR}/include/KFMElectrostaticElement.hh
    ${CMAKE_CURRENT_SOURCE_DIR}/include/KFMElectrostaticNode.hh
    ${CMAKE_CURRENT_SOURCE_DIR}/include/KFMElectrostaticTree.hh
    ${CMAKE_CURRENT_SOURCE_DIR}/include/KFMElectrostaticElementContainerBase.hh
    ${CMAKE_CURRENT_SOURCE_DIR}/include/KFMElectrostaticElementContainer.hh
    ${CMAKE_CURRENT_SOURCE_DIR}/include/KFMElectrostaticMultipoleSet.hh
    ${CMAKE_CURRENT_SOURCE_DIR}/include/KFMElectrostaticLocalCoefficientSet.hh
    ${CMAKE_CURRENT_SOURCE_DIR}/include/KFMElectrostaticMultipoleCalculator.hh
    ${CMAKE_CURRENT_SOURCE_DIR}/include/KFMElectrostaticMultipoleCalculatorAnalytic.hh
    ${CMAKE_CURRENT_SOURCE_DIR}/include/KFMElectrostaticMultipoleCalculatorNumeric.hh
    ${CMAKE_CURRENT_SOURCE_DIR}/include/KFMElectrostaticLocalCoefficientCalculatorNumeric.hh
    ${CMAKE_CURRENT_SOURCE_DIR}/include/KFMElectrostaticMultipoleBatchCalculatorBase.hh
    ${CMAKE_CURRENT_SOURCE_DIR}/include/KFMElectrostaticMultipoleBatchCalculator.hh
    ${CMAKE_CURRENT_SOURCE_DIR}/include/KFMElectrostaticRegionSizeEstimator.hh
    ${CMAKE_CURRENT_SOURCE_DIR}/include/KFMElectrostaticTreeBuilder.hh
    ${CMAKE_CURRENT_SOURCE_DIR}/include/KFMElectrostaticFieldMapper_SingleThread.hh
    ${CMAKE_CURRENT_SOURCE_DIR}/include/KFMElectrostaticBoundaryIntegratorEngine_SingleThread.hh
    ${CMAKE_CURRENT_SOURCE_DIR}/include/KFMElectrostaticLocalCoefficientFieldCalculator.hh
    ${CMAKE_CURRENT_SOURCE_DIR}/include/KFMElectrostaticTreeInformationExtractor.hh
    ${CMAKE_CURRENT_SOURCE_DIR}/include/KFMElectrostaticTreeData.hh
    ${CMAKE_CURRENT_SOURCE_DIR}/include/KFMElectrostaticNodeWorkScoreCalculator.hh
    ${CMAKE_CURRENT_SOURCE_DIR}/include/KFMElectrostaticNodeInspector.hh
    ${CMAKE_CURRENT_SOURCE_DIR}/include/KFMElectrostaticNearFieldToLocalCoefficientCalculator.hh
)

set (KFMELECTROSTATICS_SOURCEFILES
    ${CMAKE_CURRENT_SOURCE_DIR}/src/KFMElectrostaticMultipoleSet.cc
    ${CMAKE_CURRENT_SOURCE_DIR}/src/KFMElectrostaticLocalCoefficientSet.cc
    ${CMAKE_CURRENT_SOURCE_DIR}/src/KFMElectrostaticMultipoleCalculatorAnalytic.cc
    ${CMAKE_CURRENT_SOURCE_DIR}/src/KFMElectrostaticMultipoleCalculatorNumeric.cc
    ${CMAKE_CURRENT_SOURCE_DIR}/src/KFMElectrostaticLocalCoefficientCalculatorNumeric.cc
    ${CMAKE_CURRENT_SOURCE_DIR}/src/KFMElectrostaticMultipoleBatchCalculator.cc
    ${CMAKE_CURRENT_SOURCE_DIR}/src/KFMElectrostaticRegionSizeEstimator.cc
    ${CMAKE_CURRENT_SOURCE_DIR}/src/KFMElectrostaticTreeBuilder.cc
    ${CMAKE_CURRENT_SOURCE_DIR}/src/KFMElectrostaticFieldMapper_SingleThread.cc
    ${CMAKE_CURRENT_SOURCE_DIR}/src/KFMElectrostaticBoundaryIntegratorEngine_SingleThread.cc
    ${CMAKE_CURRENT_SOURCE_DIR}/src/KFMElectrostaticLocalCoefficientFieldCalculator.cc
    ${CMAKE_CURRENT_SOURCE_DIR}/src/KFMElectrostaticTreeData.cc
    ${CMAKE_CURRENT_SOURCE_DIR}/src/KFMElectrostaticNodeWorkScoreCalculator.cc
)

if (KEMField_USE_MPI)
    list (APPEND KFMELECTROSTATICS_HEADERFILES
        ${CMAKE_CURRENT_SOURCE_DIR}/include/KFMElectrostaticTreeBuilder_MPI.hh
    )
    list (APPEND KFMELECTROSTATICS_SOURCEFILES
        ${CMAKE_CURRENT_SOURCE_DIR}/src/KFMElectrostaticTreeBuilder_MPI.cc
    )
endif (KEMField_USE_MPI)

add_library (KFMElectrostatics SHARED
    ${KFMELECTROSTATICS_SOURCEFILES} ${KFMELECTROSTATICS_HEADERFILES})
target_include_directories(KFMElectrostatics
    PUBLIC $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include> $<INSTALL_INTERFACE:include>)
target_link_libraries (KFMElectrostatics
    PUBLIC
        KEMFileManipulation
        KFMCore
        KFMMath
        KFMVectorMath
        KFMKernel
        KFMTree
)

kasper_install_headers (${KFMELECTROSTATICS_HEADERFILES})
kasper_install_libraries (KFMElectrostatics)
