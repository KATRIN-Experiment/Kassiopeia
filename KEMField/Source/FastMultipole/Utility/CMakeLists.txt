# CMakeLists for FastMultipole/Utility
# Author: J. P. Barrett

set (KFM_UTIL_HEADERFILES
    ${CMAKE_CURRENT_SOURCE_DIR}/include/KFastMultipoleMatrixGenerator.hh
    ${CMAKE_CURRENT_SOURCE_DIR}/include/KFMElectrostaticTypes.hh
    ${CMAKE_CURRENT_SOURCE_DIR}/include/KFMNamedScalarData.hh
    ${CMAKE_CURRENT_SOURCE_DIR}/include/KFMNamedScalarDataCollection.hh
    ${CMAKE_CURRENT_SOURCE_DIR}/include/KFMElectrostaticFastMultipoleBoundaryValueSolver.hh
    ${CMAKE_CURRENT_SOURCE_DIR}/include/KFMElectrostaticFastMultipoleBoundaryValueSolverConfiguration.hh
    ${CMAKE_CURRENT_SOURCE_DIR}/include/KFMElectrostaticParametersConfiguration.hh
)

set (KFM_UTIL_SOURCEFILES
    ${CMAKE_CURRENT_SOURCE_DIR}/src/KFastMultipoleMatrixGenerator.cc
    ${CMAKE_CURRENT_SOURCE_DIR}/src/KFMNamedScalarData.cc
    ${CMAKE_CURRENT_SOURCE_DIR}/src/KFMNamedScalarDataCollection.cc
    ${CMAKE_CURRENT_SOURCE_DIR}/src/KFMElectrostaticFastMultipoleBoundaryValueSolver.cc
    ${CMAKE_CURRENT_SOURCE_DIR}/src/KFMElectrostaticFastMultipoleBoundaryValueSolverConfiguration.cc
)

add_library (KFMUtility SHARED
    ${KFM_UTIL_SOURCEFILES} ${KFM_UTIL_HEADERFILES})
target_include_directories(KFMUtility
    PUBLIC $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include> $<INSTALL_INTERFACE:include>)

target_link_libraries (KFMUtility
    PUBLIC
        KEMLinearAlgebraPreconditioner
        KEMLinearAlgebraSolvers
        KFMCore
        KFMInterfaceBoundaryIntegral
        KFMElectrostatics
)

if(KEMField_USE_OPENCL)
    target_link_libraries(KFMUtility PUBLIC KEMOpenCLPlugin)
endif(KEMField_USE_OPENCL)

if(KEMField_USE_VTK)
    target_link_libraries(KFMUtility PUBLIC KEMVTKPlugin)
endif(KEMField_USE_VTK)

kasper_install_headers (${KFM_UTIL_HEADERFILES})
kasper_install_libraries (KFMUtility)
