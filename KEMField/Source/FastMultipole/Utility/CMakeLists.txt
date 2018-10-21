# CMakeLists for FastMultipole/Utility
# Author: J. P. Barrett

kasper_internal_include_directories(${CMAKE_CURRENT_SOURCE_DIR}/include)

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

set (KFM_UTIL_LIBS
        KFMCore
        KFMMath
        KFMMathUtilities
        KVectorMath
        KFMKernel
        KFMTree
        KFMElectrostatics
        KFMInterfaceExtraction
        KFMInterfaceFieldSolvers
        KEMCore
        KEMFieldExceptions
        KEMFileManipulation
        KEMHashGenerator
        KEMLinearAlgebraSolvers
        KEMStructuredASCII
        KEMMath
        KEMSurfaces
        KEMIO
)

if(@PROJECT_NAME@_USE_OPENCL)
  list (APPEND KFM_UTIL_LIBS KEMOpenCLPlugin ${OPENCL_LIBRARIES})
  add_cflag (KEMFIELD_USE_OPENCL)
endif(@PROJECT_NAME@_USE_OPENCL)

if (@PROJECT_NAME@_USE_VTK)
    list (APPEND KFM_UTIL_LIBS KEMVTKPlugin)
    add_cflag (KEMFIELD_USE_VTK)
endif (@PROJECT_NAME@_USE_VTK)

add_library (KFMUtility SHARED ${KFM_UTIL_SOURCEFILES})
target_link_libraries (KFMUtility ${KFM_UTIL_LIBS})

kasper_install_headers (${KFM_UTIL_HEADERFILES})
kasper_install_libraries (KFMUtility)
