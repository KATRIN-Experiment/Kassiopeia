# CMakeLists for KEMField/FastMultipole/Subdivision
# Author: J. P. Barrett

set (INTERFACE_BOUNDARY_INTEGRAL_HEADERFILES
    ${CMAKE_CURRENT_SOURCE_DIR}/include/KFMBoundaryIntegralMatrix.hh
    ${CMAKE_CURRENT_SOURCE_DIR}/include/KFMElectrostaticBoundaryIntegrator.hh
    ${CMAKE_CURRENT_SOURCE_DIR}/include/KFMDenseBoundaryIntegralMatrix.hh
    ${CMAKE_CURRENT_SOURCE_DIR}/include/KFMDenseBlockSparseBoundaryIntegralMatrix.hh
    ${CMAKE_CURRENT_SOURCE_DIR}/include/KFMSparseBoundaryIntegralMatrix.hh
    ${CMAKE_CURRENT_SOURCE_DIR}/include/KFMSparseBoundaryIntegralMatrix_BlockCompressedRow.hh
    ${CMAKE_CURRENT_SOURCE_DIR}/include/KFMSparseBoundaryIntegralMatrix_CompressedRow.hh
)

if (KEMField_USE_MPI)
    list (APPEND INTERFACE_BOUNDARY_INTEGRAL_HEADERFILES
        ${CMAKE_CURRENT_SOURCE_DIR}/include/KFMElectrostaticBoundaryIntegrator_MPI.hh
    )
endif (KEMField_USE_MPI)

add_library(KFMInterfaceBoundaryIntegral INTERFACE)
target_include_directories(KFMInterfaceBoundaryIntegral
    INTERFACE $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include> $<INSTALL_INTERFACE:include>)
target_link_libraries(KFMInterfaceBoundaryIntegral INTERFACE KFMInterfaceFieldSolvers)

kasper_install_headers (${INTERFACE_BOUNDARY_INTEGRAL_HEADERFILES})
kasper_install_libraries (KFMInterfaceBoundaryIntegral)
