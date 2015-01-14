# CMakeLists for KEMField/FastMultipole/Subdivision
# Author: J. P. Barrett

kasper_internal_include_directories(${CMAKE_CURRENT_SOURCE_DIR}/include)

set (INTERFACE_BOUNDARY_INTEGRAL_HEADERFILES
    ${CMAKE_CURRENT_SOURCE_DIR}/include/KFMBoundaryIntegralMatrix.hh
    ${CMAKE_CURRENT_SOURCE_DIR}/include/KFMSparseBoundaryIntegralMatrix.hh
    ${CMAKE_CURRENT_SOURCE_DIR}/include/KFMDenseBoundaryIntegralMatrix.hh
    ${CMAKE_CURRENT_SOURCE_DIR}/include/KFMElectrostaticBoundaryIntegrator.hh
)

##################################################

kasper_install_headers (${INTERFACE_BOUNDARY_INTEGRAL_HEADERFILES})
