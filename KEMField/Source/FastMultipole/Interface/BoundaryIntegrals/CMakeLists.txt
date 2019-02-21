# CMakeLists for KEMField/FastMultipole/Subdivision
# Author: J. P. Barrett

kasper_internal_include_directories(${CMAKE_CURRENT_SOURCE_DIR}/include)

set (INTERFACE_BOUNDARY_INTEGRAL_HEADERFILES
    ${CMAKE_CURRENT_SOURCE_DIR}/include/KFMBoundaryIntegralMatrix.hh
    ${CMAKE_CURRENT_SOURCE_DIR}/include/KFMElectrostaticBoundaryIntegrator.hh
    ${CMAKE_CURRENT_SOURCE_DIR}/include/KFMDenseBoundaryIntegralMatrix.hh
    ${CMAKE_CURRENT_SOURCE_DIR}/include/KFMSparseBoundaryIntegralMatrix_BlockCompressedRow.hh
    ${CMAKE_CURRENT_SOURCE_DIR}/include/KFMSparseBoundaryIntegralMatrix_CompressedRow.hh
)

  if (${PROJECT_NAME}_USE_MPI)
     list (APPEND INTERFACE_BOUNDARY_INTEGRAL_HEADERFILES
          ${CMAKE_CURRENT_SOURCE_DIR}/include/KFMElectrostaticBoundaryIntegrator_MPI.hh
	  )
  endif (${PROJECT_NAME}_USE_MPI)

##################################################

kasper_install_headers (${INTERFACE_BOUNDARY_INTEGRAL_HEADERFILES})
