# CMakeLists for KEMField/FastMultipole/Math
# Author: J. P. Barrett

option (KEMField_USE_GSL "Use GSL" OFF)
if (KEMField_USE_GSL)
  find_package(GSL REQUIRED)
  kasper_external_include_directories (${GSL_INCLUDE_DIRS})
  add_cflag(KEMFIELD_USE_GSL)
endif (KEMField_USE_GSL)

option (KEMField_USE_FFTW "Use FFTW" OFF)
if (KEMField_USE_FFTW)
  find_package(FFTW 3.3.4 REQUIRED)
  kasper_external_include_directories (${FFTW_INCLUDE_DIRS})
  add_cflag(KEMFIELD_USE_FFTW)
endif (KEMField_USE_FFTW)

kasper_internal_include_directories(${CMAKE_CURRENT_SOURCE_DIR}/include)

set (MATH_UTILITIES_HEADERFILES
    ${CMAKE_CURRENT_SOURCE_DIR}/include/KFMMath.hh
    ${CMAKE_CURRENT_SOURCE_DIR}/include/KFMBitReversalPermutation.hh
    ${CMAKE_CURRENT_SOURCE_DIR}/include/KFMGaussLegendreQuadratureTableCalculator.hh
    ${CMAKE_CURRENT_SOURCE_DIR}/include/KFMMatrixOperations.hh
    ${CMAKE_CURRENT_SOURCE_DIR}/include/KFMNumericalConstants.hh
    ${CMAKE_CURRENT_SOURCE_DIR}/include/KFMFastFourierTransform.hh
    ${CMAKE_CURRENT_SOURCE_DIR}/include/KFMLinearAlgebraDefinitions.hh
    ${CMAKE_CURRENT_SOURCE_DIR}/include/KFMMatrixVectorOperations.hh
    ${CMAKE_CURRENT_SOURCE_DIR}/include/KFMVectorOperations.hh
    ${CMAKE_CURRENT_SOURCE_DIR}/include/KFMFastFourierTransformUtilities.hh
    ${CMAKE_CURRENT_SOURCE_DIR}/include/KFMLinearSystemSolver.hh
    ${CMAKE_CURRENT_SOURCE_DIR}/include/KFMMultidimensionalFastFourierTransform.hh
    ${CMAKE_CURRENT_SOURCE_DIR}/include/KFMSquareRootUtilities.hh
)

if(KEMField_USE_FFTW)
    set( MATH_UTILITIES_HEADERFILES ${MATH_UTILITIES_HEADERFILES}     ${CMAKE_CURRENT_SOURCE_DIR}/include/KFMMultidimensionalFastFourierTransformFFTW.hh )
endif(KEMField_USE_FFTW)

set (MATH_UTILITIES_SOURCEFILES
    ${CMAKE_CURRENT_SOURCE_DIR}/src/KFMMath.cc
    ${CMAKE_CURRENT_SOURCE_DIR}/src/KFMBitReversalPermutation.cc
    ${CMAKE_CURRENT_SOURCE_DIR}/src/KFMFastFourierTransformUtilities.cc
    ${CMAKE_CURRENT_SOURCE_DIR}/src/KFMLinearSystemSolver.cc
    ${CMAKE_CURRENT_SOURCE_DIR}/src/KFMMatrixVectorOperations.cc
    ${CMAKE_CURRENT_SOURCE_DIR}/src/KFMFastFourierTransform.cc
    ${CMAKE_CURRENT_SOURCE_DIR}/src/KFMGaussLegendreQuadratureTableCalculator.cc
    ${CMAKE_CURRENT_SOURCE_DIR}/src/KFMMatrixOperations.cc
    ${CMAKE_CURRENT_SOURCE_DIR}/src/KFMVectorOperations.cc
    ${CMAKE_CURRENT_SOURCE_DIR}/src/KFMSquareRootUtilities.cc
)

add_library (KFMMathUtilities SHARED ${MATH_UTILITIES_SOURCEFILES})

if(KEMField_USE_GSL)
    if(KEMField_USE_FFTW)
        target_link_libraries (KFMMathUtilities KEMCore ${GSL_LIBRARIES} ${FFTW_LIBRARIES})
    else(KEMField_USE_FFTW)
        target_link_libraries (KFMMathUtilities KEMCore ${GSL_LIBRARIES})
    endif(KEMField_USE_FFTW)
else(KEMField_USE_GSL)
    if(KEMField_USE_FFTW)
        target_link_libraries (KFMMathUtilities KEMCore ${FFTW_LIBRARIES})
    else(KEMField_USE_FFTW)
        target_link_libraries (KFMMathUtilities KEMCore )
    endif(KEMField_USE_FFTW)
endif (KEMField_USE_GSL)

kasper_install_headers (${MATH_UTILITIES_HEADERFILES})
kasper_install_libraries (KFMMathUtilities)
