# CMakeLists for KEMField/FastMultipole/Math
# Author: J. P. Barrett

set (MATH_HEADERFILES
    ${CMAKE_CURRENT_SOURCE_DIR}/include/KFMRealSphericalHarmonicExpansionRotator.hh
    ${CMAKE_CURRENT_SOURCE_DIR}/include/KFMComplexSphericalHarmonicExpansionRotator.hh
    ${CMAKE_CURRENT_SOURCE_DIR}/include/KFMTrianglePolarBasisCalculator.hh
    ${CMAKE_CURRENT_SOURCE_DIR}/include/KFMPoint.hh
    ${CMAKE_CURRENT_SOURCE_DIR}/include/KFMBall.hh
    ${CMAKE_CURRENT_SOURCE_DIR}/include/KFMCube.hh
    ${CMAKE_CURRENT_SOURCE_DIR}/include/KFMBox.hh
    ${CMAKE_CURRENT_SOURCE_DIR}/include/KFMPointCloud.hh
    ${CMAKE_CURRENT_SOURCE_DIR}/include/KFMBallCloud.hh
    ${CMAKE_CURRENT_SOURCE_DIR}/include/KFMBoxSupportSet.hh
    ${CMAKE_CURRENT_SOURCE_DIR}/include/KFMBallSupportSet.hh
    ${CMAKE_CURRENT_SOURCE_DIR}/include/KFMPointCloudToBoundingBallConverter.hh
    ${CMAKE_CURRENT_SOURCE_DIR}/include/KFMBoundaryCalculator.hh
    ${CMAKE_CURRENT_SOURCE_DIR}/include/KFMPinchonGMatrixCalculator.hh
    ${CMAKE_CURRENT_SOURCE_DIR}/include/KFMPinchonJMatrixCalculator.hh
    ${CMAKE_CURRENT_SOURCE_DIR}/include/KFMDenseBlockSparseMatrixStructure.hh
    ${CMAKE_CURRENT_SOURCE_DIR}/include/KFMDenseBlockSparseMatrix.hh
    ${CMAKE_CURRENT_SOURCE_DIR}/include/KFMDenseBlockSparseMatrix_MPI.hh
    ${CMAKE_CURRENT_SOURCE_DIR}/include/KFMCubicVolumeCollection.hh
)

set (MATH_SOURCEFILES
    ${CMAKE_CURRENT_SOURCE_DIR}/src/KFMRealSphericalHarmonicExpansionRotator.cc
    ${CMAKE_CURRENT_SOURCE_DIR}/src/KFMComplexSphericalHarmonicExpansionRotator.cc
    ${CMAKE_CURRENT_SOURCE_DIR}/src/KFMTrianglePolarBasisCalculator.cc
    ${CMAKE_CURRENT_SOURCE_DIR}/src/KFMPinchonGMatrixCalculator.cc
    ${CMAKE_CURRENT_SOURCE_DIR}/src/KFMPinchonJMatrixCalculator.cc
    ${CMAKE_CURRENT_SOURCE_DIR}/src/KFMDenseBlockSparseMatrixStructure.cc
)

add_library (KFMMath SHARED
    ${MATH_SOURCEFILES} ${MATH_HEADERFILES})
target_include_directories(KFMMath
    PUBLIC $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include> $<INSTALL_INTERFACE:include>)
target_link_libraries (KFMMath
    PUBLIC
        KEMMath
        KEMLinearAlgebraCore
        KFMCore
        KFMMathUtilities
)

kasper_install_headers (${MATH_HEADERFILES})
kasper_install_libraries (KFMMath)
