# CMakeLists for KEMField/FastMultipole/Array
# Author: J. P. Barrett

set (KFMARRAY_HEADERFILES
    ${CMAKE_CURRENT_SOURCE_DIR}/include/KFMArrayMath.hh
    ${CMAKE_CURRENT_SOURCE_DIR}/include/KFMArrayWrapper.hh
    ${CMAKE_CURRENT_SOURCE_DIR}/include/KFMArrayOperator.hh
    ${CMAKE_CURRENT_SOURCE_DIR}/include/KFMUnaryArrayOperator.hh
    ${CMAKE_CURRENT_SOURCE_DIR}/include/KFMBinaryArrayOperator.hh
    ${CMAKE_CURRENT_SOURCE_DIR}/include/KFMArrayFillingOperator.hh
    ${CMAKE_CURRENT_SOURCE_DIR}/include/KFMPointwiseArrayAdder.hh
    ${CMAKE_CURRENT_SOURCE_DIR}/include/KFMPointwiseArrayScaledAdder.hh
    ${CMAKE_CURRENT_SOURCE_DIR}/include/KFMPointwiseArrayMultiplier.hh
    ${CMAKE_CURRENT_SOURCE_DIR}/include/KFMPointwiseArrayReversedConjugateMultiplier.hh
    ${CMAKE_CURRENT_SOURCE_DIR}/include/KFMArrayScalarMultiplier.hh
    ${CMAKE_CURRENT_SOURCE_DIR}/include/KFMMessaging.hh
)

add_library(KFMArray INTERFACE)
target_include_directories(KFMArray
    INTERFACE $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include> $<INSTALL_INTERFACE:include>)

kasper_install_headers (${KFMARRAY_HEADERFILES})
kasper_install_libraries (KFMArray)
