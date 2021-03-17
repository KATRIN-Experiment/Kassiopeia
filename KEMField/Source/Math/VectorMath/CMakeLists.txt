# CMakeLists for KEMField/VectorMath
# Author: J. P. Barrett

kasper_internal_include_directories(${CMAKE_CURRENT_SOURCE_DIR}/include)

set (VECTORMATH_HEADERFILES
    ${CMAKE_CURRENT_SOURCE_DIR}/include/KVMFixedArray.hh
    ${CMAKE_CURRENT_SOURCE_DIR}/include/KVMMap.hh
    ${CMAKE_CURRENT_SOURCE_DIR}/include/KVMField.hh
    ${CMAKE_CURRENT_SOURCE_DIR}/include/KVMFieldWrapper.hh
    ${CMAKE_CURRENT_SOURCE_DIR}/include/KVMNumericalIntegrator.hh
    ${CMAKE_CURRENT_SOURCE_DIR}/include/KVMCompactCurve.hh
    ${CMAKE_CURRENT_SOURCE_DIR}/include/KVMCompactSurface.hh
    ${CMAKE_CURRENT_SOURCE_DIR}/include/KVMCompactVolume.hh
    ${CMAKE_CURRENT_SOURCE_DIR}/include/KVMSpaceTriangle.hh
    ${CMAKE_CURRENT_SOURCE_DIR}/include/KVMSpaceRectangle.hh
    ${CMAKE_CURRENT_SOURCE_DIR}/include/KVMSpaceLineSegment.hh
    ${CMAKE_CURRENT_SOURCE_DIR}/include/KVMTriangularSurface.hh
    ${CMAKE_CURRENT_SOURCE_DIR}/include/KVMRectangularSurface.hh
    ${CMAKE_CURRENT_SOURCE_DIR}/include/KVMLineSegment.hh
    ${CMAKE_CURRENT_SOURCE_DIR}/include/KVMPathIntegral.hh
    ${CMAKE_CURRENT_SOURCE_DIR}/include/KVMLineIntegral.hh
    ${CMAKE_CURRENT_SOURCE_DIR}/include/KVMSurfaceIntegral.hh
    ${CMAKE_CURRENT_SOURCE_DIR}/include/KVMFluxIntegral.hh
)


set (VECTORMATH_SOURCEFILES
    ${CMAKE_CURRENT_SOURCE_DIR}/src/KVMCompactCurve.cc
    ${CMAKE_CURRENT_SOURCE_DIR}/src/KVMCompactSurface.cc
    ${CMAKE_CURRENT_SOURCE_DIR}/src/KVMSpaceTriangle.cc
    ${CMAKE_CURRENT_SOURCE_DIR}/src/KVMSpaceRectangle.cc
    ${CMAKE_CURRENT_SOURCE_DIR}/src/KVMSpaceLineSegment.cc
    ${CMAKE_CURRENT_SOURCE_DIR}/src/KVMTriangularSurface.cc
    ${CMAKE_CURRENT_SOURCE_DIR}/src/KVMRectangularSurface.cc
    ${CMAKE_CURRENT_SOURCE_DIR}/src/KVMLineSegment.cc
    ${CMAKE_CURRENT_SOURCE_DIR}/src/KVMLineIntegral.cc
    ${CMAKE_CURRENT_SOURCE_DIR}/src/KVMFluxIntegral.cc
)

##################################################

add_library (KVectorMath SHARED ${VECTORMATH_SOURCEFILES} ${VECTORMATH_HEADERFILES})
target_link_libraries (KVectorMath  KEMMath KFMMathUtilities ${Kommon_LIBRARIES} ${KGeoBag_LIBRARIES})

kasper_install_headers (${VECTORMATH_HEADERFILES})
kasper_install_libraries (KVectorMath)
