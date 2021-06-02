# CMakeLists for KEMField/Math
# Author: T.J. Corona

set (MATH_HEADERFILES
    ${CMAKE_CURRENT_SOURCE_DIR}/include/KEllipticIntegrals.hh
    ${CMAKE_CURRENT_SOURCE_DIR}/include/KEMCoordinateSystem.hh
    ${CMAKE_CURRENT_SOURCE_DIR}/include/KEMTransformation.hh
    ${CMAKE_CURRENT_SOURCE_DIR}/include/KThreeVector_KEMField.hh
    ${CMAKE_CURRENT_SOURCE_DIR}/include/KEMStreamableThreeVector.hh
    ${CMAKE_CURRENT_SOURCE_DIR}/include/KThreeMatrix_KEMField.hh
    ${CMAKE_CURRENT_SOURCE_DIR}/include/KGaussianQuadrature.hh
    ${CMAKE_CURRENT_SOURCE_DIR}/include/KGaussLegendreQuadrature.hh
    ${CMAKE_CURRENT_SOURCE_DIR}/include/KShanksTransformation.hh
)

set (MATH_SOURCEFILES
    ${CMAKE_CURRENT_SOURCE_DIR}/src/KEllipticIntegrals.cc
    ${CMAKE_CURRENT_SOURCE_DIR}/src/KEMCoordinateSystem.cc
    ${CMAKE_CURRENT_SOURCE_DIR}/src/KGaussianQuadrature.cc
    ${CMAKE_CURRENT_SOURCE_DIR}/src/KGaussLegendreQuadrature.cc
)

add_library (KEMMath SHARED
    ${MATH_SOURCEFILES} ${MATH_HEADERFILES})
target_include_directories(KEMMath
    PUBLIC $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include> $<INSTALL_INTERFACE:include>)
target_link_libraries (KEMMath
    PUBLIC
        KGeoBagMath
)

if(KEMField_USE_GSL)
    target_link_libraries(KEMMath PUBLIC GSL::gsl)
    target_compile_definitions(KEMMath PUBLIC KEMFIELD_USE_GSL)
endif(KEMField_USE_GSL)

kasper_install_headers (${MATH_HEADERFILES})
kasper_install_libraries (KEMMath)
