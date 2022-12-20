# headers
set( MATH_HEADER_FILES
    Include/KGCoordinateTransform.hh
    Include/KGDataPoint.hh
    Include/KGInterpolator.hh
    Include/KGTrigonometricInterpolator.hh
    Include/KGLinearSplineInterpolator.hh
    Include/KGCubicSplineInterpolator.hh
    Include/KGBivariateInterpolator.hh
    Include/KGLinearCongruentialGenerator.hh
    Include/KGMathMessage.hh
)

# source
set( MATH_SOURCE_FILES
    Source/KGCoordinateTransform.cc
    Source/KGInterpolator.cc
    Source/KGTrigonometricInterpolator.cc
    Source/KGLinearSplineInterpolator.cc
    Source/KGCubicSplineInterpolator.cc
    Source/KGBivariateInterpolator.cc
    Source/KGLinearCongruentialGenerator.cc
    Source/KGMathMessage.cc
)

# target
add_library( KGeoBagMath SHARED
    ${MATH_SOURCE_FILES} ${MATH_HEADER_FILES})
target_include_directories( KGeoBagMath
    PUBLIC $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/Include> $<INSTALL_INTERFACE:include>)
target_link_libraries( KGeoBagMath
    PUBLIC
        KommonCore
        KommonMath
)

# install
kasper_install_headers( ${MATH_HEADER_FILES} )
kasper_install_libraries( KGeoBagMath )
