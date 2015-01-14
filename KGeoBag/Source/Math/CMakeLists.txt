# headers
set( MATH_HEADER_FILES
    Include/KTwoVector.hh
    Include/KTwoMatrix.hh
    Include/KThreeVector.hh
    Include/KThreeMatrix.hh
    Include/KRotation.hh
    Include/KTransformation.hh
    Include/KGCoordinateTransform.hh
    Include/KFrame.hh
    Include/KAxis.hh
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
    Source/KTwoVector.cc
    Source/KThreeVector.cc
    Source/KTwoMatrix.cc
    Source/KThreeMatrix.cc
    Source/KRotation.cc
    Source/KTransformation.cc
    Source/KGCoordinateTransform.cc
    Source/KFrame.cc
    Source/KAxis.cc
    Source/KGInterpolator.cc
    Source/KGTrigonometricInterpolator.cc
    Source/KGLinearSplineInterpolator.cc
    Source/KGCubicSplineInterpolator.cc
    Source/KGBivariateInterpolator.cc
    Source/KGLinearCongruentialGenerator.cc
    Source/KGMathMessage.cc
)

# internal
kasper_internal_include_directories(
	${CMAKE_CURRENT_SOURCE_DIR}/Include
)

# target
add_library( KGeoBagMath SHARED
	${MATH_SOURCE_FILES}
)
target_link_libraries( KGeoBagMath
    ${Kommon_LIBRARIES}
    ${GSL_LIBRARIES}
)

# install
kasper_install_headers( ${MATH_HEADER_FILES} )
kasper_install_libraries( KGeoBagMath )
