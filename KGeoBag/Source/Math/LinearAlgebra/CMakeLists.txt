# headers
set( MATH_LINALG_HEADER_FILES
    Include/KGLinearAlgebraDefinitions.hh
    Include/KGLinearSystemSolver.hh
    Include/KGMatrixOperations.hh
    Include/KGMatrixVectorOperations.hh
    Include/KGVectorOperations.hh
    Include/KGNumericalConstants.hh
    Include/KGArrayMath.hh
)

# source
set( MATH_LINALG_SOURCE_FILES
    Source/KGLinearSystemSolver.cc
    Source/KGMatrixOperations.cc
    Source/KGMatrixVectorOperations.cc
    Source/KGVectorOperations.cc
)

# internal
kasper_internal_include_directories(
	${CMAKE_CURRENT_SOURCE_DIR}/Include
)

# target
add_library( KGeoBagMathLinearAlgebra SHARED
	${MATH_LINALG_SOURCE_FILES}
)
target_link_libraries( KGeoBagMathLinearAlgebra
    KGeoBagMath
    ${Kommon_LIBRARIES}
    ${GSL_LIBRARIES}
)

# install
kasper_install_headers( ${MATH_LINALG_HEADER_FILES} )
kasper_install_libraries( KGeoBagMathLinearAlgebra )
