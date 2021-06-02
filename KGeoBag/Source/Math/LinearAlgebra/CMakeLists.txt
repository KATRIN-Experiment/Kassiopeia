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

# target
add_library( KGeoBagMathLinearAlgebra SHARED
    ${MATH_LINALG_SOURCE_FILES} ${MATH_LINALG_HEADER_FILES})
target_include_directories( KGeoBagMathLinearAlgebra
    PUBLIC $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/Include> $<INSTALL_INTERFACE:include>
)
target_link_libraries( KGeoBagMathLinearAlgebra
    PUBLIC
        KGeoBagMath
)

if(KGeoBag_USE_GSL)
    target_link_libraries( KGeoBagMathLinearAlgebra PUBLIC GSL::gsl )
    target_compile_definitions( KGeoBagMathLinearAlgebra PUBLIC KGEOBAG_MATH_USE_GSL )
endif(KGeoBag_USE_GSL)

# install
kasper_install_headers( ${MATH_LINALG_HEADER_FILES} )
kasper_install_libraries( KGeoBagMathLinearAlgebra )
