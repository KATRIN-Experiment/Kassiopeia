# CMakeLists for FastMultipole/Test
# Author: J. P. Barrett

option (${PROJECT_NAME}_ENABLE_FM_TEST "Build Fast Multipole developer test applications" OFF)
if (${PROJECT_NAME}_ENABLE_FM_TEST)
    enable_testing ()
endif (${PROJECT_NAME}_ENABLE_FM_TEST)

if (${PROJECT_NAME}_ENABLE_FM_TEST)

    set (FAST_MULTIPOLE_TEST_LIBS
        KGeoBagInterface
        KEMLinearAlgebraPreconditioner
        KFMCore
    )

    if (${PROJECT_NAME}_USE_VTK)
        list (APPEND FAST_MULTIPOLE_TEST_LIBS KEMVTKPlugin)
    endif (${PROJECT_NAME}_USE_VTK)

    if (${PROJECT_NAME}_USE_ROOT)
        list (APPEND FAST_MULTIPOLE_TEST_LIBS KEMRootPlugin)
    endif (${PROJECT_NAME}_USE_ROOT)

    if (${PROJECT_NAME}_USE_OPENCL)
        list (APPEND FAST_MULTIPOLE_TEST_LIBS KEMOpenCLPlugin)
    endif (${PROJECT_NAME}_USE_OPENCL)

    if (${PROJECT_NAME}_USE_PETSc)
        list (APPEND FAST_MULTIPOLE_TEST_LIBS KEMPETScPlugin)
    endif (${PROJECT_NAME}_USE_PETSc)

    set(SOURCE_BASENAMES
        TestArrayMath
        TestBasicMatrixOperations
        TestBasicMatrixVectorOperations
        TestBasicVectorOperations
        TestBitReversalPermutation
        TestBoundingBall
        TestCube
        TestFastFourierTransformBluestein
        TestFastFourierTransformRadixThree
        TestFastFourierTransformRadixTwo
        TestFastMultipoleTranslation
        TestKrylovSolvers
        TestM2LCoefficients
        TestMatrixSingularValueDecomposition
        TestMultidimensionalFastFourierTransform
        TestMultipoleCalculators
#        TestMultipoleExpansionSpeed
        TestNumericalIntegrator
        TestPinchonJMatrixCalculator
        TestQuadratureTableCalculator
#        TestSingleTriangleMultipoleExpansion
        TestStructuredASCII
        TestStructuredASCIICompressed
        TestVectorMath
#        TestFastMultipoleDualTree
#        TestFastMultipoleFieldMap
        TestFastMultipoleFieldSolver
        TestMultilevelPreconditioners
    )

    if (${PROJECT_NAME}_USE_OPENCL)
        list(APPEND SOURCE_BASENAMES
            TestMultidimensionalFastFourierTransformOpenCL
            TestMultidimensionalFastFourierTransformSpeed
        )
    endif (${PROJECT_NAME}_USE_OPENCL)

    foreach( BASENAME ${SOURCE_BASENAMES} )
        add_executable( ${BASENAME} ${BASENAME}.cc )
        set_property(SOURCE ${CMAKE_CURRENT_SOURCE_DIR}/${BASENAME}.cc
            APPEND PROPERTY COMPILE_DEFINITIONS DEFAULT_DATA_DIR="${DATA_INSTALL_DIR}/${KEMField_DATA_DIR}"
        )
        target_link_libraries( ${BASENAME} ${FAST_MULTIPOLE_TEST_LIBS} )
    endforeach( BASENAME )

    kasper_install_executables( ${SOURCE_BASENAMES} )

endif (${PROJECT_NAME}_ENABLE_FM_TEST)
