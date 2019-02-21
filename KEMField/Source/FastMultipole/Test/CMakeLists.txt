# CMakeLists for FastMultipole/Test
# Author: J. P. Barrett

option (${PROJECT_NAME}_ENABLE_FM_TEST "Build fast multipole developer test applications" OFF)
if (${PROJECT_NAME}_ENABLE_FM_TEST)
  enable_testing ()
endif (${PROJECT_NAME}_ENABLE_FM_TEST)

if (${PROJECT_NAME}_ENABLE_FM_TEST)

    kasper_internal_include_directories(${${PROJECT_NAME}_INCLUDE_DIRS})

    set (FAST_MULTIPOLE_TEST_LIBS
        ${KOMMON_LIBRARIES}
        ${KGeoBag_LIBRARIES}
        KGeoBagInterface
        KFMCore
        KFMMath
        KFMMathUtilities
        KVectorMath
        KFMKernel
        KFMElectrostatics
        KFMInterfaceExtraction
        KFMInterfaceFieldSolvers
        KFMUtility
        KEMCore
        KEMFileManipulation
        KEMHashGenerator
        KEMStructuredASCII
        KEMMath
        KEMSurfaces
        KEMIO
        KEMElectrostaticBoundaryIntegrals
    )

    if (${PROJECT_NAME}_USE_VTK)
    list (APPEND FAST_MULTIPOLE_TEST_LIBS
    KEMVTKPlugin
    )
    endif (${PROJECT_NAME}_USE_VTK)


    if (${PROJECT_NAME}_USE_ROOT)
    find_package(ROOT REQUIRED)
    kasper_external_include_directories( ${ROOT_INCLUDE_DIRS} )
    list (APPEND FAST_MULTIPOLE_TEST_LIBS
    KEMRootPlugin
    )
    endif (${PROJECT_NAME}_USE_ROOT)

    if (${PROJECT_NAME}_USE_OPENCL)
    kasper_external_include_directories( ${OPENCL_INCLUDE_DIRS} )
    list (APPEND FAST_MULTIPOLE_TEST_LIBS
    KEMOpenCLPlugin
    )
    endif (${PROJECT_NAME}_USE_OPENCL)

    if (${PROJECT_NAME}_USE_PETSc)
    list (APPEND FAST_MULTIPOLE_TEST_LIBS
    KEMPETScPlugin
    )
    endif (${PROJECT_NAME}_USE_PETSc)

    if (${PROJECT_NAME}_USE_VTK)
    list (APPEND FAST_MULTIPOLE_TEST_LIBS
      ${Kommon_Vtk_LIBRARIES}
      ${VTK_LIBRARIES})
    endif (${PROJECT_NAME}_USE_VTK)

    list(APPEND PROPERTY COMPILE_DEFINITIONS DEFAULT_DATA_DIR="${DATA_INSTALL_DIR}/${${PROJECT_NAME}_DATA_DIR}")

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
        TestMultipoleExpansionSpeed
        TestNumericalIntegrator
        TestPinchonJMatrixCalculator
        TestQuadratureTableCalculator
        TestSingleTriangleMultipoleExpansion
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
        target_link_libraries( ${BASENAME} ${FAST_MULTIPOLE_TEST_LIBS} )
    endforeach( BASENAME )

    kasper_install_executables( ${SOURCE_BASENAMES} )

endif (${PROJECT_NAME}_ENABLE_FM_TEST)
