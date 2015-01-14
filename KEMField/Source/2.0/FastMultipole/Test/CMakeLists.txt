# CMakeLists for FastMultipole/Test
# Author: J. P. Barrett

option (@PROJECT_NAME@_ENABLE_FM_TEST "Build fast multipole developer test applications" OFF)
if (@PROJECT_NAME@_ENABLE_FM_TEST)
  enable_testing ()
endif (@PROJECT_NAME@_ENABLE_FM_TEST)

if (@PROJECT_NAME@_ENABLE_FM_TEST)

    kasper_include_default_dirs()

    set (FAST_MULTIPOLE_TEST_LIBS
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
        KEMIntegratingFieldSolver
    )

    if (@PROJECT_NAME@_USE_VTK)
    list (APPEND FAST_MULTIPOLE_TEST_LIBS
    KEMVTKPlugin
    )
    endif (@PROJECT_NAME@_USE_VTK)


    if (@PROJECT_NAME@_USE_ROOT)
    list (APPEND FAST_MULTIPOLE_TEST_LIBS
    KEMRootPlugin
    )
    endif (@PROJECT_NAME@_USE_ROOT)

    if (@PROJECT_NAME@_USE_OPENCL)
    list (APPEND FAST_MULTIPOLE_TEST_LIBS
    KEMOpenCLPlugin
    )
    endif (@PROJECT_NAME@_USE_OPENCL)

    if (@PROJECT_NAME@_USE_KGEOBAG)
    kasper_find_module(KGeoBag)
    list (APPEND FAST_MULTIPOLE_TEST_LIBS
      KEMGeoBagPlugin
      ${KOMMON_LIBRARIES}
    )
    endif (@PROJECT_NAME@_USE_KGEOBAG)

    if (@PROJECT_NAME@_USE_KMESSAGE)
    kasper_find_module(Kommon)
    list (APPEND FAST_MULTIPOLE_TEST_LIBS
      ${KOMMON_LIBRARIES}
    )
    endif (@PROJECT_NAME@_USE_KMESSAGE)

    if (@PROJECT_NAME@_USE_PETSc)
    list (APPEND FAST_MULTIPOLE_TEST_LIBS
    KEMPETScPlugin
    )
    endif (@PROJECT_NAME@_USE_PETSc)

    if (@PROJECT_NAME@_USE_VTK)
    list (APPEND FAST_MULTIPOLE_TEST_LIBS
      ${VTK_LIBRARIES})
    endif (@PROJECT_NAME@_USE_VTK)

    set_property(
        SOURCE
        ${CMAKE_CURRENT_SOURCE_DIR}/TestStructuredASCII.cc
        ${CMAKE_CURRENT_SOURCE_DIR}/TestStructuredASCIICompressed.cc
        ${CMAKE_CURRENT_SOURCE_DIR}/TestArrayMath.cc
        ${CMAKE_CURRENT_SOURCE_DIR}/TestCube.cc
        ${CMAKE_CURRENT_SOURCE_DIR}/TestBoundingBall.cc
        ${CMAKE_CURRENT_SOURCE_DIR}/TestBasicVectorOperations.cc
        ${CMAKE_CURRENT_SOURCE_DIR}/TestBasicMatrixOperations.cc
        ${CMAKE_CURRENT_SOURCE_DIR}/TestMatrixSingularValueDecomposition.cc
        ${CMAKE_CURRENT_SOURCE_DIR}/TestBitReversalPermutation.cc
        ${CMAKE_CURRENT_SOURCE_DIR}/TestFastFourierTransformRadixTwo.cc
        ${CMAKE_CURRENT_SOURCE_DIR}/TestFastFourierTransformRadixThree.cc
        ${CMAKE_CURRENT_SOURCE_DIR}/TestFastFourierTransformBluestein.cc
        ${CMAKE_CURRENT_SOURCE_DIR}/TestOneDimensionalFastFourierTransform.cc
        ${CMAKE_CURRENT_SOURCE_DIR}/TestMultidimensionalFastFourierTransform.cc
        ${CMAKE_CURRENT_SOURCE_DIR}/TestNumericalIntegrator.cc
        ${CMAKE_CURRENT_SOURCE_DIR}/TestVectorMath.cc
        ${CMAKE_CURRENT_SOURCE_DIR}/TestKrylovSolvers.cc
        ${CMAKE_CURRENT_SOURCE_DIR}/TestFastMultipoleFieldMap.cc
        ${CMAKE_CURRENT_SOURCE_DIR}/TestFastMultipoleFieldSolver.cc
#        ${CMAKE_CURRENT_SOURCE_DIR}/ComputeSphericalCapacitor_FastMultipole.cc
#        ${CMAKE_CURRENT_SOURCE_DIR}/TestFastMultipoleDualTree.cc
#        ${CMAKE_CURRENT_SOURCE_DIR}/TestFastMultipoleChargeDensitySolver.cc
#        ${CMAKE_CURRENT_SOURCE_DIR}/TestMultipoleCalculators.cc
#        ${CMAKE_CURRENT_SOURCE_DIR}/TestFastMultipoleSphereCapacitance.cc
#        ${CMAKE_CURRENT_SOURCE_DIR}/TestSingleTriangleMultipoleExpansion.cc
#        ${CMAKE_CURRENT_SOURCE_DIR}/TestMultipoleExpansionSpeed.cc
#        ${CMAKE_CURRENT_SOURCE_DIR}/TestFastMultipoleGeometryAnalysis.cc
#        ${CMAKE_CURRENT_SOURCE_DIR}/ConvertFieldMapToROOT.cc
#        ${CMAKE_CURRENT_SOURCE_DIR}/TestM2LCorrelation.cc
#        ${CMAKE_CURRENT_SOURCE_DIR}/TestM2LMemorySavings.cc
        ${CMAKE_CURRENT_SOURCE_DIR}/TestPinchonJMatrixCalculator.cc
        APPEND PROPERTY COMPILE_DEFINITIONS DEFAULT_DATA_DIR="${DATA_INSTALL_DIR}/${@PROJECT_NAME@_DATA_DIR}"
    )

    if (@PROJECT_NAME@_USE_OPENCL)
    list(APPEND SOURCE
        ${CMAKE_CURRENT_SOURCE_DIR}/TestMultipoleBatchCalculatorOpenCL.cc
        ${CMAKE_CURRENT_SOURCE_DIR}/TestMultidimensionalFastFourierTransformOpenCL.cc
    )
    endif (@PROJECT_NAME@_USE_OPENCL)

    add_executable (TestStructuredASCII
    ${CMAKE_CURRENT_SOURCE_DIR}/TestStructuredASCII.cc)
    target_link_libraries (TestStructuredASCII ${FAST_MULTIPOLE_TEST_LIBS} )


    add_executable (TestStructuredASCIICompressed
    ${CMAKE_CURRENT_SOURCE_DIR}/TestStructuredASCIICompressed.cc)
    target_link_libraries (TestStructuredASCIICompressed ${FAST_MULTIPOLE_TEST_LIBS} )

    add_executable (TestPinchonJMatrixCalculator
    ${CMAKE_CURRENT_SOURCE_DIR}/TestPinchonJMatrixCalculator.cc)
    target_link_libraries (TestPinchonJMatrixCalculator ${FAST_MULTIPOLE_TEST_LIBS} )


    add_executable (TestArrayMath
    ${CMAKE_CURRENT_SOURCE_DIR}/TestArrayMath.cc)
    target_link_libraries (TestArrayMath ${FAST_MULTIPOLE_TEST_LIBS} )

    add_executable (TestM2LCorrelation
    ${CMAKE_CURRENT_SOURCE_DIR}/TestM2LCorrelation.cc)
    target_link_libraries (TestM2LCorrelation ${FAST_MULTIPOLE_TEST_LIBS} )

    add_executable (TestM2LMemorySavings
    ${CMAKE_CURRENT_SOURCE_DIR}/TestM2LMemorySavings.cc)
    target_link_libraries (TestM2LMemorySavings ${FAST_MULTIPOLE_TEST_LIBS} )

    add_executable (PrintM2LCoefficients
    ${CMAKE_CURRENT_SOURCE_DIR}/PrintM2LCoefficients.cc)
    target_link_libraries (PrintM2LCoefficients ${FAST_MULTIPOLE_TEST_LIBS} )

    add_executable (TestCube
    ${CMAKE_CURRENT_SOURCE_DIR}/TestCube.cc)
    target_link_libraries (TestCube ${FAST_MULTIPOLE_TEST_LIBS} )

    add_executable (TestBitReversalPermutation
    ${CMAKE_CURRENT_SOURCE_DIR}/TestBitReversalPermutation.cc)
    target_link_libraries (TestBitReversalPermutation ${FAST_MULTIPOLE_TEST_LIBS} )

    add_executable (TestFastFourierTransformRadixTwo
    ${CMAKE_CURRENT_SOURCE_DIR}/TestFastFourierTransformRadixTwo.cc)
    target_link_libraries (TestFastFourierTransformRadixTwo ${FAST_MULTIPOLE_TEST_LIBS} )

    add_executable (TestFastFourierTransformRadixThree
    ${CMAKE_CURRENT_SOURCE_DIR}/TestFastFourierTransformRadixThree.cc)
    target_link_libraries (TestFastFourierTransformRadixThree ${FAST_MULTIPOLE_TEST_LIBS} )

    add_executable (TestFastFourierTransformBluestein
    ${CMAKE_CURRENT_SOURCE_DIR}/TestFastFourierTransformBluestein.cc)
    target_link_libraries (TestFastFourierTransformBluestein ${FAST_MULTIPOLE_TEST_LIBS} )

    add_executable (TestOneDimensionalFastFourierTransform
    ${CMAKE_CURRENT_SOURCE_DIR}/TestOneDimensionalFastFourierTransform.cc)
    target_link_libraries (TestOneDimensionalFastFourierTransform ${FAST_MULTIPOLE_TEST_LIBS} )

    add_executable (TestMultidimensionalFastFourierTransform
    ${CMAKE_CURRENT_SOURCE_DIR}/TestMultidimensionalFastFourierTransform.cc)
    target_link_libraries (TestMultidimensionalFastFourierTransform ${FAST_MULTIPOLE_TEST_LIBS} )

    add_executable (TestQuadratureTableCalculator
    ${CMAKE_CURRENT_SOURCE_DIR}/TestQuadratureTableCalculator.cc)
    target_link_libraries (TestQuadratureTableCalculator ${FAST_MULTIPOLE_TEST_LIBS} )

    add_executable (TestNumericalIntegrator
    ${CMAKE_CURRENT_SOURCE_DIR}/TestNumericalIntegrator.cc)
    target_link_libraries (TestNumericalIntegrator ${FAST_MULTIPOLE_TEST_LIBS} )

    add_executable (TestMatrixSingularValueDecomposition
    ${CMAKE_CURRENT_SOURCE_DIR}/TestMatrixSingularValueDecomposition.cc)
    target_link_libraries (TestMatrixSingularValueDecomposition ${FAST_MULTIPOLE_TEST_LIBS} )

    add_executable (TestBoundingBall
    ${CMAKE_CURRENT_SOURCE_DIR}/TestBoundingBall.cc)
    target_link_libraries (TestBoundingBall ${FAST_MULTIPOLE_TEST_LIBS} )

    add_executable (TestBasicVectorOperations
    ${CMAKE_CURRENT_SOURCE_DIR}/TestBasicVectorOperations.cc)
    target_link_libraries (TestBasicVectorOperations ${FAST_MULTIPOLE_TEST_LIBS} )

    add_executable (TestBasicMatrixOperations
    ${CMAKE_CURRENT_SOURCE_DIR}/TestBasicMatrixOperations.cc)
    target_link_libraries (TestBasicMatrixOperations ${FAST_MULTIPOLE_TEST_LIBS} )

    add_executable (TestVectorMath
    ${CMAKE_CURRENT_SOURCE_DIR}/TestVectorMath.cc)
    target_link_libraries (TestVectorMath ${FAST_MULTIPOLE_TEST_LIBS} )

    add_executable (TestKrylovSolvers
    ${CMAKE_CURRENT_SOURCE_DIR}/TestKrylovSolvers.cc)
    target_link_libraries (TestKrylovSolvers ${FAST_MULTIPOLE_TEST_LIBS} )

#    add_executable (TestMultipoleCalculators
#    ${CMAKE_CURRENT_SOURCE_DIR}/TestMultipoleCalculators.cc)
#    target_link_libraries (TestMultipoleCalculators ${FAST_MULTIPOLE_TEST_LIBS} )

#    add_executable (TestSingleTriangleMultipoleExpansion
#    ${CMAKE_CURRENT_SOURCE_DIR}/TestSingleTriangleMultipoleExpansion.cc)
#    target_link_libraries (TestSingleTriangleMultipoleExpansion ${FAST_MULTIPOLE_TEST_LIBS} )

#    add_executable (TestMultipoleExpansionSpeed
#    ${CMAKE_CURRENT_SOURCE_DIR}/TestMultipoleExpansionSpeed.cc)
#    target_link_libraries (TestMultipoleExpansionSpeed ${FAST_MULTIPOLE_TEST_LIBS} )

#    add_executable (ConvertFieldMapToROOT
#    ${CMAKE_CURRENT_SOURCE_DIR}/ConvertFieldMapToROOT.cc)
#    target_link_libraries (ConvertFieldMapToROOT ${FAST_MULTIPOLE_TEST_LIBS} )

    kasper_install_executables (
        TestStructuredASCII
        TestStructuredASCIICompressed
        TestArrayMath
        PrintM2LCoefficients
        TestCube
        TestMatrixSingularValueDecomposition
        TestBitReversalPermutation
        TestFastFourierTransformRadixTwo
        TestFastFourierTransformRadixThree
        TestFastFourierTransformBluestein
        TestOneDimensionalFastFourierTransform
        TestMultidimensionalFastFourierTransform
        TestQuadratureTableCalculator
        TestNumericalIntegrator
        TestBoundingBall
        TestBasicVectorOperations
        TestBasicMatrixOperations
        TestVectorMath
        TestKrylovSolvers
#        TestMultipoleCalculators
#        ConvertFieldMapToROOT
#        TestM2LCorrelation
#        TestM2LMemorySavings
        TestPinchonJMatrixCalculator
    )


    if (KEMField_USE_KGEOBAG)

        add_executable (TestFastMultipoleFieldMap
        ${CMAKE_CURRENT_SOURCE_DIR}/TestFastMultipoleFieldMap.cc)
        target_link_libraries (TestFastMultipoleFieldMap ${FAST_MULTIPOLE_TEST_LIBS} )

        add_executable (TestFastMultipoleFieldSolver
        ${CMAKE_CURRENT_SOURCE_DIR}/TestFastMultipoleFieldSolver.cc)
        target_link_libraries (TestFastMultipoleFieldSolver ${FAST_MULTIPOLE_TEST_LIBS} )

#        add_executable (TestFastMultipoleDualTree
#        ${CMAKE_CURRENT_SOURCE_DIR}/TestFastMultipoleDualTree.cc)
#        target_link_libraries (TestFastMultipoleDualTree ${FAST_MULTIPOLE_TEST_LIBS} )

#        add_executable (TestFastMultipoleChargeDensitySolver
#        ${CMAKE_CURRENT_SOURCE_DIR}/TestFastMultipoleChargeDensitySolver.cc)
#        target_link_libraries (TestFastMultipoleChargeDensitySolver ${FAST_MULTIPOLE_TEST_LIBS} )

#        add_executable (TestFastMultipoleSphereCapacitance
#        ${CMAKE_CURRENT_SOURCE_DIR}/TestFastMultipoleSphereCapacitance.cc)
#        target_link_libraries (TestFastMultipoleSphereCapacitance ${FAST_MULTIPOLE_TEST_LIBS} )

#        add_executable (ComputeSphericalCapacitor_FastMultipole
#        ${CMAKE_CURRENT_SOURCE_DIR}/ComputeSphericalCapacitor_FastMultipole.cc)
#        target_link_libraries (ComputeSphericalCapacitor_FastMultipole ${FAST_MULTIPOLE_TEST_LIBS} )

#        add_executable (TestFastMultipoleGeometryAnalysis
#        ${CMAKE_CURRENT_SOURCE_DIR}/TestFastMultipoleGeometryAnalysis.cc)
#        target_link_libraries (TestFastMultipoleGeometryAnalysis ${FAST_MULTIPOLE_TEST_LIBS} )

        if (@PROJECT_NAME@_USE_OPENCL)

#            add_executable (TestMultipoleBatchCalculatorOpenCL
#            ${CMAKE_CURRENT_SOURCE_DIR}/TestMultipoleBatchCalculatorOpenCL.cc)
#            target_link_libraries (TestMultipoleBatchCalculatorOpenCL ${FAST_MULTIPOLE_TEST_LIBS} )

#            add_executable (TestMultidimensionalFastFourierTransformOpenCL
#            ${CMAKE_CURRENT_SOURCE_DIR}/TestMultidimensionalFastFourierTransformOpenCL.cc)
#            target_link_libraries (TestMultidimensionalFastFourierTransformOpenCL ${FAST_MULTIPOLE_TEST_LIBS} )

#            add_executable (TestMultidimensionalFastFourierTransformSpeed
#            ${CMAKE_CURRENT_SOURCE_DIR}/TestMultidimensionalFastFourierTransformSpeed.cc)
#            target_link_libraries (TestMultidimensionalFastFourierTransformSpeed ${FAST_MULTIPOLE_TEST_LIBS} )

#              kasper_install_executables (
#                    TestMultipoleBatchCalculatorOpenCL
#                    TestMultidimensionalFastFourierTransformOpenCL
#                    TestMultidimensionalFastFourierTransformSpeed
#                )

        endif (@PROJECT_NAME@_USE_OPENCL)



        if (@PROJECT_NAME@_USE_VTK)

#            add_executable (ConvertFieldMapToVTK
#            ${CMAKE_CURRENT_SOURCE_DIR}/ConvertFieldMapToVTK.cc)
#            target_link_libraries (ConvertFieldMapToVTK ${FAST_MULTIPOLE_TEST_LIBS} )

#              kasper_install_executables (
#                    ConvertFieldMapToVTK
#                )

        endif (@PROJECT_NAME@_USE_VTK)


        kasper_install_executables (
             TestFastMultipoleFieldMap
             TestFastMultipoleFieldSolver
             #TestFastMultipoleDualTree
#             TestFastMultipoleChargeDensitySolver
             #TestFastMultipoleSphereCapacitance
#             TestFastMultipoleGeometryAnalysis
#             ComputeSphericalCapacitor_FastMultipole
        )


    endif (KEMField_USE_KGEOBAG)

endif (@PROJECT_NAME@_ENABLE_FM_TEST)
