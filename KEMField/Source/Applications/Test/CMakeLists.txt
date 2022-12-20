# CMakeLists for KEMField/Tests
# Author: T.J. Corona

option (KEMField_ENABLE_TEST "Build KEMField test applications" OFF)
if (KEMField_ENABLE_TEST)
    enable_testing ()
endif (KEMField_ENABLE_TEST)

if (KEMField_ENABLE_TEST)

    set (TESTS_HEADERS
        include/KElectrostaticBoundaryIntegratorOptions.hh
        include/TestCubatureTriangleIntegrator.hh
        include/TestRWGTriangleIntegrator.hh
    )

    set (TESTS_LIBS
        KGeoBagInterface
        KEMCore
        KEMIO
        KEMFileManipulation
        KEMElectrostaticBoundaryIntegrals
        KEMElectromagnets
        KEMLinearAlgebraPreconditioner
        KEMZHSolver
        KEMVisualization
    )

    if (KEMField_USE_VTK)
        list (APPEND TESTS_LIBS KEMVTKPlugin)
    endif (KEMField_USE_VTK)

    if (KEMField_USE_ROOT)
        list (APPEND TESTS_LIBS KEMRootPlugin)
    endif (KEMField_USE_ROOT)

    if (KEMField_USE_OPENCL)
        list (APPEND TESTS_LIBS KEMOpenCLPlugin)
    endif (KEMField_USE_OPENCL)

    if (KEMField_USE_PETSc)
        list (APPEND TESTS_LIBS KEMPETScPlugin)
    endif (KEMField_USE_PETSc)

    include_directories(${CMAKE_CURRENT_SOURCE_DIR}/include)

    add_executable (TestElectromagnets
        ${CMAKE_CURRENT_SOURCE_DIR}/TestElectromagnets.cc)
    target_link_libraries (TestElectromagnets ${TESTS_LIBS} )

    add_executable (TestElectromagnetViewer
        ${CMAKE_CURRENT_SOURCE_DIR}/TestElectromagnetViewer.cc)
    target_link_libraries (TestElectromagnetViewer ${TESTS_LIBS} )

    add_executable (TestDiskCapacitance
        ${CMAKE_CURRENT_SOURCE_DIR}/TestDiskCapacitance.cc)
    target_link_libraries (TestDiskCapacitance ${TESTS_LIBS} )

    add_executable (TestEMFileInterface
        ${CMAKE_CURRENT_SOURCE_DIR}/TestEMFileInterface.cc)
    target_link_libraries (TestEMFileInterface ${TESTS_LIBS} )

    add_executable (TestGeometry
        ${CMAKE_CURRENT_SOURCE_DIR}/TestGeometry.cc)
    target_link_libraries (TestGeometry ${TESTS_LIBS} )

    add_executable (TestAnisotropicPoisson
        ${CMAKE_CURRENT_SOURCE_DIR}/TestAnisotropicPoisson.cc)
    target_link_libraries (TestAnisotropicPoisson ${TESTS_LIBS} )

    add_executable (TestCubatureSpeedStandAlone
        ${CMAKE_CURRENT_SOURCE_DIR}/TestCubatureSpeedStandAlone.cc)
    target_link_libraries (TestCubatureSpeedStandAlone ${TESTS_LIBS} )

    add_executable (TestIntegratorFunctionality
        ${CMAKE_CURRENT_SOURCE_DIR}/TestIntegratorFunctionality.cc)
    target_link_libraries (TestIntegratorFunctionality ${TESTS_LIBS} )

    add_executable (TestIntegratorSpeed
        ${CMAKE_CURRENT_SOURCE_DIR}/TestIntegratorSpeed.cc)
    target_link_libraries (TestIntegratorSpeed ${TESTS_LIBS} )

    add_executable (TestIntegratorSpeedWithKbdInput
        ${CMAKE_CURRENT_SOURCE_DIR}/TestIntegratorSpeedWithKbdInput.cc)
    target_link_libraries (TestIntegratorSpeedWithKbdInput ${TESTS_LIBS} )

    add_executable (TestIntegratorRWG
        ${CMAKE_CURRENT_SOURCE_DIR}/TestIntegratorRWG.cc)
    target_link_libraries (TestIntegratorRWG ${TESTS_LIBS} )

    add_executable (TestIntegratorRWGAccuracy
        ${CMAKE_CURRENT_SOURCE_DIR}/TestIntegratorRWGAccuracy.cc)
    target_link_libraries (TestIntegratorRWGAccuracy ${TESTS_LIBS} )

    add_executable (TestInverseDistance
        ${CMAKE_CURRENT_SOURCE_DIR}/TestInverseDistance.cc)
    target_link_libraries (TestInverseDistance ${TESTS_LIBS} )

    add_executable (TestSVDSolver
        ${CMAKE_CURRENT_SOURCE_DIR}/TestSVDSolver.cc)
    target_link_libraries (TestSVDSolver ${TESTS_LIBS} )

    add_executable (TestBinaryTruncation
        ${CMAKE_CURRENT_SOURCE_DIR}/TestBinaryTruncation.cc)
    target_link_libraries (TestBinaryTruncation ${TESTS_LIBS} )

    add_executable (TestTriangles
        ${CMAKE_CURRENT_SOURCE_DIR}/TestTriangles.cc)
    target_link_libraries (TestTriangles ${TESTS_LIBS}  )

    add_executable (TestTypelists
        ${CMAKE_CURRENT_SOURCE_DIR}/TestTypelists.cc)
    target_link_libraries (TestTypelists ${TESTS_LIBS}  )

    add_executable (TestVisitor
        ${CMAKE_CURRENT_SOURCE_DIR}/TestVisitor.cc)
    target_link_libraries (TestVisitor ${TESTS_LIBS}  )

    add_executable (TestZonalHarmonics
        ${CMAKE_CURRENT_SOURCE_DIR}/TestZonalHarmonics.cc)
    target_link_libraries (TestZonalHarmonics ${TESTS_LIBS}  )

    add_executable (TestCubeCapacitance
        ${CMAKE_CURRENT_SOURCE_DIR}/TestCubeCapacitance.cc)
    target_link_libraries (TestCubeCapacitance ${TESTS_LIBS})

    add_executable (TestSphereCapacitance
        ${CMAKE_CURRENT_SOURCE_DIR}/TestSphereCapacitance.cc)
    target_link_libraries (TestSphereCapacitance ${TESTS_LIBS})

    add_executable (TestSuperposition
        ${CMAKE_CURRENT_SOURCE_DIR}/TestSuperposition.cc)
    target_link_libraries (TestSuperposition ${TESTS_LIBS} )

    add_executable (TestSymmetryGroups
        ${CMAKE_CURRENT_SOURCE_DIR}/TestSymmetryGroups.cc)
    target_link_libraries (TestSymmetryGroups ${TESTS_LIBS} )

#    add_executable (TestSphericalImageCharge
#      ${CMAKE_CURRENT_SOURCE_DIR}/TestSphericalImageCharge.cc)
#    target_link_libraries (TestSphericalImageCharge ${TESTS_LIBS})

    kasper_install_executables (
        TestCubatureSpeedStandAlone
        TestDiskCapacitance
        TestElectromagnets
        TestElectromagnetViewer
        TestEMFileInterface
        TestGeometry
        TestAnisotropicPoisson
        TestIntegratorFunctionality
        TestIntegratorSpeed
        TestIntegratorSpeedWithKbdInput
        TestIntegratorRWG
        TestIntegratorRWGAccuracy
        TestInverseDistance
        TestSVDSolver
        TestBinaryTruncation
        TestTriangles
        TestTypelists
        TestVisitor
        TestZonalHarmonics
        TestCubeCapacitance
        TestSphereCapacitance
        TestSuperposition
        TestSymmetryGroups
        # TestSphericalImageCharge
    )

    ############
    ### ROOT ###
    ############

    if (KEMField_USE_ROOT)

    add_executable (TestDielectrics
        ${CMAKE_CURRENT_SOURCE_DIR}/TestDielectrics.cc)
    target_link_libraries (TestDielectrics ${TESTS_LIBS} )

    add_executable (TestIntegratorDistRatioLineSegmentROOT
        ${CMAKE_CURRENT_SOURCE_DIR}/TestIntegratorDistRatioLineSegmentROOT.cc)
    target_link_libraries (TestIntegratorDistRatioLineSegmentROOT ${TESTS_LIBS} )

    add_executable (TestIntegratorDistRatioRectangleROOT
        ${CMAKE_CURRENT_SOURCE_DIR}/TestIntegratorDistRatioRectangleROOT.cc)
    target_link_libraries (TestIntegratorDistRatioRectangleROOT ${TESTS_LIBS} )

    add_executable (TestIntegratorDistRatioTriangleROOT
        ${CMAKE_CURRENT_SOURCE_DIR}/TestIntegratorDistRatioTriangleROOT.cc)
    target_link_libraries (TestIntegratorDistRatioTriangleROOT ${TESTS_LIBS} )

    add_executable (TestVectorTypes
        ${CMAKE_CURRENT_SOURCE_DIR}/TestVectorTypes.cc)
    target_link_libraries (TestVectorTypes ${TESTS_LIBS} )

    kasper_install_executables (
        TestDielectrics
        TestIntegratorDistRatioLineSegmentROOT
        TestIntegratorDistRatioRectangleROOT
        TestIntegratorDistRatioTriangleROOT
        TestVectorTypes
    )

    endif (KEMField_USE_ROOT)

    ################
    ### KMessage ###
    ################

    #if (KEMField_USE_KMESSAGE)

    add_executable (TestKMessageInterface
        ${CMAKE_CURRENT_SOURCE_DIR}/TestKMessageInterface.cc)
    target_link_libraries (TestKMessageInterface ${TESTS_LIBS})

    kasper_install_executables (
        TestKMessageInterface
    )

    #endif (KEMField_USE_KMESSAGE)

    ####################
    ### OpenCL + MPI ###
    ####################

    if (KEMField_USE_OPENCL AND KEMField_USE_MPI)

        # add_executable (TestOpenCL
        #     ${CMAKE_CURRENT_SOURCE_DIR}/TestOpenCL.cc)
        # target_link_libraries (TestOpenCL ${TESTS_LIBS} )

        # kasper_install_executables (
        #     TestOpenCL
        # )

    endif (KEMField_USE_OPENCL AND KEMField_USE_MPI)

    ##############
    ### OpenCL ###
    ##############

    if (KEMField_USE_OPENCL)

        if (KEMField_USE_ROOT)

            add_executable (TestIntegratorTriangleROOTOpenCL
                ${CMAKE_CURRENT_SOURCE_DIR}/TestIntegratorTriangleROOTOpenCL.cc)
            target_link_libraries (TestIntegratorTriangleROOTOpenCL ${TESTS_LIBS} )

            add_executable (TestIntegratorRectangleROOTOpenCL
                ${CMAKE_CURRENT_SOURCE_DIR}/TestIntegratorRectangleROOTOpenCL.cc)
            target_link_libraries (TestIntegratorRectangleROOTOpenCL ${TESTS_LIBS} )

            add_executable (TestIntegratorLineSegmentROOTOpenCL
                ${CMAKE_CURRENT_SOURCE_DIR}/TestIntegratorLineSegmentROOTOpenCL.cc)
            target_link_libraries (TestIntegratorLineSegmentROOTOpenCL ${TESTS_LIBS} )

            kasper_install_executables (
                TestIntegratorTriangleROOTOpenCL
                TestIntegratorRectangleROOTOpenCL
                TestIntegratorLineSegmentROOTOpenCL
            )

        endif (KEMField_USE_ROOT)

        add_executable (TestOpenCLPlugin
            ${CMAKE_CURRENT_SOURCE_DIR}/TestOpenCLPlugin.cc)
        target_link_libraries (TestOpenCLPlugin ${TESTS_LIBS} )

        add_executable (TestIntegratorFunctionalityOpenCL
            ${CMAKE_CURRENT_SOURCE_DIR}/TestIntegratorFunctionalityOpenCL.cc)
        target_link_libraries (TestIntegratorFunctionalityOpenCL ${TESTS_LIBS} )

        kasper_install_executables (
            TestIntegratorFunctionalityOpenCL
            TestOpenCLPlugin
        )

    endif (KEMField_USE_OPENCL)

    #############
    ### PETSc ###
    #############

    if (KEMField_USE_PETSc)

        set_property(SOURCE ${CMAKE_CURRENT_SOURCE_DIR}/TestPETSc.cc
            APPEND PROPERTY COMPILE_DEFINITIONS DEFAULT_DATA_DIR="${DATA_INSTALL_DIR}/${KEMField_DATA_DIR}"
        )

        add_executable (TestPETSc
            ${CMAKE_CURRENT_SOURCE_DIR}/TestPETSc.cc)
        target_link_libraries (TestPETSc ${TESTS_LIBS})

        add_executable (TestPETSc1
            ${CMAKE_CURRENT_SOURCE_DIR}/TestPETSc1.cc)
        target_link_libraries (TestPETSc1 ${TESTS_LIBS})

        add_executable (TestPETSc2
            ${CMAKE_CURRENT_SOURCE_DIR}/TestPETSc2.cc)
        target_link_libraries (TestPETSc2 ${TESTS_LIBS})

        kasper_install_executables (
            TestPETSc
            TestPETSc1
            TestPETSc2
        )

    endif (KEMField_USE_PETSc)

endif (KEMField_ENABLE_TEST)
