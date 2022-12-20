option (KGeoBag_ENABLE_TEST "Build KGeoBag test applications" OFF)
if (KGeoBag_ENABLE_TEST)

    set (TEST_LIBS
        KGeoBagCore
        KGeoBagShapes
        KGeoBagMesh
    )

    if(KGeoBag_USE_VTK)
        list( APPEND TEST_LIBS
            KGeoBagVisualization
        )
    endif(KGeoBag_USE_VTK)

    #add_executable (TestKGBeam
    #${CMAKE_CURRENT_SOURCE_DIR}/TestKGBeam.cc)
    #target_link_libraries (TestKGBeam ${TEST_LIBS})

    #kasper_install_executables (
    #TestKGBeam
    #)

    #add_executable (TestStructure
    #${CMAKE_CURRENT_SOURCE_DIR}/TestStructure.cc)
    #target_link_libraries (TestStructure ${TEST_LIBS})

    #add_executable (TestCylinders
    #${CMAKE_CURRENT_SOURCE_DIR}/TestCylinders.cc)
    #target_link_libraries (TestCylinders ${TEST_LIBS})

    #add_executable (TestEMCylinders
    #${CMAKE_CURRENT_SOURCE_DIR}/TestEMCylinders.cc)
    #target_link_libraries (TestEMCylinders ${TEST_LIBS})

    #add_executable (TestInterpolation
    #${CMAKE_CURRENT_SOURCE_DIR}/TestInterpolation.cc)
    #target_link_libraries (TestInterpolation ${TEST_LIBS} KGeoBagKatrin)

    #kasper_install_executables (
    #    TestStructure
    #    TestCylinders
    #    TestEMCylinders
    #    TestInterpolation
    #)

endif(KGeoBag_ENABLE_TEST)
