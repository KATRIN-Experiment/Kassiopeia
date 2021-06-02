if(KGeoBag_USE_VTK)

    set (TESTMESHNAVIGATION_LIBS
        Kommon
        KommonVtk
        KGeoBagMath
        KGeoBagCore
        KGeoBagShapes
        KGeoBagAppearance
        KGeoBagMesh
        KGeoBagAxialMesh
        KGeoBagBindings
        KGeoBagRandom
        KGeoBagMetrics
        KGeoBagBasicVisualization
        KGeoBagVtkVisualization
    )

    add_executable (TestMeshNavigation
        ${CMAKE_CURRENT_SOURCE_DIR}/TestMeshNavigation.cc)
    target_link_libraries (TestMeshNavigation ${TESTMESHNAVIGATION_LIBS})

    kasper_install_executables (
        TestMeshNavigation
    )

endif(KGeoBag_USE_VTK)
