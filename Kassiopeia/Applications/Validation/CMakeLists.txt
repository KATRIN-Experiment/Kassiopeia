option (Kassiopeia_ENABLE_APP "Build Kassiopeia applications" ON)
if (Kassiopeia_ENABLE_APP)

    # executables
    set( VALIDATION_SOURCE_BASENAMES
        TestField
        # TestTrajectory
        # TestGenerator
        # TestSpaceInteraction
        # TestInteractionArgon
    )

    if(Kassiopeia_USE_ROOT)
        list( APPEND VALIDATION_SOURCE_BASENAMES
            TestInteraction
            TestHydrogenInteraction
            TestIonInteraction
            TestArgonInteraction
            TestGlukhov
            TestZonalHarmonicsConvergence
            TestRampedField
            TestSynchrotron
        )
    endif(Kassiopeia_USE_ROOT)

    if(Kassiopeia_USE_VTK)
        if(KEMField_USE_VTK)
            list( APPEND VALIDATION_SOURCE_BASENAMES
                TestPotentialmap
            )
        endif(KEMField_USE_VTK)
    endif(Kassiopeia_USE_VTK)

    set( VALIDATION_SOURCE_PATH
        ${CMAKE_CURRENT_SOURCE_DIR}/Source
    )

    if( NOT BUILD_KSC )
        foreach( BASENAME ${VALIDATION_SOURCE_BASENAMES} )
            add_executable( ${BASENAME} ${VALIDATION_SOURCE_PATH}/${BASENAME}.cxx )
            target_link_libraries( ${BASENAME} PUBLIC KassiopeiaBindings )
        endforeach( BASENAME )

        kasper_install_executables( ${VALIDATION_SOURCE_BASENAMES} )
    endif( NOT BUILD_KSC )

endif(Kassiopeia_ENABLE_APP)
