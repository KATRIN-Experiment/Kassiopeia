option (Kassiopeia_ENABLE_APP "Build Kassiopeia applications" ON)
if (Kassiopeia_ENABLE_APP)

    set( APPLICATIONS_EXAMPLES_BASENAMES
        QuadrupoleTrapAnalysis
        MultiFileAnalysis
    )

    if( Kassiopeia_USE_ROOT )
        list( APPEND APPLICATIONS_EXAMPLES_BASENAMES
            TrackMagneticTransitionCheck
            DipoleTrapComparison
        )
    endif( Kassiopeia_USE_ROOT )

    set( APPLICATIONS_EXAMPLES_PATH
        ${CMAKE_CURRENT_SOURCE_DIR}/Source
    )

    foreach( BASENAME ${APPLICATIONS_EXAMPLES_BASENAMES} )
        add_executable( ${BASENAME} ${APPLICATIONS_EXAMPLES_PATH}/${BASENAME}.cxx )
        target_link_libraries( ${BASENAME} PUBLIC KassiopeiaBindings )
    endforeach( BASENAME )

    kasper_install_executables( ${APPLICATIONS_EXAMPLES_BASENAMES} )

endif(Kassiopeia_ENABLE_APP)
