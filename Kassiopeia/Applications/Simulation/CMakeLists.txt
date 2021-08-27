option (Kassiopeia_ENABLE_APP "Build Kassiopeia applications" ON)
if (Kassiopeia_ENABLE_APP)

    set( APPLICATIONS_MAIN_BASENAMES
        Kassiopeia
        ParticleGenerator
    )

    set( APPLICATIONS_MAIN_PATH
        ${CMAKE_CURRENT_SOURCE_DIR}/Source
    )

    if( NOT BUILD_KSC )
        foreach( BASENAME ${APPLICATIONS_MAIN_BASENAMES} )
            add_executable( ${BASENAME} ${APPLICATIONS_MAIN_PATH}/${BASENAME}.cxx )
            target_link_libraries( ${BASENAME} PUBLIC KassiopeiaBindings )
        endforeach( BASENAME )

        kasper_install_executables( ${APPLICATIONS_MAIN_BASENAMES} )
    endif( NOT BUILD_KSC )

endif(Kassiopeia_ENABLE_APP)
