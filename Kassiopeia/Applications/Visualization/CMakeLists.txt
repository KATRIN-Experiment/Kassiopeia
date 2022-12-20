option (Kassiopeia_ENABLE_APP "Build Kassiopeia applications" ON)
if (Kassiopeia_ENABLE_APP)

    set( APPLICATIONS_VISUALIZATION_BASENAMES )

    if( Kassiopeia_USE_VTK )
        list( APPEND APPLICATIONS_VISUALIZATION_BASENAMES
            TrackViewer
        )
    endif( Kassiopeia_USE_VTK )

    if( Kassiopeia_USE_ROOT )
        list( APPEND APPLICATIONS_VISUALIZATION_BASENAMES
            ZonalHarmonicViewerROOT
        )
    endif( Kassiopeia_USE_ROOT )

    set( APPLICATIONS_VISUALIZATION_PATH
        ${CMAKE_CURRENT_SOURCE_DIR}/Source
    )

    if( NOT BUILD_KSC )
        foreach( BASENAME ${APPLICATIONS_VISUALIZATION_BASENAMES} )
            add_executable( ${BASENAME} ${APPLICATIONS_VISUALIZATION_PATH}/${BASENAME}.cxx )
            target_link_libraries( ${BASENAME}
                KassiopeiaBindings KGeoBagBindings KEMFieldBindings )
        endforeach( BASENAME )

        kasper_install_executables( ${APPLICATIONS_VISUALIZATION_BASENAMES} )
    endif( NOT BUILD_KSC )

endif(Kassiopeia_ENABLE_APP)
