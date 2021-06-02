option (Kassiopeia_ENABLE_APP "Build Kassiopeia applications" ON)
if (Kassiopeia_ENABLE_APP)

    set( APPLICATIONS_OTHER_BASENAMES )

    if( Kassiopeia_USE_ROOT )
        list( APPEND APPLICATIONS_OTHER_BASENAMES
            ROOTFileMerge
        )
        if( KASPER_USE_BOOST )
            find_package( Boost ${BOOST_MINIMUM_VERSION} REQUIRED COMPONENTS program_options)
            list( APPEND APPLICATIONS_OTHER_BASENAMES
                TrackCompare
            )
        endif()
    endif()

    set( APPLICATIONS_OTHER_LIBRARIES
        KassiopeiaBindings
    )

    if( KASPER_USE_BOOST )
        if( TARGET Boost::program_options )
            list( APPEND APPLICATIONS_OTHER_LIBRARIES
                Boost::program_options
            )
        endif()
    endif()

    set( APPLICATIONS_OTHER_PATH
        ${CMAKE_CURRENT_SOURCE_DIR}/Source
    )

    if( NOT BUILD_KSC )
        foreach( BASENAME ${APPLICATIONS_OTHER_BASENAMES} )
            add_executable( ${BASENAME} ${APPLICATIONS_OTHER_PATH}/${BASENAME}.cxx )
            target_link_libraries( ${BASENAME} PUBLIC ${APPLICATIONS_OTHER_LIBRARIES} )
        endforeach( BASENAME )

        kasper_install_executables( ${APPLICATIONS_OTHER_BASENAMES} )
    endif( NOT BUILD_KSC )

endif(Kassiopeia_ENABLE_APP)
