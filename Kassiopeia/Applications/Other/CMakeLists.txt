set( APPLICATIONS_OTHER_BASENAMES
)

if( Kassiopeia_USE_ROOT )
    list( APPEND APPLICATIONS_OTHER_BASENAMES
        ROOTFileMerge
    )
    if( Kommon_USE_BOOST )
        list( APPEND APPLICATIONS_OTHER_BASENAMES
            TrackCompare
        )
    endif( Kommon_USE_BOOST )
endif( Kassiopeia_USE_ROOT )



set( APPLICATIONS_OTHER_LIBRARIES
    ${Kommon_LIBRARIES}
    ${KGeoBag_LIBRARIES}
    ${KEMField_LIBRARIES}
    KassiopeiaUtility
    KassiopeiaMath
    KassiopeiaObjects
    KassiopeiaOperators
    KassiopeiaFields
    KassiopeiaTrajectories
    KassiopeiaInteractions
    KassiopeiaNavigators
    KassiopeiaTerminators
    KassiopeiaReaders
    KassiopeiaWriters
    KassiopeiaSimulation
)
set( APPLICATIONS_OTHER_PATH
	${CMAKE_CURRENT_SOURCE_DIR}/Source
)

kasper_internal_include_directories()
foreach( BASENAME ${APPLICATIONS_OTHER_BASENAMES} )
	add_executable( ${BASENAME} ${APPLICATIONS_OTHER_PATH}/${BASENAME}.cxx )
	target_link_libraries( ${BASENAME} ${APPLICATIONS_OTHER_LIBRARIES} )
endforeach( BASENAME )

kasper_install_executables( ${APPLICATIONS_OTHER_BASENAMES} )
