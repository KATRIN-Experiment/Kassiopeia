set( APPLICATIONS_OTHER_BASENAMES
)

if( Kassiopeia_USE_ROOT )
    list( APPEND APPLICATIONS_OTHER_BASENAMES
        ROOTFileMerge
    )
    if( KASPER_USE_BOOST )
        kasper_find_boost( 1.44.0 REQUIRED COMPONENTS program_options)
        list( APPEND APPLICATIONS_OTHER_BASENAMES
            TrackCompare
        )
    endif()
endif()

set( APPLICATIONS_OTHER_LIBRARIES
    ${Kommon_LIBRARIES}
    ${Kommon_Vtk_LIBRARIES}
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

if( KASPER_USE_BOOST )
    list( APPEND APPLICATIONS_OTHER_LIBRARIES
        ${Boost_LIBRARIES}
    )
endif()

set( APPLICATIONS_OTHER_PATH
	${CMAKE_CURRENT_SOURCE_DIR}/Source
)

kasper_internal_include_directories()
foreach( BASENAME ${APPLICATIONS_OTHER_BASENAMES} )
	add_executable( ${BASENAME} ${APPLICATIONS_OTHER_PATH}/${BASENAME}.cxx )
	target_link_libraries( ${BASENAME} ${APPLICATIONS_OTHER_LIBRARIES} )
endforeach( BASENAME )

kasper_install_executables( ${APPLICATIONS_OTHER_BASENAMES} )
