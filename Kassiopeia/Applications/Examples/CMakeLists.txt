set( APPLICATIONS_EXAMPLES_BASENAMES
	QuadrupoleTrapAnalysis
)
set( APPLICATIONS_EXAMPLES_LIBRARIES )

if( Kassiopeia_USE_ROOT )
list( APPEND APPLICATIONS_EXAMPLES_BASENAMES
        TrackMagneticTransitionCheck
        DipoleTrapComparison
        )

list( APPEND APPLICATIONS_EXAMPLES_LIBRARIES
        ${ROOT_LIBRARIES}
)
endif( Kassiopeia_USE_ROOT )

list( APPEND APPLICATIONS_EXAMPLES_LIBRARIES
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
    KassiopeiaVisualization
    KassiopeiaSimulation
)
set( APPLICATIONS_EXAMPLES_PATH
	${CMAKE_CURRENT_SOURCE_DIR}/Source
)
kasper_internal_include_directories()
foreach( BASENAME ${APPLICATIONS_EXAMPLES_BASENAMES} )
	add_executable( ${BASENAME} ${APPLICATIONS_EXAMPLES_PATH}/${BASENAME}.cxx )
	target_link_libraries( ${BASENAME} ${APPLICATIONS_EXAMPLES_LIBRARIES} )
endforeach( BASENAME )

kasper_install_executables( ${APPLICATIONS_EXAMPLES_BASENAMES} )
