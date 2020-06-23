set( MAIN_BASENAMES
    Kassiopeia
    ParticleGenerator
)

set( MAIN_LIBRARIES
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
    KassiopeiaTerminators
    KassiopeiaWriters
    KassiopeiaVisualization
    KassiopeiaReaders
    KassiopeiaSimulation
    KassiopeiaBindings
)

set( MAIN_PATH
	${CMAKE_CURRENT_SOURCE_DIR}/Source
)

kasper_internal_include_directories()
foreach( BASENAME ${MAIN_BASENAMES} )
	add_executable( ${BASENAME} ${MAIN_PATH}/${BASENAME}.cxx )
	target_link_libraries( ${BASENAME} ${MAIN_LIBRARIES} )
endforeach( BASENAME )

kasper_install_executables( ${MAIN_BASENAMES} )
