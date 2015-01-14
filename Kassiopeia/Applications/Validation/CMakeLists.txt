# executables
set( VALIDATION_SOURCE_BASENAMES
#	TestField
#	TestGenerator
#	TestTrajectory
#	TestSpaceInteraction
#	TestInteractionArgon    
)

if(Kassiopeia_USE_ROOT)
	list( APPEND VALIDATION_SOURCE_BASENAMES
		TestInteraction
        TestHydrogenInteraction
        TestArgonInteraction
        TestZonalHarmonicsConvergence
	)
endif(Kassiopeia_USE_ROOT)

set( VALIDATION_SOURCE_LIBRARIES
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
    KassiopeiaTerminators
    KassiopeiaWriters
    KassiopeiaVisualization
    KassiopeiaReaders
    KassiopeiaSimulation
    KassiopeiaBindings
)	
set( VALIDATION_SOURCE_PATH 
	${CMAKE_CURRENT_SOURCE_DIR}/Source
)
kasper_internal_include_directories()
foreach( BASENAME ${VALIDATION_SOURCE_BASENAMES} )
	add_executable( ${BASENAME} ${VALIDATION_SOURCE_PATH}/${BASENAME}.cxx )
	target_link_libraries( ${BASENAME} ${VALIDATION_SOURCE_LIBRARIES} )
endforeach( BASENAME )

kasper_install_executables( ${VALIDATION_SOURCE_BASENAMES} )
