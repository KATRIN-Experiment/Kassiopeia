# header files
set( SIMULATION_HEADER_BASENAMES
    KSMainMessage.h
    KSRunMessage.h
    KSEventMessage.h
    KSTrackMessage.h
    KSStepMessage.h
	KSSimulation.h
	KSRun.h
	KSEvent.h
	KSTrack.h
	KSStep.h
    KSRootElectricField.h
    KSRootMagneticField.h
    KSRootStepModifier.h
    KSRootSpace.h
    KSRootGenerator.h
    KSRootTrajectory.h
    KSRootSpaceInteraction.h
    KSRootSurfaceInteraction.h
    KSRootSpaceNavigator.h
    KSRootSurfaceNavigator.h
    KSRootTerminator.h
    KSRootWriter.h
    KSRoot.h
)
set( SIMULATION_HEADER_PATH 
	${CMAKE_CURRENT_SOURCE_DIR}/Include
)
foreach( BASENAME ${SIMULATION_HEADER_BASENAMES} )
	list( APPEND SIMULATION_HEADER_FILENAMES ${SIMULATION_HEADER_PATH}/${BASENAME} )
endforeach( BASENAME )

# source files
set( SIMULATION_SOURCE_BASENAMES
    KSMainMessage.cxx
    KSRunMessage.cxx
    KSEventMessage.cxx
    KSTrackMessage.cxx
    KSStepMessage.cxx
    KSSimulation.cxx
	KSRun.cxx
	KSEvent.cxx
	KSTrack.cxx
	KSStep.cxx
    KSRootElectricField.cxx
    KSRootMagneticField.cxx
    KSRootStepModifier.cxx
    KSRootSpace.cxx
    KSRootGenerator.cxx
    KSRootTrajectory.cxx
    KSRootSpaceInteraction.cxx
    KSRootSurfaceInteraction.cxx
    KSRootSpaceNavigator.cxx
    KSRootSurfaceNavigator.cxx
    KSRootTerminator.cxx
    KSRootWriter.cxx
    KSRoot.cxx
)
set( SIMULATION_SOURCE_PATH 
	${CMAKE_CURRENT_SOURCE_DIR}/Source
)
foreach( BASENAME ${SIMULATION_SOURCE_BASENAMES} )
	list( APPEND SIMULATION_SOURCE_FILENAMES ${SIMULATION_SOURCE_PATH}/${BASENAME} )
endforeach( BASENAME )

# data files
set( SIMULATION_DATA_BASENAMES
)
set( SIMULATION_DATA_PATH 
	${CMAKE_CURRENT_SOURCE_DIR}/DataFiles
)
foreach( BASENAME ${SIMULATION_DATA_BASENAMES} )
	list( APPEND SIMULATION_DATA_FILENAMES ${SIMULATION_DATA_PATH}/${BASENAME} )
endforeach( BASENAME )

# library
kasper_internal_include_directories( ${SIMULATION_HEADER_PATH} )
add_library( KassiopeiaSimulation SHARED ${SIMULATION_SOURCE_FILENAMES} )
set_target_properties( KassiopeiaSimulation PROPERTIES COMPILE_FLAGS -Wno-overloaded-virtual )
target_link_libraries( KassiopeiaSimulation
    ${Kommon_LIBRARIES}
    ${KGeoBag_LIBRARIES}
    KassiopeiaUtility
    KassiopeiaObjects
    KassiopeiaOperators
    KassiopeiaFields
    KassiopeiaGeometry
    KassiopeiaGenerators
    KassiopeiaTrajectories
    KassiopeiaInteractions
    KassiopeiaNavigators
    KassiopeiaTerminators
    KassiopeiaWriters
    KassiopeiaModifiers
    #KassiopeiaReaders
)

# install
kasper_install_headers( ${SIMULATION_HEADER_FILENAMES} )
kasper_install_data( ${SIMULATION_DATA_FILENAMES} )
kasper_install_libraries( KassiopeiaSimulation )
