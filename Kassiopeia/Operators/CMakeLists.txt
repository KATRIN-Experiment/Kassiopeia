# header files
set( OPERATORS_HEADER_BASENAMES
    KSEventModifier.h
	KSOperatorsMessage.h
	KSParticle.h
	KSParticleFactory.h
	KSMagneticField.h
	KSElectricField.h
	KSGenerator.h
	KSTrajectory.h
	KSSpace.h
	KSSpaceInteraction.h
	KSSpaceNavigator.h
	KSSurface.h
	KSSurfaceInteraction.h
	KSSurfaceNavigator.h
    KSSide.h
	KSStepModifier.h
    KSTrackModifier.h
    KSEventModifier.h
    KSRunModifier.h
	KSTerminator.h
	KSWriter.h
)
set( OPERATORS_HEADER_PATH
	${CMAKE_CURRENT_SOURCE_DIR}/Include
)
foreach( BASENAME ${OPERATORS_HEADER_BASENAMES} )
	list( APPEND OPERATORS_HEADER_FILENAMES ${OPERATORS_HEADER_PATH}/${BASENAME} )
endforeach( BASENAME )

# source files
set( OPERATORS_SOURCE_BASENAMES
    KSEventModifier.cxx
	KSOperatorsMessage.cxx
	KSParticle.cxx
	KSParticleFactory.cxx
	KSMagneticField.cxx
	KSElectricField.cxx
	KSGenerator.cxx
	KSTrajectory.cxx
	KSSpace.cxx
    KSSpaceInteraction.cxx
    KSSpaceNavigator.cxx
    KSSurface.cxx
	KSSurfaceInteraction.cxx
	KSSurfaceNavigator.cxx
    KSSide.cxx
	KSStepModifier.cxx
    KSTrackModifier.cxx
    KSEventModifier.cxx
    KSRunModifier.cxx
	KSTerminator.cxx
	KSWriter.cxx
)
set( OPERATORS_SOURCE_PATH
	${CMAKE_CURRENT_SOURCE_DIR}/Source
)
foreach( BASENAME ${OPERATORS_SOURCE_BASENAMES} )
	list( APPEND OPERATORS_SOURCE_FILENAMES ${OPERATORS_SOURCE_PATH}/${BASENAME} )
endforeach( BASENAME )

# library
kasper_internal_include_directories( ${OPERATORS_HEADER_PATH} )
add_library( KassiopeiaOperators SHARED ${OPERATORS_SOURCE_FILENAMES} )
set_target_properties( KassiopeiaOperators PROPERTIES COMPILE_FLAGS -Wno-overloaded-virtual )
target_link_libraries( KassiopeiaOperators
    ${Kommon_LIBRARIES}
    ${KGeoBag_LIBRARIES}
    KassiopeiaUtility
    KassiopeiaObjects
)

# install
kasper_install_headers( ${OPERATORS_HEADER_FILENAMES} )
kasper_install_libraries( KassiopeiaOperators )
