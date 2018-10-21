# header files
set( TERMINATORS_HEADER_BASENAMES
    KSTerminatorsMessage.h

    KSTermDeath.h
    KSTermMinZ.h
    KSTermMaxZ.h
    KSTermMinR.h
    KSTermMaxR.h
    KSTermMinEnergy.h
    KSTermMaxEnergy.h
    KSTermMinLongEnergy.h
    KSTermMaxLongEnergy.h
    KSTermMaxTime.h
    KSTermMaxLength.h
    KSTermMaxSteps.h
    KSTermTrapped.h
    KSTermSecondaries.h
    KSTermOutput.h
    KSTermMinDistance.h
    KSTermStepsize.h
    KSTermMagnetron.h
)
set( TERMINATORS_HEADER_PATH 
	${CMAKE_CURRENT_SOURCE_DIR}/Include
)
foreach( BASENAME ${TERMINATORS_HEADER_BASENAMES} )
	list( APPEND TERMINATORS_HEADER_FILENAMES ${TERMINATORS_HEADER_PATH}/${BASENAME} )
endforeach( BASENAME )

# source files
set( TERMINATORS_SOURCE_BASENAMES
    KSTerminatorsMessage.cxx

    KSTermDeath.cxx
    KSTermMinZ.cxx
    KSTermMaxZ.cxx
    KSTermMinR.cxx
    KSTermMaxR.cxx
    KSTermMinEnergy.cxx
    KSTermMaxEnergy.cxx
    KSTermMinLongEnergy.cxx
    KSTermMaxLongEnergy.cxx
    KSTermMaxTime.cxx
    KSTermMaxLength.cxx
    KSTermMaxSteps.cxx
    KSTermTrapped.cxx
    KSTermSecondaries.cxx
    KSTermMinDistance.cxx
    KSTermStepsize.cxx
    KSTermMagnetron.cxx
)
set( TERMINATORS_SOURCE_PATH 
	${CMAKE_CURRENT_SOURCE_DIR}/Source
)
foreach( BASENAME ${TERMINATORS_SOURCE_BASENAMES} )
	list( APPEND TERMINATORS_SOURCE_FILENAMES ${TERMINATORS_SOURCE_PATH}/${BASENAME} )
endforeach( BASENAME )

# data files
set( TERMINATORS_DATA_BASENAMES
)
set( TERMINATORS_DATA_PATH 
	${CMAKE_CURRENT_SOURCE_DIR}/Data
)
foreach( BASENAME ${TERMINATORS_DATA_BASENAMES} )
	list( APPEND TERMINATORS_DATA_FILENAMES ${TERMINATORS_DATA_PATH}/${BASENAME} )
endforeach( BASENAME )

# library
kasper_internal_include_directories( ${TERMINATORS_HEADER_PATH} )
add_library( KassiopeiaTerminators SHARED ${TERMINATORS_SOURCE_FILENAMES} )
set_target_properties( KassiopeiaTerminators PROPERTIES COMPILE_FLAGS -Wno-overloaded-virtual )
target_link_libraries( KassiopeiaTerminators
    ${Kommon_LIBRARIES}
    ${KGeoBag_LIBRARIES}
    KassiopeiaUtility
    KassiopeiaObjects
    KassiopeiaOperators
)

# install
kasper_install_headers( ${TERMINATORS_HEADER_FILENAMES} )
kasper_install_data( ${TERMINATORS_DATA_FILENAMES} )
kasper_install_libraries( KassiopeiaTerminators )
