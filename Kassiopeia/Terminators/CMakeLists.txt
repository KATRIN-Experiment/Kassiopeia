# header files
set( TERMINATORS_HEADER_BASENAMES
    KSTerminatorsMessage.h

    KSTermDeath.h
    KSTermMinZ.h
    KSTermMaxZ.h
    KSTermMinR.h
    KSTermMaxR.h
        KSTermXRange.h
        KSTermYRange.h
        KSTermZRange.h
    KSTermMinEnergy.h
    KSTermMaxEnergy.h
    KSTermMinLongEnergy.h
    KSTermMaxLongEnergy.h
    KSTermMaxTime.h
    KSTermMaxTotalTime.h
    KSTermMaxStepTime.h
    KSTermMaxLength.h
    KSTermMaxSteps.h
    KSTermTrapped.h
    KSTermSecondaries.h
    KSTermOutput.h
    KSTermMinDistance.h
    KSTermStepsize.h
    KSTermMagnetron.h
    KSTermZHRadius.h
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
        KSTermXRange.cxx
        KSTermYRange.cxx
        KSTermZRange.cxx
    KSTermMinEnergy.cxx
    KSTermMaxEnergy.cxx
    KSTermMinLongEnergy.cxx
    KSTermMaxLongEnergy.cxx
    KSTermMaxTime.cxx
    KSTermMaxTotalTime.cxx
    KSTermMaxStepTime.cxx
    KSTermMaxLength.cxx
    KSTermMaxSteps.cxx
    KSTermTrapped.cxx
    KSTermSecondaries.cxx
    KSTermMinDistance.cxx
    KSTermStepsize.cxx
    KSTermMagnetron.cxx
    KSTermZHRadius.cxx
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
add_library( KassiopeiaTerminators SHARED
    ${TERMINATORS_SOURCE_FILENAMES} ${TERMINATORS_HEADER_FILENAMES} )
target_include_directories(KassiopeiaTerminators
    PUBLIC $<BUILD_INTERFACE:${TERMINATORS_HEADER_PATH}> $<INSTALL_INTERFACE:include> )
target_link_libraries( KassiopeiaTerminators
    PUBLIC
        KassiopeiaOperators
        KassiopeiaGeometry
        KassiopeiaFields
)

# install
kasper_install_headers( ${TERMINATORS_HEADER_FILENAMES} )
kasper_install_libraries( KassiopeiaTerminators )
kasper_install_data( ${TERMINATORS_DATA_FILENAMES} )
