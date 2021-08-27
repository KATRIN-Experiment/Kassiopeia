# header files
set( OBJECTS_HEADER_BASENAMES
    KSObjectsMessage.h
    KSObject.h
    KSCommand.h
    KSCommandGroup.h
    KSComponent.h
    KSComponentGroup.h
    KSDictionary.h
    KSCommandMember.h
    KSComponentMember.h
    KSComponentValue.h
    KSComponentDelta.h
    KSComponentIntegral.h
    KSComponentMaximum.h
    KSComponentMinimum.h
    KSComponentMaximumAt.h
    KSComponentMinimumAt.h
    KSComponentTemplate.h
)

if( Kassiopeia_USE_ROOT )
    list( APPEND OBJECTS_HEADER_BASENAMES
        KSComponentMath.h
    )
endif( Kassiopeia_USE_ROOT )


set( OBJECTS_HEADER_PATH
    ${CMAKE_CURRENT_SOURCE_DIR}/Include
)

foreach( BASENAME ${OBJECTS_HEADER_BASENAMES} )
    list( APPEND OBJECTS_HEADER_FILENAMES ${OBJECTS_HEADER_PATH}/${BASENAME} )
endforeach( BASENAME )

# source files
set( OBJECTS_SOURCE_BASENAMES
    KSObjectsMessage.cxx
    KSObject.cxx
    KSCommand.cxx
    KSCommandGroup.cxx
    KSComponent.cxx
    KSComponentGroup.cxx
    KSDictionary.cxx
)

set( OBJECTS_SOURCE_PATH
    ${CMAKE_CURRENT_SOURCE_DIR}/Source
)

foreach( BASENAME ${OBJECTS_SOURCE_BASENAMES} )
    list( APPEND OBJECTS_SOURCE_FILENAMES ${OBJECTS_SOURCE_PATH}/${BASENAME} )
endforeach( BASENAME )

# library
add_library( KassiopeiaObjects SHARED
    ${OBJECTS_SOURCE_FILENAMES} ${OBJECTS_HEADER_FILENAMES} )
target_include_directories(KassiopeiaObjects
    PUBLIC $<BUILD_INTERFACE:${OBJECTS_HEADER_PATH}> $<INSTALL_INTERFACE:include> )
target_link_libraries( KassiopeiaObjects
    PUBLIC
        KassiopeiaUtility
)

# install
kasper_install_headers( ${OBJECTS_HEADER_FILENAMES} )
kasper_install_libraries( KassiopeiaObjects )
