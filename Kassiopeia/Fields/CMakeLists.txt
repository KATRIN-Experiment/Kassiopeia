# header files
set( FIELDS_HEADER_BASENAMES
    KSElectricKEMField.h
    KSFieldsMessage.h
    KSExampleFields.h
    KSFieldFinder.h
    KSMagneticKEMField.h

)

set( FIELDS_HEADER_PATH
    ${CMAKE_CURRENT_SOURCE_DIR}/Include
)
foreach( BASENAME ${FIELDS_HEADER_BASENAMES} )
    list( APPEND FIELDS_HEADER_FILENAMES ${FIELDS_HEADER_PATH}/${BASENAME} )
endforeach( BASENAME )

# source files
set( FIELDS_SOURCE_BASENAMES
    KSElectricKEMField.cxx
    KSFieldsMessage.cxx
    KSExampleFields.cxx
    KSFieldFinder.cxx
    KSMagneticKEMField.cxx

)

set( FIELDS_SOURCE_PATH
    ${CMAKE_CURRENT_SOURCE_DIR}/Source
)
foreach( BASENAME ${FIELDS_SOURCE_BASENAMES} )
    list( APPEND FIELDS_SOURCE_FILENAMES ${FIELDS_SOURCE_PATH}/${BASENAME} )
endforeach( BASENAME )

# library
add_library( KassiopeiaFields SHARED ${FIELDS_SOURCE_FILENAMES} ${FIELDS_HEADER_FILENAMES} )

target_include_directories(KassiopeiaFields
    PUBLIC $<BUILD_INTERFACE:${FIELDS_HEADER_PATH}> $<INSTALL_INTERFACE:include> )
target_link_libraries( KassiopeiaFields
    PUBLIC
        KassiopeiaOperators
        KEMFieldsElectric
        KEMFieldsMagnetic
)

# install
kasper_install_headers( ${FIELDS_HEADER_FILENAMES} )
kasper_install_libraries( KassiopeiaFields )
