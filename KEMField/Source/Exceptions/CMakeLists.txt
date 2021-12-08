# header files
set( EXCEPTIONS_HEADER_BASENAMES
    KKeyNotFoundException.hh
    KEMSimpleException.hh
)

set( EXCEPTIONS_HEADER_PATH
    ${CMAKE_CURRENT_SOURCE_DIR}/include
)

foreach( BASENAME ${EXCEPTIONS_HEADER_BASENAMES} )
    list( APPEND EXCEPTIONS_HEADER_FILENAMES ${EXCEPTIONS_HEADER_PATH}/${BASENAME} )
endforeach( BASENAME )

# source files
set( EXCEPTIONS_SOURCE_BASENAMES
    KKeyNotFoundException.cc
    KEMSimpleException.cc
)

set( EXCEPTIONS_SOURCE_PATH
    ${CMAKE_CURRENT_SOURCE_DIR}/src
)
foreach( BASENAME ${EXCEPTIONS_SOURCE_BASENAMES} )
    list( APPEND EXCEPTIONS_SOURCE_FILENAMES ${EXCEPTIONS_SOURCE_PATH}/${BASENAME} )
endforeach( BASENAME )

# library
add_library( KEMFieldExceptions SHARED
    ${EXCEPTIONS_SOURCE_FILENAMES} ${EXCEPTIONS_HEADER_FILENAMES} )
target_include_directories(KEMFieldExceptions
    PUBLIC $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include> $<INSTALL_INTERFACE:include>)
target_link_libraries( KEMFieldExceptions
    PUBLIC
        KommonBase
)
#set_target_properties( KEMFieldExceptions PROPERTIES COMPILE_FLAGS -Wno-overloaded-virtual )

# install
kasper_install_headers( ${EXCEPTIONS_HEADER_FILENAMES} )
kasper_install_libraries( KEMFieldExceptions )
