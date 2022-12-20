# header files
set( READERS_HEADER_BASENAMES
    KSReadersMessage.h

    KSReadValue.h
    KSReadSet.h
    KSReadIterator.h
    KSReadFile.h
)

if( Kassiopeia_USE_ROOT )
    list( APPEND READERS_HEADER_BASENAMES
        KSReadIteratorROOT.h
        KSReadRunROOT.h
        KSReadEventROOT.h
        KSReadTrackROOT.h
        KSReadStepROOT.h
        KSReadObjectROOT.h
        KSReadFileROOT.h
    )
endif( Kassiopeia_USE_ROOT )

set( READERS_HEADER_PATH
    ${CMAKE_CURRENT_SOURCE_DIR}/Include
)
foreach( BASENAME ${READERS_HEADER_BASENAMES} )
    list( APPEND READERS_HEADER_FILENAMES ${READERS_HEADER_PATH}/${BASENAME} )
endforeach( BASENAME )

# source files
set( READERS_SOURCE_BASENAMES
    KSReadersMessage.cxx
    KSReadIterator.cxx
    KSReadFile.cxx
)

if( Kassiopeia_USE_ROOT )
    list( APPEND READERS_SOURCE_BASENAMES
        KSReadIteratorROOT.cxx
        KSReadRunROOT.cxx
        KSReadEventROOT.cxx
        KSReadTrackROOT.cxx
        KSReadStepROOT.cxx
        KSReadObjectROOT.cxx
        KSReadFileROOT.cxx
    )
endif( Kassiopeia_USE_ROOT )

set( READERS_SOURCE_PATH
    ${CMAKE_CURRENT_SOURCE_DIR}/Source
)
foreach( BASENAME ${READERS_SOURCE_BASENAMES} )
    list( APPEND READERS_SOURCE_FILENAMES ${READERS_SOURCE_PATH}/${BASENAME} )
endforeach( BASENAME )

# data files
set( READERS_DATA_BASENAMES
)
set( READERS_DATA_PATH
    ${CMAKE_CURRENT_SOURCE_DIR}/Data
)
foreach( BASENAME ${READERS_DATA_BASENAMES} )
    list( APPEND READERS_DATA_FILENAMES ${READERS_DATA_PATH}/${BASENAME} )
endforeach( BASENAME )

# library
add_library( KassiopeiaReaders SHARED
    ${READERS_SOURCE_FILENAMES} ${READERS_HEADER_FILENAMES} )
target_include_directories(KassiopeiaReaders
    PUBLIC $<BUILD_INTERFACE:${READERS_HEADER_PATH}> $<INSTALL_INTERFACE:include> )
target_link_libraries( KassiopeiaReaders
    PUBLIC
        KommonRoot
        KassiopeiaOperators
        ROOT::Tree
)

# install
kasper_install_headers( ${READERS_HEADER_FILENAMES} )
kasper_install_data( ${READERS_DATA_FILENAMES} )
kasper_install_libraries( KassiopeiaReaders )
