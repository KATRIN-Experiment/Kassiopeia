# header files
set( WRITERS_HEADER_BASENAMES
    KSWritersMessage.h
    KSWriteASCII.h
)
if( Kassiopeia_USE_VTK )
    set( WRITERS_HEADER_BASENAMES
    	${WRITER_HEADER_BASENAMES}
    	KSWriteVTK.h
    )
endif( Kassiopeia_USE_VTK )
if( Kassiopeia_USE_ROOT )
    set( WRITERS_HEADER_BASENAMES
    	${WRITER_HEADER_BASENAMES}
    	KSWriteROOT.h
    )
endif( Kassiopeia_USE_ROOT )
set( WRITERS_HEADER_PATH 
	${CMAKE_CURRENT_SOURCE_DIR}/Include
)
foreach( BASENAME ${WRITERS_HEADER_BASENAMES} )
	list( APPEND WRITERS_HEADER_FILENAMES ${WRITERS_HEADER_PATH}/${BASENAME} )
endforeach( BASENAME )

# source files
set( WRITERS_SOURCE_BASENAMES
    KSWritersMessage.cxx
    KSWriteASCII.cxx
)
if( Kassiopeia_USE_VTK )
    set( WRITERS_SOURCE_BASENAMES
    	${WRITERS_SOURCE_BASENAMES}
    	KSWriteVTK.cxx
    )
endif( Kassiopeia_USE_VTK )
if( Kassiopeia_USE_ROOT )
    set( WRITERS_SOURCE_BASENAMES
    	${WRITERS_SOURCE_BASENAMES}
    	KSWriteROOT.cxx
    )
endif( Kassiopeia_USE_ROOT )
set( WRITERS_SOURCE_PATH 
	${CMAKE_CURRENT_SOURCE_DIR}/Source
)
foreach( BASENAME ${WRITERS_SOURCE_BASENAMES} )
	list( APPEND WRITERS_SOURCE_FILENAMES ${WRITERS_SOURCE_PATH}/${BASENAME} )
endforeach( BASENAME )

if( Kassiopeia_USE_VTK )
	if( VTK_MAJOR_VERSION GREATER 5 )
		set_property( SOURCE ${WRITERS_SOURCE_FILENAMES} APPEND PROPERTY 
	COMPILE_DEFINITIONS VTK6 )
	endif()
endif( Kassiopeia_USE_VTK )

# data files
set( WRITERS_DATA_BASENAMES
)
set( WRITERS_DATA_PATH 
	${CMAKE_CURRENT_SOURCE_DIR}/Data
)
foreach( BASENAME ${WRITERS_DATA_BASENAMES} )
	list( APPEND WRITERS_DATA_FILENAMES ${WRITERS_DATA_PATH}/${BASENAME} )
endforeach( BASENAME )

# library
kasper_internal_include_directories( ${WRITERS_HEADER_PATH} )
add_library( KassiopeiaWriters SHARED ${WRITERS_SOURCE_FILENAMES} )
set_target_properties( KassiopeiaWriters PROPERTIES COMPILE_FLAGS -Wno-overloaded-virtual )
target_link_libraries( KassiopeiaWriters
    ${Kommon_LIBRARIES}
    ${KGeoBag_LIBRARIES}
    KassiopeiaUtility
    KassiopeiaObjects
    KassiopeiaOperators
)

# install
kasper_install_headers( ${WRITERS_HEADER_FILENAMES} )
kasper_install_data( ${WRITERS_DATA_FILENAMES} )
kasper_install_libraries( KassiopeiaWriters )
