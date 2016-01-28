# header files
set( VISUALIZATION_HEADER_BASENAMES
    KSVisualizationMessage.h
)
if( Kassiopeia_USE_ROOT )
    set( VISUALIZATION_HEADER_BASENAMES
    	${VISUALIZATION_HEADER_BASENAMES}
    	KSROOTTrackPainter.h
	KSROOTPotentialPainter.h
        KSROOTMagFieldPainter.h
    )
	if( Kassiopeia_USE_VTK )
	    set( VISUALIZATION_HEADER_BASENAMES
	    	${VISUALIZATION_HEADER_BASENAMES}
	    	KSVTKTrackPainter.h
	    	KSVTKTrackTerminatorPainter.h
	    )
	endif( Kassiopeia_USE_VTK )
endif( Kassiopeia_USE_ROOT )

set( VISUALIZATION_HEADER_PATH
	${CMAKE_CURRENT_SOURCE_DIR}/Include
)
foreach( BASENAME ${VISUALIZATION_HEADER_BASENAMES} )
	list( APPEND VISUALIZATION_HEADER_FILENAMES ${VISUALIZATION_HEADER_PATH}/${BASENAME} )
endforeach( BASENAME )

# source files
set( VISUALIZATION_SOURCE_BASENAMES
    KSVisualizationMessage.cxx
)

if( Kassiopeia_USE_ROOT )
    set( VISUALIZATION_SOURCE_BASENAMES
    	${VISUALIZATION_SOURCE_BASENAMES}
    	KSROOTTrackPainter.cxx
	KSROOTPotentialPainter.cxx
        KSROOTMagFieldPainter.cxx
    )
	if( Kassiopeia_USE_VTK )
	    set( VISUALIZATION_SOURCE_BASENAMES
	    	${VISUALIZATION_SOURCE_BASENAMES}
	    	KSVTKTrackPainter.cxx
	    	KSVTKTrackTerminatorPainter.cxx
	    )
	endif( Kassiopeia_USE_VTK )
endif( Kassiopeia_USE_ROOT )

set( VISUALIZATION_SOURCE_PATH
	${CMAKE_CURRENT_SOURCE_DIR}/Source
)
foreach( BASENAME ${VISUALIZATION_SOURCE_BASENAMES} )
	list( APPEND VISUALIZATION_SOURCE_FILENAMES ${VISUALIZATION_SOURCE_PATH}/${BASENAME} )
endforeach( BASENAME )

if( Kassiopeia_USE_VTK )
	if( VTK_MAJOR_VERSION GREATER 5 )
		set_property( SOURCE ${VISUALIZATION_SOURCE_FILENAMES} APPEND PROPERTY
	COMPILE_DEFINITIONS VTK6 )
	endif()
endif( Kassiopeia_USE_VTK )

# data files
set( VISUALIZATION_DATA_BASENAMES
)
set( VISUALIZATION_DATA_PATH
	${CMAKE_CURRENT_SOURCE_DIR}/Data
)
foreach( BASENAME ${VISUALIZATION_DATA_BASENAMES} )
	list( APPEND VISUALIZATION_DATA_FILENAMES ${VISUALIZATION_DATA_PATH}/${BASENAME} )
endforeach( BASENAME )

# library
kasper_internal_include_directories( ${VISUALIZATION_HEADER_PATH} )
add_library( KassiopeiaVisualization SHARED ${VISUALIZATION_SOURCE_FILENAMES} )
set_target_properties( KassiopeiaVisualization PROPERTIES COMPILE_FLAGS -Wno-overloaded-virtual )
target_link_libraries( KassiopeiaVisualization
    ${Kommon_LIBRARIES}
    ${KGeoBag_LIBRARIES}
    KassiopeiaUtility
    KassiopeiaObjects
    KassiopeiaOperators
    KassiopeiaReaders
)

# install
kasper_install_headers( ${VISUALIZATION_HEADER_FILENAMES} )
kasper_install_data( ${VISUALIZATION_DATA_FILENAMES} )
kasper_install_libraries( KassiopeiaVisualization )
