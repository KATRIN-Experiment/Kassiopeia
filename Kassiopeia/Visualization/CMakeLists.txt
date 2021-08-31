# header files
set( VISUALIZATION_HEADER_BASENAMES
    KSVisualizationMessage.h
)
if( Kassiopeia_USE_ROOT )
    list(APPEND VISUALIZATION_HEADER_BASENAMES
        KSROOTTrackPainter.h
        KSROOTPotentialPainter.h
        KSROOTMagFieldPainter.h
        KSROOTZonalHarmonicsPainter.h
    )
    if( Kassiopeia_USE_VTK )
        list(APPEND VISUALIZATION_HEADER_BASENAMES
            KSVTKGeneratorPainter.h
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
    list(APPEND VISUALIZATION_SOURCE_BASENAMES
        KSROOTTrackPainter.cxx
        KSROOTPotentialPainter.cxx
        KSROOTMagFieldPainter.cxx
        KSROOTZonalHarmonicsPainter.cxx
    )
    if( Kassiopeia_USE_VTK )
        list(APPEND VISUALIZATION_SOURCE_BASENAMES
            KSVTKGeneratorPainter.cxx
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
add_library( KassiopeiaVisualization SHARED
    ${VISUALIZATION_SOURCE_FILENAMES} ${VISUALIZATION_HEADER_FILENAMES} )
target_include_directories(KassiopeiaVisualization
    PUBLIC $<BUILD_INTERFACE:${VISUALIZATION_HEADER_PATH}> $<INSTALL_INTERFACE:include> )
target_link_libraries( KassiopeiaVisualization
    PUBLIC
        KassiopeiaFields
        KassiopeiaReaders
        KassiopeiaSimulation
        KGeoBagInterface
)

if( Kassiopeia_USE_VTK )
    if( VTK_MAJOR_VERSION GREATER 5 )
        target_compile_definitions( KassiopeiaVisualization PRIVATE VTK6 )
    endif()
endif( Kassiopeia_USE_VTK )

# install
kasper_install_headers( ${VISUALIZATION_HEADER_FILENAMES} )
kasper_install_libraries( KassiopeiaVisualization )
kasper_install_data( ${VISUALIZATION_DATA_FILENAMES} )
