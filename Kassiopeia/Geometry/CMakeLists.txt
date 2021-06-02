# header files
set( GEOMETRY_HEADER_BASENAMES
    KSGeometryMessage.h

    KSGeoSpace.h
    KSGeoSurface.h
    KSGeoSide.h
)
set( GEOMETRY_HEADER_PATH
    ${CMAKE_CURRENT_SOURCE_DIR}/Include
)
foreach( BASENAME ${GEOMETRY_HEADER_BASENAMES} )
    list( APPEND GEOMETRY_HEADER_FILENAMES ${GEOMETRY_HEADER_PATH}/${BASENAME} )
endforeach( BASENAME )

# source files
set( GEOMETRY_SOURCE_BASENAMES
    KSGeometryMessage.cxx

    KSGeoSpace.cxx
    KSGeoSurface.cxx
    KSGeoSide.cxx
)
set( GEOMETRY_SOURCE_PATH
    ${CMAKE_CURRENT_SOURCE_DIR}/Source
)
foreach( BASENAME ${GEOMETRY_SOURCE_BASENAMES} )
    list( APPEND GEOMETRY_SOURCE_FILENAMES ${GEOMETRY_SOURCE_PATH}/${BASENAME} )
endforeach( BASENAME )

# data files
set( GEOMETRY_DATA_BASENAMES
)
set( GEOMETRY_DATA_PATH
    ${CMAKE_CURRENT_SOURCE_DIR}/Data
)
foreach( BASENAME ${GEOMETRY_DATA_BASENAMES} )
    list( APPEND GEOMETRY_DATA_FILENAMES ${GEOMETRY_DATA_PATH}/${BASENAME} )
endforeach( BASENAME )

# config files
set( GEOMETRY_CONFIG_BASENAMES
)
set( GEOMETRY_CONFIG_PATH
    ${CMAKE_CURRENT_SOURCE_DIR}/Config
)
foreach( BASENAME ${GEOMETRY_CONFIG_BASENAMES} )
    list( APPEND GEOMETRY_CONFIG_FILENAMES ${GEOMETRY_CONFIG_PATH}/${BASENAME} )
endforeach( BASENAME )

# library
add_library( KassiopeiaGeometry SHARED
    ${GEOMETRY_SOURCE_FILENAMES} ${GEOMETRY_HEADER_FILENAMES} )
target_include_directories(KassiopeiaGeometry
    PUBLIC $<BUILD_INTERFACE:${GEOMETRY_HEADER_PATH}> $<INSTALL_INTERFACE:include> )
target_link_libraries( KassiopeiaGeometry
    PUBLIC
        KGeoBagCore
        KassiopeiaOperators
)

# install
kasper_install_headers( ${GEOMETRY_HEADER_FILENAMES} )
kasper_install_data( ${GEOMETRY_DATA_FILENAMES} )
kasper_install_config_subdir( Complete ${GEOMETRY_CONFIG_FILENAMES} )
kasper_install_libraries( KassiopeiaGeometry )
