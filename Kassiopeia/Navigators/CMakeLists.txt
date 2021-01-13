set(KSNAV_CACHE_DIR "${${PROJECT_NAME}_CACHE_INSTALL_DIR}" CACHE STRING "Kassiopeia navigator cache directory")
mark_as_advanced(FORCE KSNAV_CACHE_DIR)


# header files
set( NAVIGATORS_HEADER_BASENAMES
    KSNavigatorsMessage.h
    KSNavSurface.h
    KSNavSpace.h
    KSNavMeshedSpace.h
    KSNavOctreeData.h
)
set( NAVIGATORS_HEADER_PATH
	${CMAKE_CURRENT_SOURCE_DIR}/Include
)
foreach( BASENAME ${NAVIGATORS_HEADER_BASENAMES} )
	list( APPEND NAVIGATORS_HEADER_FILENAMES ${NAVIGATORS_HEADER_PATH}/${BASENAME} )
endforeach( BASENAME )

# source files
set( NAVIGATORS_SOURCE_BASENAMES
    KSNavigatorsMessage.cxx
    KSNavSurface.cxx
    KSNavSpace.cxx
    KSNavMeshedSpace.cxx
)
set( NAVIGATORS_SOURCE_PATH
	${CMAKE_CURRENT_SOURCE_DIR}/Source
)
foreach( BASENAME ${NAVIGATORS_SOURCE_BASENAMES} )
	list( APPEND NAVIGATORS_SOURCE_FILENAMES ${NAVIGATORS_SOURCE_PATH}/${BASENAME} )
endforeach( BASENAME )

# data files
set( NAVIGATORS_DATA_BASENAMES
)
set( NAVIGATORS_DATA_PATH
	${CMAKE_CURRENT_SOURCE_DIR}/Data
)
foreach( BASENAME ${NAVIGATORS_DATA_BASENAMES} )
	list( APPEND NAVIGATORS_DATA_FILENAMES ${NAVIGATORS_DATA_PATH}/${BASENAME} )
endforeach( BASENAME )

# config files
set( NAVIGATORS_CONFIG_BASENAMES
)
set( NAVIGATORS_CONFIG_PATH
	${CMAKE_CURRENT_SOURCE_DIR}/Config
)
foreach( BASENAME ${NAVIGATORS_CONFIG_BASENAMES} )
	list( APPEND NAVIGATORS_CONFIG_FILENAMES ${NAVIGATORS_CONFIG_PATH}/${BASENAME} )
endforeach( BASENAME )


# library
kasper_internal_include_directories( ${NAVIGATORS_HEADER_PATH} )
add_library( KassiopeiaNavigators SHARED ${NAVIGATORS_SOURCE_FILENAMES} ${NAVIGATORS_HEADER_FILENAMES} )
set_target_properties( KassiopeiaNavigators PROPERTIES COMPILE_FLAGS -Wno-overloaded-virtual )
target_link_libraries( KassiopeiaNavigators
    ${Kommon_LIBRARIES}
    ${KGeoBag_LIBRARIES}
    KEMFileManipulation
    KEMHashGenerator
    KassiopeiaUtility
    KassiopeiaObjects
    KassiopeiaOperators
    KassiopeiaGeometry
)

# install
kasper_install_headers( ${NAVIGATORS_HEADER_FILENAMES} )
kasper_install_data( ${NAVIGATORS_DATA_FILENAMES} )
kasper_install_config_subdir( Complete ${NAVIGATORS_CONFIG_FILENAMES} )
kasper_install_libraries( KassiopeiaNavigators )
