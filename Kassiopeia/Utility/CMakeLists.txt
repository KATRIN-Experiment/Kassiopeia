# header files
set( UTILITY_HEADER_BASENAMES
    KSNumerical.h
    KSMutex.h
    KSCondition.h
    KSCyclicIterator.h
    KSException.h
    KSExpression.h
    KSList.h
    KSUtilityMessage.h
    KSPathResolver.h
)
set( UTILITY_HEADER_PATH 
	${CMAKE_CURRENT_SOURCE_DIR}/Include
)
foreach( BASENAME ${UTILITY_HEADER_BASENAMES} )
	list( APPEND UTILITY_HEADER_FILENAMES ${UTILITY_HEADER_PATH}/${BASENAME} )
endforeach( BASENAME )

# source files
set( UTILITY_SOURCE_BASENAMES
    KSMutex.cxx
    KSCondition.cxx
    KSUtilityMessage.cxx
)
set( UTILITY_SOURCE_PATH 
	${CMAKE_CURRENT_SOURCE_DIR}/Source
)
foreach( BASENAME ${UTILITY_SOURCE_BASENAMES} )
	list( APPEND UTILITY_SOURCE_FILENAMES ${UTILITY_SOURCE_PATH}/${BASENAME} )
endforeach( BASENAME )
    
# internal
kasper_internal_include_directories(
	${CMAKE_CURRENT_SOURCE_DIR}/Include
)

# target
kasper_internal_include_directories( ${UTILITY_HEADER_PATH} )
add_library( KassiopeiaUtility SHARED ${UTILITY_SOURCE_FILENAMES} )
target_link_libraries( KassiopeiaUtility
    pthread
    ${Kommon_LIBRARIES}
    ${KGeoBag_LIBRARIES}
)

# install
kasper_install_headers( ${UTILITY_HEADER_FILENAMES} )
kasper_install_libraries( KassiopeiaUtility )

