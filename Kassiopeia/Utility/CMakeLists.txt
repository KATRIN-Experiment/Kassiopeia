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

# pthread
set(THREADS_PREFER_PTHREAD_FLAG ON)
find_package(Threads REQUIRED)

# library
add_library( KassiopeiaUtility SHARED
    ${UTILITY_SOURCE_FILENAMES} ${UTILITY_HEADER_FILENAMES} )
target_include_directories(KassiopeiaUtility
    PUBLIC $<BUILD_INTERFACE:${UTILITY_HEADER_PATH}> $<INSTALL_INTERFACE:include> )
target_link_libraries( KassiopeiaUtility
    PUBLIC
        Kommon
    PRIVATE
        Threads::Threads
)

if( Kassiopeia_ENABLE_DEBUG )
    target_compile_definitions( KassiopeiaUtility PUBLIC Kassiopeia_ENABLE_DEBUG )
endif( Kassiopeia_ENABLE_DEBUG )

if( Kassiopeia_USE_BOOST )
    target_compile_definitions( KassiopeiaUtility PUBLIC Kassiopeia_USE_BOOST )
endif( Kassiopeia_USE_BOOST )

if( Kassiopeia_USE_ROOT )
    target_compile_definitions( KassiopeiaUtility PUBLIC Kassiopeia_USE_ROOT )
endif( Kassiopeia_USE_ROOT )

if( Kassiopeia_USE_VTK )
    target_compile_definitions( KassiopeiaUtility PUBLIC Kassiopeia_USE_VTK )
endif( Kassiopeia_USE_VTK )

# install
kasper_install_headers( ${UTILITY_HEADER_FILENAMES} )
kasper_install_libraries( KassiopeiaUtility )
