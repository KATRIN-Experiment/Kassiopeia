# CMakeLists for KEMField/Core
# Author: T.J. Corona, D. Hilk

# Bindings need to be linked against KGeoBag, Kommon and KMessage

option(KEMField_USE_REALTIME_CLOCK "Use real time clock for timing applications." OFF)
mark_as_advanced(FORCE KEMField_USE_REALTIME_CLOCK)

set (KEMCORE_HEADERFILES
    ${CMAKE_CURRENT_SOURCE_DIR}/include/KComplexStreamer.hh
    ${CMAKE_CURRENT_SOURCE_DIR}/include/KDataComparator.hh
    ${CMAKE_CURRENT_SOURCE_DIR}/include/KDataDisplay.hh
    ${CMAKE_CURRENT_SOURCE_DIR}/include/KEMCout.hh
    ${CMAKE_CURRENT_SOURCE_DIR}/include/KEMCoreMessage.hh
    ${CMAKE_CURRENT_SOURCE_DIR}/include/KEMConstants.hh
    ${CMAKE_CURRENT_SOURCE_DIR}/include/KEMStringUtils.hh
    ${CMAKE_CURRENT_SOURCE_DIR}/include/KEMTicker.hh
    ${CMAKE_CURRENT_SOURCE_DIR}/include/KFundamentalTypeCounter.hh
    ${CMAKE_CURRENT_SOURCE_DIR}/include/KFundamentalTypes.hh
    ${CMAKE_CURRENT_SOURCE_DIR}/include/KMPIEnvironment.hh
    #${CMAKE_CURRENT_SOURCE_DIR}/include/KSmartPointer.hh
    ${CMAKE_CURRENT_SOURCE_DIR}/include/KTimer.hh
    ${CMAKE_CURRENT_SOURCE_DIR}/include/KTypeManipulation.hh
    ${CMAKE_CURRENT_SOURCE_DIR}/include/KTypelist.hh
    ${CMAKE_CURRENT_SOURCE_DIR}/include/KTypelistVisitor.hh
)

set (KEMCORE_SOURCEFILES
    ${CMAKE_CURRENT_SOURCE_DIR}/src/KEMConstants.cc
    ${CMAKE_CURRENT_SOURCE_DIR}/src/KEMCout.cc
    ${CMAKE_CURRENT_SOURCE_DIR}/src/KEMCoreMessage.cc
    ${CMAKE_CURRENT_SOURCE_DIR}/src/KEMStringUtils.cc
    ${CMAKE_CURRENT_SOURCE_DIR}/src/KEMTicker.cc
    ${CMAKE_CURRENT_SOURCE_DIR}/src/KFundamentalTypes.cc
    ${CMAKE_CURRENT_SOURCE_DIR}/src/KTimer.cc
)

if (KEMField_USE_MPI)
    list (APPEND KEMCORE_HEADERFILES
        ${CMAKE_CURRENT_SOURCE_DIR}/include/KMPIInterface.hh
    )
    list (APPEND KEMCORE_SOURCEFILES
        ${CMAKE_CURRENT_SOURCE_DIR}/src/KMPIInterface.cc
    )
endif (KEMField_USE_MPI)

if (KEMField_USE_KMESSAGE)
    list (APPEND KEMCORE_HEADERFILES
        ${CMAKE_CURRENT_SOURCE_DIR}/include/KMessageInterface.hh
    )
    list (APPEND KEMCORE_SOURCEFILES
        ${CMAKE_CURRENT_SOURCE_DIR}/src/KMessageInterface.cc
    )
endif (KEMField_USE_KMESSAGE)

add_library (KEMCore SHARED
    ${KEMCORE_SOURCEFILES} ${KEMCORE_HEADERFILES})
target_include_directories(KEMCore
    PUBLIC $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include> $<INSTALL_INTERFACE:include>)

target_link_libraries (KEMCore
    PUBLIC
        KommonCore
)

target_compile_definitions( KEMCore PUBLIC KEMFIELD_USE_KOMMON )

if( KEMField_ENABLE_DEBUG )
    target_compile_definitions( KEMCore PUBLIC KEMField_ENABLE_DEBUG )
endif( KEMField_ENABLE_DEBUG )

#if (KEMField_SILENT)
#  target_compile_definitions(KEMCore PUBLIC KEMFIELD_SILENT)
#endif (KEMField_SILENT)

if (KEMField_USE_KMESSAGE)
    target_compile_definitions(KEMCore PUBLIC KEMFIELD_USE_KMESSAGE)
endif (KEMField_USE_KMESSAGE)

if(KEMField_USE_MPI)
    target_link_libraries(KEMCore PUBLIC MPI::MPI_CXX)
    target_compile_definitions(KEMCore PUBLIC KEMFIELD_USE_MPI)
endif(KEMField_USE_MPI)

if(KEMField_USE_REALTIME_CLOCK)
    target_compile_definitions (KEMCore PUBLIC KEMFIELD_USE_REALTIME_CLOCK)
endif(KEMField_USE_REALTIME_CLOCK)

kasper_install_headers (${KEMCORE_HEADERFILES})
kasper_install_libraries (KEMCore)

# add_executable( KSmartPointer_test test/KSmartPointer_test.cc )
# target_link_libraries( KSmartPointer_test KEMCore)
# kasper_install_executables( KSmartPointer_test )
