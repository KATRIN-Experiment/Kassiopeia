# CMakeLists for KEMField/Core
# Author: T.J. Corona, D. Hilk

# Bindings need to be linked against KGeoBag, Kommon and KMessage

#option (${PROJECT_NAME}_SILENT "Disable all messaging" OFF)
#if (${PROJECT_NAME}_SILENT)
#  add_cflag (KEMFIELD_SILENT)
#endif (${PROJECT_NAME}_SILENT)

kasper_require_cpp11()
kasper_find_module(Kommon)
add_cflag (KEMFIELD_USE_KOMMON)

#cmake_dependent_option (${PROJECT_NAME}_USE_KMESSAGE "Use KMessage" ON "NOT ${PROJECT_NAME}_SILENT" OFF)


#if (${PROJECT_NAME}_USE_KMESSAGE)
  add_cflag (KEMFIELD_USE_KMESSAGE)
#endif (${PROJECT_NAME}_USE_KMESSAGE)

option (${PROJECT_NAME}_USE_MPI "Use MPI to accelerate calculations using multiple processors" OFF)
if (${PROJECT_NAME}_USE_MPI)
  find_package(MPI)
  if (MPI_CXX_FOUND)
    if (MPI_CXX_COMPILER)  # MPICH2 style
      set (CMAKE_C_COMPILER ${MPI_COMPILER})
      set (CMAKE_CXX_COMPILER ${MPI_COMPILER})
    else (MPI_CXX_COMPILER)  # OpenMPI style
      set (CMAKE_C_COMPILER ${MPI_C_COMPILER})
      set (CMAKE_CXX_COMPILER ${MPI_CXX_COMPILER})
    endif (MPI_CXX_COMPILER)
  else (MPI_CXX_FOUND)
    message( "Warning: MPI not found on your system, using default compiler" )
    set (CMAKE_C_COMPILER mpicc)
    set (CMAKE_CXX_COMPILER mpicxx)
  endif (MPI_CXX_FOUND)
  add_cflag (KEMFIELD_USE_MPI)
  set (STATIC_LIBRARY_SUFFIX _mpi)
  set (SHARED_LIBRARY_SUFFIX _mpi)
  set (CMAKE_STATIC_LIBRARY_SUFFIX ${STATIC_LIBRARY_SUFFIX}${CMAKE_STATIC_LIBRARY_SUFFIX})
  set (CMAKE_SHARED_LIBRARY_SUFFIX ${SHARED_LIBRARY_SUFFIX}${CMAKE_SHARED_LIBRARY_SUFFIX})
  set (CMAKE_EXECUTABLE_SUFFIX ${SHARED_LIBRARY_SUFFIX}${CMAKE_EXECUTABLE_SUFFIX})

  set (CMAKE_C_COMPILER ${CMAKE_C_COMPILER} PARENT_SCOPE)
  set (CMAKE_CXX_COMPILER ${CMAKE_CXX_COMPILER} PARENT_SCOPE)
  set (STATIC_LIBRARY_SUFFIX ${STATIC_SUFFIX_LIBRARY} PARENT_SCOPE)
  set (SHARED_LIBRARY_SUFFIX ${SHARED_LIBRARY_SUFFIX} PARENT_SCOPE)
  set (CMAKE_STATIC_LIBRARY_SUFFIX ${CMAKE_STATIC_LIBRARY_SUFFIX} PARENT_SCOPE)
  set (CMAKE_SHARED_LIBRARY_SUFFIX ${CMAKE_SHARED_LIBRARY_SUFFIX} PARENT_SCOPE)
  set (CMAKE_EXECUTABLE_SUFFIX ${CMAKE_EXECUTABLE_SUFFIX} PARENT_SCOPE)

#  set( ${PROJECT_NAME}_USE_ROOT OFF CACHE BOOL "(Required)" FORCE )
    set( ${PROJECT_NAME}_USE_ROOT ${KASPER_USE_ROOT} )

endif (${PROJECT_NAME}_USE_MPI)

option(${PROJECT_NAME}_USE_REALTIME_CLOCK "Use real time clock for timing applications." OFF)
mark_as_advanced(FORCE ${PROJECT_NAME}_USE_REALTIME_CLOCK)
if (${PROJECT_NAME}_USE_REALTIME_CLOCK)
  add_cflag (KEMFIELD_USE_REALTIME_CLOCK)
endif (${PROJECT_NAME}_USE_REALTIME_CLOCK)

kasper_internal_include_directories (${CMAKE_CURRENT_SOURCE_DIR}/include)

set (KEMCORE_HEADERFILES
  ${CMAKE_CURRENT_SOURCE_DIR}/include/KComplexStreamer.hh
  ${CMAKE_CURRENT_SOURCE_DIR}/include/KDataComparator.hh
  ${CMAKE_CURRENT_SOURCE_DIR}/include/KDataDisplay.hh
  ${CMAKE_CURRENT_SOURCE_DIR}/include/KEMCout.hh
  ${CMAKE_CURRENT_SOURCE_DIR}/include/KEMConstants.hh
  ${CMAKE_CURRENT_SOURCE_DIR}/include/KEMStringUtils.hh
  ${CMAKE_CURRENT_SOURCE_DIR}/include/KEMTicker.hh
  ${CMAKE_CURRENT_SOURCE_DIR}/include/KFundamentalTypeCounter.hh
  ${CMAKE_CURRENT_SOURCE_DIR}/include/KFundamentalTypes.hh
  ${CMAKE_CURRENT_SOURCE_DIR}/include/KMPIEnvironment.hh
  ${CMAKE_CURRENT_SOURCE_DIR}/include/KSmartPointer.hh
  ${CMAKE_CURRENT_SOURCE_DIR}/include/KTimer.hh
  ${CMAKE_CURRENT_SOURCE_DIR}/include/KTypelist.hh
  ${CMAKE_CURRENT_SOURCE_DIR}/include/KTypelistVisitor.hh
  )

set (KEMCORE_SOURCEFILES
  ${CMAKE_CURRENT_SOURCE_DIR}/src/KEMConstants.cc
  ${CMAKE_CURRENT_SOURCE_DIR}/src/KEMCout.cc
  ${CMAKE_CURRENT_SOURCE_DIR}/src/KEMStringUtils.cc
  ${CMAKE_CURRENT_SOURCE_DIR}/src/KEMTicker.cc
  ${CMAKE_CURRENT_SOURCE_DIR}/src/KFundamentalTypes.cc
  ${CMAKE_CURRENT_SOURCE_DIR}/src/KTimer.cc
  )

if (${PROJECT_NAME}_USE_MPI)
  list (APPEND KEMCORE_HEADERFILES
    ${CMAKE_CURRENT_SOURCE_DIR}/include/KMPIInterface.hh
    )
  list (APPEND KEMCORE_SOURCEFILES
    ${CMAKE_CURRENT_SOURCE_DIR}/src/KMPIInterface.cc
    )
endif (${PROJECT_NAME}_USE_MPI)

#if (${PROJECT_NAME}_USE_KMESSAGE)
  list (APPEND KEMCORE_HEADERFILES
    ${CMAKE_CURRENT_SOURCE_DIR}/include/KMessageInterface.hh
    )
  list (APPEND KEMCORE_SOURCEFILES
    ${CMAKE_CURRENT_SOURCE_DIR}/src/KMessageInterface.cc
    )
#endif (${PROJECT_NAME}_USE_KMESSAGE)

add_library (KEMCore SHARED ${KEMCORE_SOURCEFILES})
target_link_libraries (KEMCore ${Kommon_LIBRARIES})

add_executable( KSmartPointer_test test/KSmartPointer_test.cc )
  target_link_libraries( KSmartPointer_test KEMCore
  )
  
  kasper_install_executables( KSmartPointer_test )

kasper_install_headers (${KEMCORE_HEADERFILES})
kasper_install_libraries (KEMCore)
