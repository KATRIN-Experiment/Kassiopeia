# attention:
# i spent some time making this file neat and organized.
# this is important as it allows changes to be performed quickly.
# please be considerate and make additions as neatly as you can.
# thank you! -dan.

cmake_minimum_required( VERSION ${CMAKE_MINIMUM_VERSION} )

if(POLICY CMP0053)
    cmake_policy(SET CMP0053 NEW)
endif()

# Kassiopeia version
set( MODULE_VERSION_MAJOR 3 )
set( MODULE_VERSION_MINOR 7 )
set( MODULE_VERSION_PATCH 7 )
set( MODULE_VERSION "${MODULE_VERSION_MAJOR}.${MODULE_VERSION_MINOR}.${MODULE_VERSION_PATCH}" )

#project( Kassiopeia VERSION ${MODULE_VERSION} )
project( Kassiopeia )

include( KasperDefaults )

# require c++11

# module options
# build in boost functionality [utilities]
SET( Kassiopeia_USE_BOOST ${KASPER_USE_BOOST} )
# build in root functionality [writers, visualization]
SET( Kassiopeia_USE_ROOT ${KASPER_USE_ROOT} )
# build in vtk functionality [visualization]
SET( Kassiopeia_USE_VTK ${KASPER_USE_VTK} )

# module paths
kasper_module_paths( Kassiopeia )

# module debugging
kasper_module_debug()
if( Kassiopeia_ENABLE_DEBUG )
    add_cflag( Kassiopeia_ENABLE_DEBUG )
endif()

# external dependencies: boost
if( Kassiopeia_USE_BOOST )
    find_package( Boost 1.43.0 COMPONENTS filesystem system REQUIRED )
    if( Boost_INCLUDE_DIR )
        kasper_external_include_directories( ${Boost_INCLUDE_DIR} )
        list(APPEND EXTERNAL_LIBRARIES ${Boost_LIBRARIES} )
    else()
        list(APPEND EXTERNAL_LIBRARIES Boost::headers Boost::filesystem Boost::system )
    endif()
    add_definitions( -DKassiopeia_USE_BOOST )
endif()

# external dependencies: root
if( Kassiopeia_USE_ROOT )
    find_package( ROOT 5.24.0 REQUIRED )
    kasper_external_include_directories( ${ROOT_INCLUDE_DIRS} )
    add_definitions( -DKassiopeia_USE_ROOT )
endif()

# external dependencies: vtk
if( Kassiopeia_USE_VTK )
    kasper_find_vtk()
    add_definitions( -DKassiopeia_USE_VTK )
endif()


# internal dependencies: kommon
kasper_find_module( Kommon )
foreach( FLAG ${Kommon_CFLAGS} )
    add_definitions( -D${FLAG} )
endforeach()

# internal dependencies: kgeobag
kasper_find_module( KGeoBag )
foreach( FLAG ${KGeoBag_CFLAGS} )
    add_definitions( -D${FLAG} )
endforeach()

# # internal dependencies: kemfield
kasper_find_module( KEMField )
foreach( FLAG ${KEMField_CFLAGS} )
   add_definitions( -D${FLAG} )
   if( ${FLAG} STREQUAL KEMFIELD_USE_MPI )
        find_package(MPI)
        if (MPI_FOUND)
            include_directories(${MPI_INCLUDE_PATH})
        endif()
    endif()
 endforeach()

if (KEMField_USE_MPI)
  find_package(MPI)
  if (MPI_FOUND)
    if (MPI_COMPILER)  # MPICH2 style
      set (CMAKE_C_COMPILER ${MPI_COMPILER})
      set (CMAKE_CXX_COMPILER ${MPI_COMPILER})
    else ()  # OpenMPI style
      set (CMAKE_C_COMPILER ${MPI_C_COMPILER})
      set (CMAKE_CXX_COMPILER ${MPI_CXX_COMPILER})
    endif ()
  else ()
    set (CMAKE_C_COMPILER mpicc)
    set (CMAKE_CXX_COMPILER mpicxx)
  endif ()
  set (STATIC_LIBRARY_SUFFIX _mpi)
  set (SHARED_LIBRARY_SUFFIX _mpi)
  set (CMAKE_STATIC_LIBRARY_SUFFIX ${STATIC_LIBRARY_SUFFIX}${CMAKE_STATIC_LIBRARY_SUFFIX})
  set (CMAKE_SHARED_LIBRARY_SUFFIX ${SHARED_LIBRARY_SUFFIX}${CMAKE_SHARED_LIBRARY_SUFFIX})
  set (CMAKE_EXECUTABLE_SUFFIX ${SHARED_LIBRARY_SUFFIX}${CMAKE_EXECUTABLE_SUFFIX})
  set (CMAKE_EXECUTABLE_SUFFIX ${CMAKE_EXECUTABLE_SUFFIX} PARENT_SCOPE)
endif ()


# subdirectories
add_subdirectory( Documentation )
add_subdirectory( Utility )
add_subdirectory( Math )
add_subdirectory( Objects )
add_subdirectory( Operators )
add_subdirectory( Fields )
add_subdirectory( Geometry )
add_subdirectory( Generators )
add_subdirectory( Trajectories )
add_subdirectory( Navigators )
add_subdirectory( Interactions )
add_subdirectory( Terminators )
add_subdirectory( Modifiers )
add_subdirectory( Writers )
add_subdirectory( Simulation )
add_subdirectory( Readers )
add_subdirectory( XML )
add_subdirectory( Visualization )
add_subdirectory( Bindings )
add_subdirectory( Python )
add_subdirectory( Applications/Simulation )
add_subdirectory( Applications/FieldCalculation )


add_subdirectory( Applications/Validation )
if( Kassiopeia_USE_ROOT )
    add_subdirectory( Applications/Examples )
    add_subdirectory( Applications/Other )
endif( )

kasper_install_module()
