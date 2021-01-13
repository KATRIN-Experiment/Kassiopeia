# CMakeLists for KEMField/UnitTest
# Author: J. Behrens adapted for KEMField W. Gosda

kasper_find_module( KEMField )

if( KEMField_USE_MPI )
  #find_package(MPI)
  if (MPI_CXX_FOUND)
    if (MPI_CXX_COMPILER)  # MPICH2 style
      set (CMAKE_C_COMPILER ${MPI_CXX_COMPILER})
      set (CMAKE_CXX_COMPILER ${MPI_CXX_COMPILER})
    else (MPI_CXX_COMPILER)  # OpenMPI style
      set (CMAKE_C_COMPILER ${MPI_C_COMPILER})
      set (CMAKE_CXX_COMPILER ${MPI_CXX_COMPILER})
    endif (MPI_CXX_COMPILER)
  else (MPI_CXX_FOUND)
    message(WARNING "Warning: MPI not found on your system, using default compiler" )
    set (CMAKE_C_COMPILER mpicc)
    set (CMAKE_CXX_COMPILER mpicxx)
  endif (MPI_CXX_FOUND)
  add_cflag (KEMFIELD_USE_MPI)

endif( KEMField_USE_MPI )

enable_testing()

set( gtest_SOURCE_DIR
    ${CMAKE_CURRENT_SOURCE_DIR}/../GoogleTest
)

include_directories(
    ${CMAKE_CURRENT_SOURCE_DIR}/include
    ${CMAKE_CURRENT_SOURCE_DIR}/Fields/include
    ${CMAKE_CURRENT_SOURCE_DIR}/LinearAlgebra/Solvers/include
    ${gtest_SOURCE_DIR}/include
    ${KEMField_INCLUDE_DIRS}
)

set (UNIT_TESTS_HEADERFILES
    ${CMAKE_CURRENT_SOURCE_DIR}/include/KEMFieldTest.hh
    #${CMAKE_CURRENT_SOURCE_DIR}/LinearAlgebra/Solvers/include/KrylovFactoryFixture.hh
)

set (UNIT_TESTS_SOURCEFILES
    ${CMAKE_CURRENT_SOURCE_DIR}/Fields/src/KElectrostaticConstantFieldTest.cc
    ${CMAKE_CURRENT_SOURCE_DIR}/Fields/src/KElectrostaticLinearFieldTest.cc
    ${CMAKE_CURRENT_SOURCE_DIR}/Fields/src/KElectrostaticQuadrupoleFieldTest.cc
    #${CMAKE_CURRENT_SOURCE_DIR}/LinearAlgebra/Solvers/src/KrylovFactoryFixture.cc
    #${CMAKE_CURRENT_SOURCE_DIR}/LinearAlgebra/Solvers/src/KrylovFactoryTest.cc
    ${CMAKE_CURRENT_SOURCE_DIR}/Fields/src/KMagnetostaticConstantFieldTest.cc
    ${CMAKE_CURRENT_SOURCE_DIR}/Fields/src/KMagnetostaticDipoleFieldTest.cc
    ${CMAKE_CURRENT_SOURCE_DIR}/Fields/src/CubeCapacitanceTest.cc
    ${CMAKE_CURRENT_SOURCE_DIR}/Fields/src/DiskCapacitanceTest.cc
    ${CMAKE_CURRENT_SOURCE_DIR}/Fields/src/SphereCapacitanceTest.cc
    ${CMAKE_CURRENT_SOURCE_DIR}/Fields/src/IntegratorsTest.cc
    ${CMAKE_CURRENT_SOURCE_DIR}/Fields/src/ElectromagnetsTest.cc
    ${CMAKE_CURRENT_SOURCE_DIR}/Fields/src/DielectricsTest.cc
    ${CMAKE_CURRENT_SOURCE_DIR}/Fields/src/ZonalHarmonicsTest.cc
)

if(KEMField_USE_OPENCL)
    list(APPEND UNIT_TESTS_SOURCEFILES
        ${CMAKE_CURRENT_SOURCE_DIR}/Fields/src/OpenCLTest.cc
    )
endif()

# tests depending on plugins, unfortunately these flags are not accessible from here
# if(KEMField_USE_KOMMON_BINDINGS)
    #add_subdirectory(Plugins/Bindings)
    #set(KEMFIELD_UNIT_TEST_LIBS ${KEMFIELD_UNIT_TEST_LIBS} KEMBindingsUnitTests)
# endif(KEMField_USE_KOMMON_BINDINGS)



# Build (static) library for KEMField's unit tests

if (KEMField_USE_OPENCL)
    find_package (OpenCL REQUIRED)
    kasper_external_include_directories( ${OPENCL_INCLUDE_DIRS} )
    add_cflag (KEMFIELD_USE_OPENCL)
    list (APPEND KEMFIELD_UNIT_TEST_LIBS
        KEMOpenCLPlugin
    )
endif (KEMField_USE_OPENCL)

if (KEMField_USE_PETSc)
    find_package (PETSc REQUIRED)
    kasper_external_include_directories (${PETSC_INCLUDES})
    add_cflag (KEMFIELD_USE_PETSC)
    list (APPEND KEMFIELD_UNIT_TEST_LIBS
        KEMPETScPlugin
    )
endif (KEMField_USE_PETSc)


add_library(KEMFieldUnitTests SHARED ${UNIT_TESTS_SOURCEFILES} ${UNIT_TESTS_HEADERFILES})
target_link_libraries(KEMFieldUnitTests
    gtest
    ${GSL_LIBRARIES}
    ${FFTW_LIBRARIES}
    ${Kommon_LIBRARIES}
    ${Kommon_Vtk_LIBRARIES}
    ${KEMField_LIBRARIES}
    ${KEMFIELD_UNIT_TEST_LIBS}
)
kasper_install_libraries(KEMFieldUnitTests)


# Build executable for KEMField's unit tests

add_executable(UnitTestKEMField ${gtest_SOURCE_DIR}/src/gtest_main.cc)
target_link_libraries(UnitTestKEMField KEMFieldUnitTests gtest)
kasper_install_executables(UnitTestKEMField)

add_test (NAME UnitTestKEMField COMMAND UnitTestKEMField)
