# CMakeLists for KEMField/UnitTest
# Author: J. Behrens adapted for KEMField W. Gosda

enable_testing()

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

add_library(KEMFieldUnitTests SHARED ${UNIT_TESTS_SOURCEFILES} ${UNIT_TESTS_HEADERFILES})
target_include_directories( KEMFieldUnitTests
    PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/include)
target_link_libraries(KEMFieldUnitTests
    PUBLIC
        UnitTestBase
        KGeoBagInterface
        KEMCore
        KEMLinearAlgebraCore
        KEMBoundaryIntegralsCore
        KEMElectrostaticBoundaryIntegrals
        KEMFieldMagneticChargeDensitySolvers        
        KEMFieldChargeDensitySolvers
        KEMFieldsElectric
        KEMFieldsMagnetic
        KEMElectromagnets
        KEMZHSolver
)


if (KEMField_USE_OPENCL)
    find_package (OpenCL REQUIRED)
    target_compile_definitions (KEMFieldUnitTests PUBLIC KEMFIELD_USE_OPENCL)
    target_link_libraries (KEMFieldUnitTests PUBLIC KEMOpenCLPlugin)
endif (KEMField_USE_OPENCL)

if (KEMField_USE_PETSc)
    find_package (PETSc REQUIRED)
    target_compile_definitions (KEMFieldUnitTests PUBLIC KEMFIELD_USE_PETSC)
    target_link_libraries (KEMFieldUnitTests PUBLIC KEMPETScPlugin)
endif (KEMField_USE_PETSc)

kasper_install_libraries(KEMFieldUnitTests)


# Build executable for KEMField's unit tests

add_executable(UnitTestKEMField ${GTEST_MAIN_SOURCE})
target_link_libraries(UnitTestKEMField KEMFieldUnitTests)
kasper_install_executables(UnitTestKEMField)

add_test (NAME UnitTestKEMField COMMAND UnitTestKEMField)
