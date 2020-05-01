# CMakeLists for KEMField/UnitTest
# Author: J. Behrens adapted for KEMField W. Gosda

kasper_find_module( KEMField )

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
    #${CMAKE_CURRENT_SOURCE_DIR}/LinearAlgebra/Solvers/src/KrylovFactoryFixture.cc
    #${CMAKE_CURRENT_SOURCE_DIR}/LinearAlgebra/Solvers/src/KrylovFactoryTest.cc
)

# tests depending on plugins, unfortunately these flags are not accessible from here
# if(KEMField_USE_KOMMON_BINDINGS)
    #add_subdirectory(Plugins/Bindings)
    #set(KEMFIELD_UNIT_TEST_LIBS ${KEMFIELD_UNIT_TEST_LIBS} KEMBindingsUnitTests)
# endif(KEMField_USE_KOMMON_BINDINGS)



# Build (static) library for KEMField's unit tests

add_library(KEMFieldUnitTests SHARED ${UNIT_TESTS_SOURCEFILES} ${UNIT_TESTS_HEADERFILES})
target_link_libraries(KEMFieldUnitTests
    ${KEMField_LIBRARIES}
    ${KEMFIELD_UNIT_TEST_LIBS}
    KEMFieldsElectric
    KEMLinearAlgebraSolvers
    KFMMathUtilities
	gtest
    ${GSL_LIBRARIES}
    ${Kommon_LIBRARIES}
    ${Kommon_Vtk_LIBRARIES}
)
kasper_install_libraries(KEMFieldUnitTests)


# Build executable for KEMField's unit tests

add_executable(UnitTestKEMField ${gtest_SOURCE_DIR}/src/gtest_main.cc)
target_link_libraries(UnitTestKEMField KEMFieldUnitTests)
kasper_install_executables(UnitTestKEMField)

add_test (NAME UnitTestKEMField COMMAND UnitTestKEMField)
