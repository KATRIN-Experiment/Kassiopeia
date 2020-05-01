# CMakeLists for Kasper/UnitTest - this will test *all* Kasper modules!
# Author: J. Behrens

enable_testing()

set( gtest_SOURCE_DIR
    ${CMAKE_CURRENT_SOURCE_DIR}/../GoogleTest
)


include_directories(
    ${CMAKE_CURRENT_SOURCE_DIR}/Include
    ${gtest_SOURCE_DIR}/include
)


## Build executable for Kasper's global unit tests

add_executable (UnitTestKasper ${gtest_SOURCE_DIR}/src/gtest_main.cc)
target_link_libraries (UnitTestKasper ${UNIT_TEST_LIBS} gtest)
# Important: we need this to avoid gcc throwing out the unit test symbols during linking
# set_target_properties(UnitTestKasper PROPERTIES LINK_FLAGS "-Wl,--no-as-needed" )
kasper_install_executables(UnitTestKasper)

add_test(NAME UnitTestKasper COMMAND UnitTestKasper)
