# CMakeLists for Kasper/UnitTest - this will test *all* Kasper modules!
# Author: J. Behrens

enable_testing()

## Build executable for Kasper's global unit tests

add_executable (UnitTestKasper ${GTEST_MAIN_SOURCE})
target_link_libraries (UnitTestKasper ${UNIT_TEST_LIBS})
# Important: we need this to avoid gcc throwing out the unit test symbols during linking
# set_target_properties(UnitTestKasper PROPERTIES LINK_FLAGS "-Wl,--no-as-needed" )
kasper_install_executables(UnitTestKasper)

add_test(NAME UnitTestKasper COMMAND UnitTestKasper)
