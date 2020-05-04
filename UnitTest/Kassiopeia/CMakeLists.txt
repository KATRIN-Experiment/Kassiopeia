# CMakeLists for Kassiopeia/UnitTest
# Author: J. Behrens

kasper_find_module( Kassiopeia )

enable_testing()

set( gtest_SOURCE_DIR
    ${CMAKE_CURRENT_SOURCE_DIR}/../GoogleTest
)

include_directories(
    ${CMAKE_CURRENT_SOURCE_DIR}/Include
    ${gtest_SOURCE_DIR}/include
    ${Kommon_INCLUDE_DIRS}
    ${Kassiopeia_INCLUDE_DIRS}
)

set (UNIT_TESTS_HEADERFILES
    ${CMAKE_CURRENT_SOURCE_DIR}/Include/Generators.h
    ${CMAKE_CURRENT_SOURCE_DIR}/Include/Operators.h
    ${CMAKE_CURRENT_SOURCE_DIR}/Include/Terminators.h
)

set (UNIT_TESTS_SOURCEFILES
    ${CMAKE_CURRENT_SOURCE_DIR}/Source/Messages.cxx
    ${CMAKE_CURRENT_SOURCE_DIR}/Source/Generators.cxx
    ${CMAKE_CURRENT_SOURCE_DIR}/Source/Operators.cxx
    ${CMAKE_CURRENT_SOURCE_DIR}/Source/Terminators.cxx
)


## Build (static) library for Kassiopeia's unit tests

add_library (KassiopeiaUnitTests SHARED ${UNIT_TESTS_SOURCEFILES} ${UNIT_TESTS_HEADERFILES})
target_link_libraries (KassiopeiaUnitTests
    gtest
    ${ROOT_LIBRARIES}
    ${Boost_LIBRARIES}
    ${GSL_LIBRARIES}
    ${KasperCommon_LIBRARIES}
    ${Kassiopeia_LIBRARIES}
)
kasper_install_libraries (KassiopeiaUnitTests)


## Build executable for Kassiopeia's unit tests

add_executable (UnitTestKassiopeia ${gtest_SOURCE_DIR}/src/gtest_main.cc)
target_link_libraries (UnitTestKassiopeia KassiopeiaUnitTests gtest)
# Important: we need this to avoid gcc throwing out the unit test symbols during linking
# set_target_properties (UnitTestKassiopeia PROPERTIES LINK_FLAGS "-Wl,--no-as-needed")
kasper_install_executables (UnitTestKassiopeia)

add_test (NAME UnitTestKassiopeia COMMAND UnitTestKassiopeia)
