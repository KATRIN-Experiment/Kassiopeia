# CMakeLists for Kommon/UnitTest
# Author: J. Behrens

kasper_find_module( Kommon )

enable_testing()

set( gtest_SOURCE_DIR ../GoogleTest )

if (KASPER_USE_TBB)
    find_package(TBB REQUIRED)
    add_definitions(-DTBB)
endif ()

if (KASPER_USE_ROOT)
    find_package(ROOT REQUIRED)
    add_definitions(-DROOT)
endif ()

if (KASPER_USE_BOOST)
    find_package(Boost REQUIRED)
    add_definitions(-DBOOST)
endif()

include_directories(
    ${CMAKE_CURRENT_SOURCE_DIR}/Include
    ${gtest_SOURCE_DIR}/include
    ${Kommon_INCLUDE_DIRS}
    ${TBB_INCLUDE_DIR}
)

set (UNIT_TESTS_SOURCEFILES
    Math.cxx
    Integrator.cxx
    KComplexElement.cxx
    StringUtils.cxx
    TestXML.cxx
)

## Build (static) library for Kommon's unit tests

add_library (KommonUnitTests SHARED ${UNIT_TESTS_SOURCEFILES} ${UNIT_TESTS_HEADERFILES})
target_link_libraries (KommonUnitTests
    gtest
    ${Kommon_LIBRARIES}
    ${TBB_LIBRARY}
)
kasper_install_libraries (KommonUnitTests)

# xml configs used for unit tests
set( TEST_CONFIG_FILES
    ConfigFiles/TestXMLTokenizer.xml
    ConfigFiles/TestXMLVariables.xml
    ConfigFiles/TestXMLFormulas.xml
    ConfigFiles/TestXMLIncludes.xml
    ConfigFiles/TestXMLLoops.xml
    ConfigFiles/TestXMLConditions.xml
    ConfigFiles/TestXMLElements.xml
    ConfigFiles/TestXMLSerialization.xml
    ConfigFiles/TestXMLPrint.xml
)

kasper_install_config( ${TEST_CONFIG_FILES} )

## Build executable for Kommon's unit tests

add_executable (UnitTestKommon ${gtest_SOURCE_DIR}/src/gtest_main.cc)
target_link_libraries (UnitTestKommon gtest KommonUnitTests)
# Important: we need this to avoid gcc throwing out the unit test symbols during linking
# set_target_properties (UnitTestKommon PROPERTIES LINK_FLAGS "-Wl,--no-as-needed")
kasper_install_executables (UnitTestKommon)

add_test (NAME UnitTestKommon COMMAND UnitTestKommon)
