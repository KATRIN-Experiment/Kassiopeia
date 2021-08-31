# CMakeLists for Kommon/UnitTest
# Author: J. Behrens

enable_testing()

set (UNIT_TESTS_SOURCEFILES
    Math.cxx
    Integrator.cxx
    KComplexElement.cxx
    StringUtils.cxx
    TestXML.cxx
)

## Build (static) library for Kommon's unit tests

add_library (KommonUnitTests SHARED
    ${UNIT_TESTS_SOURCEFILES} ${UNIT_TESTS_HEADERFILES})
target_include_directories( KommonUnitTests
    PRIVATE ${CMAKE_CURRENT_SOURCE_DIR})

target_link_libraries (KommonUnitTests
    PUBLIC
        UnitTestBase
        Kommon
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

add_executable (UnitTestKommon ${GTEST_MAIN_SOURCE})
target_link_libraries (UnitTestKommon KommonUnitTests)
kasper_install_executables (UnitTestKommon)

add_test (NAME UnitTestKommon COMMAND UnitTestKommon)
