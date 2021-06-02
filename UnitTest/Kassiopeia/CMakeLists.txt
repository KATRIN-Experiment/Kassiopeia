# CMakeLists for Kassiopeia/UnitTest
# Author: J. Behrens

enable_testing()

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
target_include_directories( KassiopeiaUnitTests
    PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/Include)
target_link_libraries (KassiopeiaUnitTests
    PUBLIC
        UnitTestBase
        KassiopeiaSimulation
        KassiopeiaVisualization

)
kasper_install_libraries (KassiopeiaUnitTests)


## Build executable for Kassiopeia's unit tests

add_executable (UnitTestKassiopeia ${GTEST_MAIN_SOURCE})
target_link_libraries (UnitTestKassiopeia KassiopeiaUnitTests)
kasper_install_executables (UnitTestKassiopeia)

add_test (NAME UnitTestKassiopeia COMMAND UnitTestKassiopeia)
