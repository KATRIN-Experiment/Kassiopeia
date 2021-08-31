# CMakeLists for KGeoBag/UnitTest
# Author: J. Behrens

enable_testing()

set (UNIT_TESTS_HEADERFILES
    ${CMAKE_CURRENT_SOURCE_DIR}/Include/Shapes.h
)

set (UNIT_TESTS_SOURCEFILES
    ${CMAKE_CURRENT_SOURCE_DIR}/Source/Shapes_Surfaces.cxx
    ${CMAKE_CURRENT_SOURCE_DIR}/Source/Shapes_Volumes.cxx
    ${CMAKE_CURRENT_SOURCE_DIR}/Source/SpaceTree.cxx
)

## Build (static) library for KGeoBag's unit tests

add_library (KGeoBagUnitTests SHARED ${UNIT_TESTS_SOURCEFILES} ${UNIT_TESTS_HEADERFILES})
target_include_directories( KGeoBagUnitTests
    PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/Include )
target_link_libraries( KGeoBagUnitTests
    PUBLIC
        UnitTestBase
        KGeoBagCore
        KGeoBagShapes
        KGeoBagMathSpaceTree
)
kasper_install_libraries( KGeoBagUnitTests )


## Build executable for KGeoBag's unit tests

add_executable (UnitTestKGeoBag ${GTEST_MAIN_SOURCE})
target_link_libraries (UnitTestKGeoBag KGeoBagUnitTests)
kasper_install_executables (UnitTestKGeoBag)

add_test (NAME UnitTestKGeoBag COMMAND UnitTestKGeoBag)
