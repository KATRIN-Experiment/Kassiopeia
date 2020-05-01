# CMakeLists for KGeoBag/UnitTest
# Author: J. Behrens

kasper_find_module( Kommon )
kasper_find_module( KGeoBag )

enable_testing()

set( gtest_SOURCE_DIR
    ${CMAKE_CURRENT_SOURCE_DIR}/../GoogleTest
)

include_directories(
    ${CMAKE_CURRENT_SOURCE_DIR}/Include
    ${gtest_SOURCE_DIR}/include
    ${Kommon_INCLUDE_DIRS}
    ${KGeoBag_INCLUDE_DIRS}
)

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
target_link_libraries( KGeoBagUnitTests
	gtest
    ${Boost_LIBRARIES}
    ${Kommon_LIBRARIES}
    ${Kommon_Vtk_LIBRARIES}
    ${KGeoBag_LIBRARIES}
)
kasper_install_libraries( KGeoBagUnitTests )


## Build executable for KGeoBag's unit tests

add_executable (UnitTestKGeoBag ${gtest_SOURCE_DIR}/src/gtest_main.cc)
target_link_libraries (UnitTestKGeoBag KGeoBagUnitTests)
# Important: we need this to avoid gcc throwing out the unit test symbols during linking
# set_target_properties (UnitTestKGeoBag PROPERTIES LINK_FLAGS "-Wl,--no-as-needed")
kasper_install_executables (UnitTestKGeoBag)

add_test (NAME UnitTestKGeoBag COMMAND UnitTestKGeoBag)
