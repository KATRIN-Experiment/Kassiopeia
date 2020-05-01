# CMakeLists for UnitTest/KEMField/Plugins/Bindings
# Author: J. Behrens adapted for KEMField W. Gosda

enable_testing()

#set( gtest_SOURCE_DIR
#    ${CMAKE_CURRENT_SOURCE_DIR}/../GoogleTest
#)

include_directories(
    ${CMAKE_CURRENT_SOURCE_DIR}/include
    ${KEMField_INCLUDE_DIRS}
)

set (UNIT_TESTS_HEADERFILES
    ${CMAKE_CURRENT_SOURCE_DIR}/include/KSmartPointerReleaseTest.hh
    ${CMAKE_CURRENT_SOURCE_DIR}/include/KEMToolboxFixture.hh
)

set (UNIT_TESTS_SOURCEFILES
    ${CMAKE_CURRENT_SOURCE_DIR}/src/KEMToolboxTest.cc
    ${CMAKE_CURRENT_SOURCE_DIR}/src/KEMToolboxFixture.cc
    ${CMAKE_CURRENT_SOURCE_DIR}/src/KSmartPointerReleaseTest.cc
    ${CMAKE_CURRENT_SOURCE_DIR}/src/KSmartPointerRelease.cc
)

set (UNIT_TESTS_LINK_LIBRARIES
    gtest
    ${Boost_LIBRARIES}
    ${Kommon_LIBRARIES}
    ${KEMField_LIBRARIES}
)

# options not accessible from here
#if(@PROJECT_NAME@_USE_VTK)
    set (UNIT_TESTS_SOURCEFILES ${UNIT_TESTS_SOURCEFILES}
        ${CMAKE_CURRENT_SOURCE_DIR}/src/KBoundaryField_Visitor.cc
    )
    set (UNIT_TESTS_LINK_LIBRARIES ${UNIT_TESTS_LINK_LIBRARIES}
        KEMVTKPart2
    )
#endif()

# Build (static) library for KEMField's binding's unit tests

add_library( KEMBindingsUnitTests SHARED ${UNIT_TESTS_SOURCEFILES} ${UNIT_TESTS_HEADERFILES} )
target_link_libraries (KEMBindingsUnitTests
    ${UNIT_TESTS_LINK_LIBRARIES}
)
kasper_install_libraries (KEMBindingsUnitTests)


# Build executable for KEMField's unit tests

add_executable (UnitTestKEMBindings ${gtest_SOURCE_DIR}/src/gtest_main.cc)
target_link_libraries (UnitTestKEMBindings gtest KEMBindingsUnitTests)
kasper_install_executables (UnitTestKEMBindings)

add_test (NAME UnitTestKEMBindings COMMAND UnitTestKEMBindings)
