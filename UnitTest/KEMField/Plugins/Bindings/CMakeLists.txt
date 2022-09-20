# CMakeLists for UnitTest/KEMField/Plugins/Bindings
# Author: J. Behrens adapted for KEMField W. Gosda

enable_testing()

set (UNIT_TESTS_HEADERFILES
    #${CMAKE_CURRENT_SOURCE_DIR}/include/KSmartPointerReleaseTest.hh
    ${CMAKE_CURRENT_SOURCE_DIR}/include/KEMToolboxFixture.hh
)

set (UNIT_TESTS_SOURCEFILES
    ${CMAKE_CURRENT_SOURCE_DIR}/src/KEMToolboxTest.cc
    ${CMAKE_CURRENT_SOURCE_DIR}/src/KEMToolboxFixture.cc
    #${CMAKE_CURRENT_SOURCE_DIR}/src/KSmartPointerReleaseTest.cc
    #${CMAKE_CURRENT_SOURCE_DIR}/src/KSmartPointerRelease.cc
)

# options not accessible from here
#if(KEMField_USE_VTK)
    set (UNIT_TESTS_SOURCEFILES ${UNIT_TESTS_SOURCEFILES}
        ${CMAKE_CURRENT_SOURCE_DIR}/src/KBoundaryField_Visitor.cc
    )
    set (UNIT_TESTS_LINK_LIBRARIES ${UNIT_TESTS_LINK_LIBRARIES}
        KEMVTKPart2
    )
#endif(KEMField_USE_VTK)

# Build (static) library for KEMField's binding's unit tests

add_library( KEMBindingsUnitTests SHARED ${UNIT_TESTS_SOURCEFILES} ${UNIT_TESTS_HEADERFILES} )
target_include_directories( KEMBindingsUnitTests
    PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/include ${CMAKE_CURRENT_SOURCE_DIR}/include )
target_link_libraries (KEMBindingsUnitTests
    PUBLIC
        UnitTestBase
        KEMFieldBindings
)
kasper_install_libraries (KEMBindingsUnitTests)


# Build executable for KEMField's unit tests

add_executable (UnitTestKEMBindings ${GTEST_MAIN_SOURCE})
target_link_libraries (UnitTestKEMBindings KEMBindingsUnitTests)
kasper_install_executables (UnitTestKEMBindings)

add_test (NAME UnitTestKEMBindings COMMAND UnitTestKEMBindings)
