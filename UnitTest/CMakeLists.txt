cmake_minimum_required( VERSION ${CMAKE_MINIMUM_VERSION} )

# UnitTest version
set( MODULE_VERSION_MAJOR 0 )
set( MODULE_VERSION_MINOR 2 )
set( MODULE_VERSION_PATCH 6 )
set( MODULE_VERSION "${MODULE_VERSION_MAJOR}.${MODULE_VERSION_MINOR}.${MODULE_VERSION_PATCH}" )

#project( UnitTest VERSION ${MODULE_VERSION} )
project( UnitTest )

include( KasperDefaults )


# module paths
kasper_module_paths( UnitTest )

include_directories(BEFORE ${CMAKE_CURRENT_BINARY_DIR})

## GoogleTest main module

# Add GoogleTest directly to our build. This defines
# the gtest and gtest_main targets.
add_subdirectory(GoogleTest)

## MockUps sub-modules

add_subdirectory( MockUps )

add_library( UnitTestBase INTERFACE )
target_include_directories( UnitTestBase INTERFACE
    $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/Include>
    $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/MockUps/Include>
    $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/GoogleTest/include>
)
target_link_libraries( UnitTestBase INTERFACE gtest )
kasper_install_libraries( UnitTestBase )

set(GTEST_MAIN_SOURCE ${CMAKE_CURRENT_SOURCE_DIR}/GoogleTest//src/gtest_main.cc)

## Kasper sub-modules

if( BUILD_KOMMON )
    add_subdirectory( Kommon )
    list(APPEND UNIT_TEST_LIBS KommonUnitTests)
endif()

if( BUILD_KGEOBAG )
    add_subdirectory( KGeoBag )
    list(APPEND UNIT_TEST_LIBS KGeoBagUnitTests)
endif()

if( BUILD_KEMFIELD )
    add_subdirectory( KEMField )
    list(APPEND UNIT_TEST_LIBS KEMFieldUnitTests)
endif()

if( BUILD_KASSIOPEIA )
    add_subdirectory( Kassiopeia )
    list(APPEND UNIT_TEST_LIBS KassiopeiaUnitTests)
endif()


## Global Kasper module

add_subdirectory( Kasper )
kasper_install_module()
