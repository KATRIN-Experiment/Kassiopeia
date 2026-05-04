cmake_minimum_required( VERSION ${CMAKE_MINIMUM_VERSION} )

# UnitTest version
set( MODULE_VERSION_MAJOR 0 )
set( MODULE_VERSION_MINOR 2 )
set( MODULE_VERSION_PATCH 7 )
set( MODULE_VERSION "${MODULE_VERSION_MAJOR}.${MODULE_VERSION_MINOR}.${MODULE_VERSION_PATCH}" )

#project( UnitTest VERSION ${MODULE_VERSION} )
project( UnitTest )

include( KasperDefaults )

# module paths
kasper_module_paths( UnitTest )

# dependencies
find_package(Boost ${BOOST_MINIMUM_VERSION} REQUIRED COMPONENTS filesystem)

include_directories(BEFORE ${CMAKE_CURRENT_BINARY_DIR})

## doctest main module

## MockUps sub-modules

add_subdirectory( MockUps )

add_library( UnitTestBase INTERFACE )
target_include_directories( UnitTestBase INTERFACE
    $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/Include>
    $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/MockUps/Include>
    $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/Doctest/include>
)
kasper_install_libraries( UnitTestBase )

set(DOCTEST_MAIN_SOURCE ${CMAKE_CURRENT_SOURCE_DIR}/Doctest/src/doctest_main.cc)

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
