cmake_minimum_required( VERSION ${CMAKE_MINIMUM_VERSION} )

# UnitTest version
set( MODULE_VERSION_MAJOR 0 )
set( MODULE_VERSION_MINOR 2 )
set( MODULE_VERSION_PATCH 0 )
set( MODULE_VERSION "${MODULE_VERSION_MAJOR}.${MODULE_VERSION_MINOR}.${MODULE_VERSION_PATCH}" )

project( UnitTest )

include( KasperDefaults )


# module paths
kasper_module_paths( UnitTest )

include_directories(BEFORE ${CMAKE_CURRENT_BINARY_DIR})

## GoogleTest main module

# Add GoogleTest directly to our build. This defines
# the gtest and gtest_main targets.
add_subdirectory(GoogleTest)

if (CMAKE_VERSION VERSION_LESS 2.8.11)
  include_directories("${CMAKE_CURRENT_SOURCE_DIR}/../GoogleTest/include")
endif()

include_directories(
    ${CMAKE_CURRENT_SOURCE_DIR}/Include
    ${CMAKE_CURRENT_SOURCE_DIR}/MockUps/Include
)


## MockUps sub-modules

add_subdirectory( MockUps )

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
