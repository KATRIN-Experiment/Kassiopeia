cmake_minimum_required( VERSION ${CMAKE_MINIMUM_VERSION} )

# Kommon version
set( MODULE_VERSION_MAJOR 1 )
set( MODULE_VERSION_MINOR 4 )
set( MODULE_VERSION_PATCH 0 )
set( MODULE_VERSION "${MODULE_VERSION_MAJOR}.${MODULE_VERSION_MINOR}.${MODULE_VERSION_PATCH}" )

#project( Kommon VERSION ${MODULE_VERSION} )
project( Kommon )

#set( CMAKE_MODULE_PATH ${PROJECT_SOURCE_DIR}/cmake ${PROJECT_SOURCE_DIR}/GoogleTest/cmake )
include( KasperDefaults )


# paths
kasper_module_paths( Kommon )

# debugging
kasper_module_debug()
#if( Kommon_ENABLE_DEBUG )
#	add_cflag (Kommon_ENABLE_DEBUG)
#endif()

set(EXTERNAL_LIBRARIES)

# external
find_package( Log4CXX )
if(LOG4CXX_FOUND)
    option( Kommon_USE_Log4CXX "Enable advanced logging facilities, if LOG4CXX library is available." ON )
endif()
mark_as_advanced( CLEAR Kommon_USE_Log4CXX )

if(Kommon_USE_Log4CXX)
        set(LOGGER_CONFIGURATION "${Kommon_CONFIG_INSTALL_DIR}/log4cxx.properties" )
        set_source_files_properties(Core/Logging/KLogger.cxx PROPERTIES COMPILE_DEFINITIONS
            "LOG4CXX;LOGGER_CONFIGURATION=${LOGGER_CONFIGURATION}"
        )
        kasper_external_include_directories( ${LOG4CXX_INCLUDE_DIR} )
        list(APPEND EXTERNAL_LIBRARIES ${LOG4CXX_LIBRARY} )
endif()

option( KLogger_THROW_EXCEPTIONS "Throw exception when using KERROR/KFATAL macros." OFF )
if( KLogger_THROW_EXCEPTIONS )
        set_source_files_properties(Core/Logging/KLogger.cxx PROPERTIES COMPILE_DEFINITIONS
            "KLOGGER_THROW_EXCEPTIONS"
        )
endif()
mark_as_advanced( CLEAR KLogger_THROW_EXCEPTIONS )


# to be populated by subdirectories
set(KOMMON_HEADER_FILES)
set(KOMMON_SOURCE_FILES)

# subdirectories
add_subdirectory( cmake )
add_subdirectory( Documentation )

if( KASPER_USE_BOOST )
    find_package( Boost 1.43.0 COMPONENTS filesystem REQUIRED )
    if( Boost_INCLUDE_DIR )
        kasper_external_include_directories( ${Boost_INCLUDE_DIR} )
        list(APPEND EXTERNAL_LIBRARIES ${Boost_LIBRARIES} )
    else()
        list(APPEND EXTERNAL_LIBRARIES Boost::headers Boost::filesystem )
    endif()
    add_compile_definitions( KASPER_USE_BOOST )
    add_subdirectory( Boost )
endif()

if( KASPER_USE_ROOT )
    find_package( ROOT 5.24.0 REQUIRED )
    kasper_external_include_directories( ${ROOT_INCLUDE_DIR} )
    add_definitions( -DKommon_USE_ROOT )
    list(APPEND EXTERNAL_LIBRARIES ${ROOT_LIBRARIES} )
    add_compile_definitions( KASPER_USE_ROOT )
    add_subdirectory( Root )
endif()

if( KASPER_USE_GSL )
    find_package( GSL REQUIRED )
    kasper_external_include_directories( ${GSL_INCLUDE_DIRS} )
    list(APPEND EXTERNAL_LIBRARIES ${GSL_LIBRARIES} )
    add_compile_definitions( KASPER_USE_GSL )
    add_subdirectory( Gsl )
endif()

if( KASPER_USE_TBB )
    find_package( TBB REQUIRED )
    kasper_external_include_directories( ${TBB_INCLUDE_DIR} )
    add_compile_definitions( KASPER_USE_TBB )
    add_subdirectory( TBB )
endif()

add_subdirectory( Core )

kasper_include_default_dirs()
kasper_install_headers( ${KOMMON_HEADER_FILES} )

add_library( Kommon SHARED ${KOMMON_SOURCE_FILES} )
target_link_libraries( Kommon ${EXTERNAL_LIBRARIES} )
kasper_install_libraries( Kommon )

# a distinct shared library "KommonVtk" is built here!
if( KASPER_USE_VTK )
        add_subdirectory( Vtk )
endif()

kasper_install_module()
