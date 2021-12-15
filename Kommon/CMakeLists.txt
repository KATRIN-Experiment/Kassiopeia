cmake_minimum_required( VERSION ${CMAKE_MINIMUM_VERSION} )

# Kommon version
set( MODULE_VERSION_MAJOR 1 )
set( MODULE_VERSION_MINOR 5 )
set( MODULE_VERSION_PATCH 0 )
set( MODULE_VERSION "${MODULE_VERSION_MAJOR}.${MODULE_VERSION_MINOR}.${MODULE_VERSION_PATCH}" )

#project( Kommon VERSION ${MODULE_VERSION} )
project( Kommon )

include( KasperDefaults )

# paths
kasper_module_paths( Kommon )

# debugging
kasper_module_debug()

# external
find_package( LOG4CXX )
if(LOG4CXX_FOUND)
    option( Kommon_USE_Log4CXX "Enable advanced logging facilities, if LOG4CXX library is available." ON )
endif()
mark_as_advanced( CLEAR Kommon_USE_Log4CXX )

if(Kommon_USE_Log4CXX)
        set(LOGGER_CONFIGURATION "${Kommon_CONFIG_INSTALL_DIR}/log4cxx.properties" )
        set_source_files_properties(Core/Logging/KLogger.cxx PROPERTIES COMPILE_DEFINITIONS
            "LOG4CXX;LOGGER_CONFIGURATION=${LOGGER_CONFIGURATION}"
        )
endif()

option( KLogger_THROW_EXCEPTIONS "Throw exception when using KERROR/KFATAL macros." OFF )
if( KLogger_THROW_EXCEPTIONS )
    set_source_files_properties(Core/Logging/KLogger.cxx PROPERTIES COMPILE_DEFINITIONS
        "KLOGGER_THROW_EXCEPTIONS"
    )
endif()
mark_as_advanced( CLEAR KLogger_THROW_EXCEPTIONS )

set( KConst_REFERENCE_EPOCH 2021 CACHE STRING "Reference epoch for physical constants defined in 'KConst.h'. Follows PDG data releases." )
set_property(CACHE KConst_REFERENCE_EPOCH PROPERTY STRINGS 2006 2021)  # make drop-down list of allowed values

if(KConst_REFERENCE_EPOCH GREATER 2006)
    message(WARNING "*** Using updated physical constants ***\n"
    "Kasper was configured to use physical constants based on 2021 datasets. This will affect the outcome "
    "of some calculations. To restore *old* behavior, set KConst_REFERENCE_EPOCH=2006.")
elseif(KConst_REFERENCE_EPOCH LESS 2021)
    message(WARNING "*** Using outdated physical constants ***\n"
    "Kasper was configured to use physical constants based on 2006 datasets. This will affect the outcome "
    "of some calculations. To restore *default* behavior, set KConst_REFERENCE_EPOCH=2021.")
endif()

# to be populated by subdirectories
set(KOMMON_HEADER_FILES)
set(KOMMON_SOURCE_FILES)

# subdirectories
add_subdirectory( cmake )
add_subdirectory( Documentation )

if( KASPER_USE_BOOST )
    find_package( Boost ${BOOST_MINIMUM_VERSION} REQUIRED COMPONENTS filesystem system )
    add_subdirectory( Boost )
endif()

if( KASPER_USE_ROOT )
    find_package( ROOT ${ROOT_MINIMUM_VERSION} CONFIG REQUIRED )
    add_subdirectory( Root )
endif()

if( KASPER_USE_GSL )
    find_package( GSL REQUIRED )
    add_subdirectory( Gsl )
endif()

if( KASPER_USE_TBB )
    find_package( TBB REQUIRED )
    add_subdirectory( TBB )
endif()

add_subdirectory( Core )
add_subdirectory( Core/Binning )
add_subdirectory( Core/Maths )

# Fix build issues for certain source files
#set_property(SOURCE Core/Initialization/KVariant.cxx APPEND PROPERTY COMPILE_DEFINITIONS __GLIBCXX_USE_CXX11_ABI=0)
set_property(SOURCE Core/Initialization/tinyexpr.c APPEND PROPERTY COMPILE_OPTIONS "-Wno-array-bounds" )  # FIXME

kasper_install_headers( ${KOMMON_HEADER_FILES} )

add_library( Kommon SHARED
    ${KOMMON_SOURCE_FILES} ${KOMMON_HEADER_FILES} )

# get header paths from collected header files
foreach(HEADER ${KOMMON_HEADER_FILES})
    get_filename_component(DIRNAME ${HEADER} DIRECTORY)
    target_include_directories(Kommon PUBLIC $<BUILD_INTERFACE:${DIRNAME}>)
endforeach(HEADER)
target_include_directories(Kommon PUBLIC $<INSTALL_INTERFACE:include>)
target_compile_definitions(Kommon PUBLIC KConst_REFERENCE_EPOCH=${KConst_REFERENCE_EPOCH})

# Fixes Boost build error on some Mac/Cygwin installations,
# see: https://github.com/boostorg/stacktrace/issues/88
if(UNIX OR APPLE OR MINGW)
    target_compile_definitions(Kommon PRIVATE _GNU_SOURCE)
endif()

if( KASPER_USE_BOOST )
    target_compile_definitions(Kommon PUBLIC KASPER_USE_BOOST)
    target_link_libraries(Kommon
        PUBLIC
            Boost::boost
            Boost::system
        PRIVATE
            Boost::filesystem
    )
endif( KASPER_USE_BOOST )

if( KASPER_USE_ROOT )
    target_compile_definitions(Kommon PUBLIC KASPER_USE_ROOT)
    target_link_libraries( Kommon
        PUBLIC
            ROOT::Core ROOT::Graf ROOT::Gpad ROOT::Hist
        PRIVATE
            ROOT::MathCore
    )
endif(KASPER_USE_ROOT)

if( KASPER_USE_GSL )
    target_compile_definitions(Kommon PUBLIC KASPER_USE_GSL)
    target_link_libraries(Kommon
        PRIVATE
            GSL::gsl
    )
endif(KASPER_USE_GSL)

if( KASPER_USE_TBB )
    target_include_directories(Kommon PRIVATE ${TBB_INCLUDE_DIRS})
    target_compile_definitions(Kommon PUBLIC KASPER_USE_TBB)
    target_link_libraries(Kommon
        PRIVATE
            ${TBB_LIBRARIES}
    )
endif( KASPER_USE_TBB )

if( Kommon_USE_Log4CXX )
    target_include_directories( Kommon PRIVATE ${LOG4CXX_INCLUDE_DIR})
    target_link_libraries( Kommon
        PRIVATE
            ${LOG4CXX_LIBRARY}
    )
endif(Kommon_USE_Log4CXX)

kasper_install_libraries( Kommon )

# a distinct shared library "KommonVtk" is built here!
if( KASPER_USE_VTK )
        add_subdirectory( Vtk )
endif()

kasper_install_module()
