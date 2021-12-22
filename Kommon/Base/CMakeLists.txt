# headers
set(KOMMON_BASE_HEADER_FILES
    File/KFileMessage.h
    File/KFile.h
    
    Initialization/KInitializationMessage.hh
    Initialization/KGlobals.hh
    
    Initialization/KToken.hh
    Initialization/KTypedTokens.hh
    Initialization/KProcessor.hh

    Initialization/KContainer.hh
    Initialization/KAttributeBase.hh
    Initialization/KAttribute.hh
    Initialization/KElementBase.hh
    Initialization/KSimpleElement.hh
    Initialization/KComplexElement.hh
    Initialization/KElementProcessor.hh
    
    Initialization/KApplicationRunner.h
    Initialization/KRoot.h
    Initialization/KToolbox.h
    
    Typelists/KTypeNull.h
    Typelists/KTypeList.h
    Typelists/KTypeChain.h
    Typelists/KTypeOperation.h

    Units/KDimensions.h
    Units/KUnits.h

    Random/KNamedRandomGenerator.h
    Random/KNamedRandomPrototype.h
    Random/KRandom.h
    Random/KRandomPrototype.h

    Utility/Gnuplot.hpp

    Utility/KUtilityMessage.h
    Utility/KAlgorithm.h
    Utility/KException.h
    Utility/KCast.h
    Utility/KConst.h
    Utility/KConst_2006.h
    Utility/KConsoleMuter.h
    Utility/KHash.h
    Utility/OstreamJoiner.h
    Utility/KHistogram.h
    Utility/KMakeArray.h
    Utility/KNonCopyable.h
    Utility/KNumeric.h
    Utility/KField.h
    Utility/KInitializer.h
    Utility/KSingleton.h
    Utility/KTypeTraits.h
    Utility/KVariadicOperations.h
    Utility/KNamed.h
    Utility/KTagged.h
    Utility/KWindow.h
    Utility/KPainter.h
    Utility/KHashMap.h
    Utility/KApplication.h
    Utility/KProgressDisplay.h
    Utility/PositiveValue.h
    Utility/Printable.h
    Utility/SimplyPrintableCollection.h
    Utility/KVariant.h
    Utility/KBaseStringUtils.h

    Logging/KLogger.h
    Logging/KMessage.h
)

# sources
set(KOMMON_BASE_SOURCE_FILES
    File/KFileMessage.cxx
    File/KFile.cxx
    
    Initialization/KInitializationMessage.cc
    Initialization/KGlobals.cc
    
    Initialization/KToken.cc
    Initialization/KProcessor.cc
    
    Initialization/KContainer.cc
    Initialization/KAttributeBase.cc
    Initialization/KElementBase.cc
    Initialization/KElementProcessor.cc
    
    Initialization/KApplicationRunner.cxx
    Initialization/KRoot.cxx
    Initialization/KToolbox.cxx
    
    Units/KDimensions.cxx
    Units/KUnits.cxx

    Utility/KUtilityMessage.cxx
    Utility/Gnuplot.cpp

    Utility/KNamed.cxx
    Utility/KTagged.cxx
    Utility/KWindow.cxx
    Utility/KPainter.cxx
    Utility/KApplication.cxx
    Utility/Printable.cxx
    Utility/KVariant.cxx
    Utility/KBaseStringUtils.cxx
    
    Logging/KLogger.cxx
    Logging/KMessage.cxx
)

# BEGIN LOGGING
# external: LOG4CXX
find_package( LOG4CXX )
if(LOG4CXX_FOUND)
    option( Kommon_USE_Log4CXX "Enable advanced logging facilities, if LOG4CXX library is available." ON )
endif()
mark_as_advanced( CLEAR Kommon_USE_Log4CXX )

if(Kommon_USE_Log4CXX)
        set(LOGGER_CONFIGURATION "${Kommon_CONFIG_INSTALL_DIR}/log4cxx.properties" )
        set_source_files_properties(Logging/KLogger.cxx PROPERTIES COMPILE_DEFINITIONS
            "LOG4CXX;LOGGER_CONFIGURATION=${LOGGER_CONFIGURATION}"
        )
endif()

if( LOG4CXX_FOUND )
    if (NOT CMAKE_BUILD_TYPE OR CMAKE_BUILD_TYPE MATCHES "DEBUG" OR CMAKE_BUILD_TYPE MATCHES "RELWITHDEBINFO" )
        set (LEVEL "DEBUG")
    else()
        set (LEVEL "INFO")
    endif()

    configure_file( Logging/log4cxx.properties.in ${CMAKE_CURRENT_BINARY_DIR}/log4cxx.properties @ONLY )
    kasper_install_optional( ${Kommon_CONFIG_INSTALL_DIR} ${CMAKE_CURRENT_BINARY_DIR}/log4cxx.properties )
endif()

option( KLogger_THROW_EXCEPTIONS "Throw exception when using KERROR/KFATAL macros." OFF )
if( KLogger_THROW_EXCEPTIONS )
    set_source_files_properties(Logging/KLogger.cxx PROPERTIES COMPILE_DEFINITIONS
        "KLOGGER_THROW_EXCEPTIONS"
    )
endif()
mark_as_advanced( CLEAR KLogger_THROW_EXCEPTIONS )
# END LOGGING

# # propagate path variable to parent scope
# kasper_append_paths( KOMMON_HEADER_FILES ${KOMMON_BASE_HEADER_FILES} )
# kasper_append_paths( KOMMON_SOURCE_FILES ${KOMMON_BASE_SOURCE_FILES} ${KOMMON_BASE_EXTRA_FILES})

add_library( KommonBase SHARED
    ${KOMMON_BASE_SOURCE_FILES} ${KOMMON_BASE_HEADER_FILES} )

# LOGGING
if( Kommon_USE_Log4CXX )
    target_include_directories(KommonBase PRIVATE ${LOG4CXX_INCLUDE_DIR})
    target_link_libraries(KommonBase
        PRIVATE
            ${LOG4CXX_LIBRARY}
    )
endif( Kommon_USE_Log4CXX )

# get header paths from collected header files
foreach(HEADER ${KOMMON_BASE_HEADER_FILES})
    get_filename_component(DIRNAME ${CMAKE_CURRENT_SOURCE_DIR}/${HEADER} DIRECTORY)
    target_include_directories(KommonBase PUBLIC $<BUILD_INTERFACE:${DIRNAME}>)
endforeach(HEADER)
target_include_directories(KommonBase PUBLIC $<INSTALL_INTERFACE:include>)


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

# Needed for physical constants defined in Utility/KConst.h
target_compile_definitions(KommonBase PUBLIC KConst_REFERENCE_EPOCH=${KConst_REFERENCE_EPOCH})

kasper_install_libraries( KommonBase )
kasper_install_headers( ${KOMMON_BASE_HEADER_FILES} )
