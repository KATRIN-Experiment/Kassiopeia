# include directories
kasper_internal_include_directories(
    Functions
    Geometry
    Logging
    File
    Initialization
    Typelists
    Units
    Utility
    ../Extensions/Root/Initialization
)

# headers
set(KOMMON_CORE_HEADER_FILES
    Typelists/KTypeNull.h
    Typelists/KTypeList.h
    Typelists/KTypeChain.h
    Typelists/KTypeOperation.h
    Units/KDimensions.h
    Units/KUnits.h
    File/KFileMessage.h
    File/KFile.h
    File/KTextFile.h
    File/KPathResolver.h
    Initialization/KInitializationMessage.hh
    Initialization/KToken.hh
    Initialization/KTypedTokens.hh
    Initialization/KProcessor.hh
    Initialization/KCommandLineTokenizer.hh
    Initialization/KXMLTokenizer.hh
    Initialization/KChattyProcessor.hh
    Initialization/KVariableProcessor.hh
    Initialization/KIncludeProcessor.hh
    Initialization/KLoopProcessor.hh
    Initialization/KConditionProcessor.hh
    Initialization/KSerializationProcessor.hh
    Initialization/KPrintProcessor.hh
    Initialization/KTagProcessor.hh
    Initialization/KContainer.hh
    Initialization/KAttributeBase.hh
    Initialization/KAttribute.hh
    Initialization/KElementBase.hh
    Initialization/KSimpleElement.hh
    Initialization/KComplexElement.hh
    Initialization/KElementProcessor.hh
    Initialization/KVariant.h
    Initialization/KArgumentList.h
    Initialization/KXMLInitializer.hh
    Utility/Gnuplot.hpp
    Utility/KException.h
    Utility/KRandom.h
    Utility/KRandomBuilder.h
    Utility/KConst.h
    Utility/KConsoleMuter.h
    Utility/KHash.h
    Utility/KHistogram.h
    Utility/KMathConvolution.h
    Utility/KMathGamma.h
    Utility/KMathKahanSum.h
    Utility/KMathLagrangeInterpolator.h
    Utility/KMathIntegrator.h
    Utility/KMathIntegrator2D.h
    Utility/KMathIntegratorFerenc.h
    Utility/KMathLagrangeInterpolator.h
    Utility/KMathOperands.h
    Utility/KMathRegulaFalsi.h
    Utility/KMathShepardInterpolator.h
    Utility/KNonCopyable.h
    Utility/KNumeric.h
    Utility/KField.h
    Utility/KInitializer.h
    Utility/KSingleton.h
    Utility/KSmartPointer.h
    Utility/KTypeTraits.h
    Utility/KMessage.h
    Utility/KUtilityMessage.h
    Utility/KNamed.h
    Utility/KTagged.h
    Utility/KWindow.h
    Utility/KPainter.h
    Utility/KMathBilinearInterpolator.h
    Utility/KHashMap.h
    Utility/KApplication.h
    Utility/KApplicationRunner.h
    Utility/KApplicationRunnerBuilder.h
    Utility/KRoot.h
    Utility/KToolbox.h
    Utility/KMessageBuilder.h
    Utility/KNamedBuilder.h
    Utility/KProgressDisplay.h
    Utility/PositiveValue.h
    Utility/Printable.h
    Logging/KLogger.h
)

# sources
set(KOMMON_CORE_SOURCE_FILES
    File/KFileMessage.cxx
    File/KFile.cxx
    File/KTextFile.cxx
    File/KPathResolver.cxx
    Initialization/KInitializationMessage.cc
    Initialization/KToken.cc
    Initialization/KProcessor.cc
    Initialization/KXMLTokenizer.cc
    Initialization/KCommandLineTokenizer.cc
    Initialization/KChattyProcessor.cc
    Initialization/KVariableProcessor.cc
    Initialization/KIncludeProcessor.cc
    Initialization/KLoopProcessor.cc
    Initialization/KConditionProcessor.cc
    Initialization/KSerializationProcessor.cc
    Initialization/KPrintProcessor.cc
    Initialization/KTagProcessor.cc
    Initialization/KContainer.cc
    Initialization/KAttributeBase.cc
    Initialization/KElementBase.cc
    Initialization/KElementProcessor.cc
    Initialization/KVariant.cxx
    Initialization/KArgumentList.cxx
    Initialization/KXMLInitializer.cc
    Units/KDimensions.cxx
    Units/KUnits.cxx
    Utility/Gnuplot.cpp
    Utility/KRandomBuilder.cxx
    Utility/KMessage.cxx
    Utility/KUtilityMessage.cxx
    Utility/KNamed.cxx
    Utility/KTagged.cxx
    Utility/KWindow.cxx
    Utility/KPainter.cxx
    Utility/KApplication.cxx
    Utility/KApplicationRunner.cxx
    Utility/KApplicationRunnerBuilder.cxx
    Utility/KRoot.cxx
    Utility/KToolbox.cxx
    Utility/KMessageBuilder.cxx
    Utility/KNamedBuilder.cxx
    Utility/Printable.cxx
    Logging/KLogger.cxx
)

set_source_files_properties(Initialization/KVariant.cxx PROPERTIES COMPILE_FLAGS -D__GLIBCXX_USE_CXX11_ABI=0)

if( LOG4CXX_FOUND )
    if (NOT CMAKE_BUILD_TYPE OR CMAKE_BUILD_TYPE MATCHES "DEBUG" OR CMAKE_BUILD_TYPE MATCHES "RELWITHDEBINFO" )
        set (LEVEL "DEBUG")
    else()
        set (LEVEL "INFO")
    endif()

    configure_file( Logging/log4cxx.properties.in ${CMAKE_CURRENT_BINARY_DIR}/log4cxx.properties @ONLY )
    kasper_install_optional( ${Kommon_CONFIG_INSTALL_DIR} ${CMAKE_CURRENT_BINARY_DIR}/log4cxx.properties )
endif()

# propagate path variable to parent scope
kasper_append_paths( KOMMON_HEADER_FILES ${KOMMON_CORE_HEADER_FILES} )
kasper_append_paths( KOMMON_SOURCE_FILES ${KOMMON_CORE_SOURCE_FILES} )
