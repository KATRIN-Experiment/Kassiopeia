# headers
set(KOMMON_CORE_HEADER_FILES
    File/KTextFile.h
    File/KPathResolver.h

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
    Initialization/KXMLInitializer.hh

    Initialization/KArgumentList.h

    Bindings/KApplicationRunnerBuilder.h
    Bindings/KMessageBuilder.h
    Bindings/KNamedBuilder.h
    Bindings/KRandomBuilder.h
    Bindings/KNamedRandomGeneratorBuilder.h
)

# sources
set(KOMMON_CORE_SOURCE_FILES
    File/KTextFile.cxx
    File/KPathResolver.cxx

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
    Initialization/KXMLInitializer.cc

    Initialization/KArgumentList.cxx

    Bindings/KApplicationRunnerBuilder.cxx
    Bindings/KMessageBuilder.cxx
    Bindings/KNamedBuilder.cxx
    Bindings/KRandomBuilder.cxx
    Bindings/KNamedRandomGeneratorBuilder.cxx
)

# # propagate path variable to parent scope
# kasper_append_paths( KOMMON_HEADER_FILES ${KOMMON_CORE_HEADER_FILES} )
# kasper_append_paths( KOMMON_SOURCE_FILES ${KOMMON_CORE_SOURCE_FILES} ${KOMMON_CORE_EXTRA_FILES})

# Fix build issues for certain source files
#set_property(SOURCE Initialization/KVariant.cxx APPEND PROPERTY COMPILE_DEFINITIONS __GLIBCXX_USE_CXX11_ABI=0)

add_library( KommonCore SHARED
    ${KOMMON_CORE_SOURCE_FILES} ${KOMMON_CORE_HEADER_FILES} )

# get header paths from collected header files
foreach(HEADER ${KOMMON_CORE_HEADER_FILES})
    get_filename_component(DIRNAME ${CMAKE_CURRENT_SOURCE_DIR}/${HEADER} DIRECTORY)
    target_include_directories(KommonCore PUBLIC $<BUILD_INTERFACE:${DIRNAME}>)
endforeach(HEADER)
target_include_directories(KommonCore PUBLIC $<INSTALL_INTERFACE:include>)

target_link_libraries(KommonCore
    PUBLIC
        KommonBase
        KommonBoost
        ${CMAKE_DL_LIBS}
)

if( KASPER_USE_ROOT )
    target_link_libraries(KommonCore
        PRIVATE
            KommonRoot
    )
endif( KASPER_USE_ROOT )

kasper_install_libraries( KommonCore )
kasper_install_headers( ${KOMMON_CORE_HEADER_FILES} )
