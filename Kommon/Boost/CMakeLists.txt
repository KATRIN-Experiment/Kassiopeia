find_package( Boost ${BOOST_MINIMUM_VERSION} REQUIRED COMPONENTS filesystem system )

# include directories

# headers
set(KOMMON_BOOST_HEADER_FILES
    Utility/KAccumulators.h
    Utility/KFastMath.h
    Utility/KFunctionCache.h
    Utility/KPathUtils.h
    Utility/KStringUtils.h
)

# headers
set(KOMMON_BOOST_SOURCE_FILES
    Utility/KPathUtils.cxx
    Utility/KStringUtils.cxx
    Utility/KBoostStacktracePrinter.cxx
)

# # propagate path variable to parent scope
# kasper_append_paths( KOMMON_HEADER_FILES ${KOMMON_BOOST_HEADER_FILES} )

add_library( KommonBoost SHARED
    ${KOMMON_BOOST_SOURCE_FILES} ${KOMMON_BOOST_HEADER_FILES} )

# get header paths from collected header files
foreach(HEADER ${KOMMON_BOOST_HEADER_FILES})
    get_filename_component(DIRNAME ${CMAKE_CURRENT_SOURCE_DIR}/${HEADER} DIRECTORY)
    target_include_directories(KommonBoost PUBLIC $<BUILD_INTERFACE:${DIRNAME}>)
endforeach(HEADER)
target_include_directories(KommonBoost PUBLIC $<INSTALL_INTERFACE:include>)

target_compile_definitions(KommonBoost PUBLIC KASPER_USE_BOOST)
target_link_libraries(KommonBoost
    PUBLIC
        KommonBase
        Boost::boost
        Boost::system
        Boost::filesystem
)

# Fixes Boost build error on some Mac/Cygwin installations,
# see: https://github.com/boostorg/stacktrace/issues/88
if(UNIX OR APPLE OR MINGW)
    target_compile_definitions(KommonBoost PRIVATE _GNU_SOURCE)
endif()

kasper_install_libraries( KommonBoost )
kasper_install_headers( ${KOMMON_BOOST_HEADER_FILES} )
