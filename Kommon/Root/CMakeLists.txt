find_package( ROOT ${ROOT_MINIMUM_VERSION} CONFIG REQUIRED )

# headers
set(KOMMON_ROOT_HEADER_FILES
    File/KRootFile.h
    Initialization/KFormulaProcessor.hh
    Utility/KTRandom.h
    Utility/KROOTPainter.h
    Utility/KROOTWindow.h
    Utility/KROOTPad.h
    Utility/KROOTWindowBuilder.h
    Utility/KROOTPadBuilder.h
)

# source
set(KOMMON_ROOT_SOURCE_FILES
    File/KRootFile.cxx
    Initialization/KFormulaProcessor.cc
    Utility/KTRandom.cxx
    Utility/KROOTPainter.cxx
    Utility/KROOTWindow.cxx
    Utility/KROOTPad.cxx
    Utility/KROOTWindowBuilder.cxx
    Utility/KROOTPadBuilder.cxx
)

# header added here to not include it in installation
set(KOMMON_ROOT_EXTRA_FILES
    Initialization/tinyexpr.c
    Initialization/tinyexpr.h
)

# # propagate path variable to parent scope
# kasper_append_paths( KOMMON_HEADER_FILES ${KOMMON_ROOT_HEADER_FILES} )
# kasper_append_paths( KOMMON_SOURCE_FILES ${KOMMON_ROOT_SOURCE_FILES} )

set_property(SOURCE Initialization/tinyexpr.c APPEND PROPERTY COMPILE_OPTIONS "-Wno-array-bounds" )  # FIXME

add_library( KommonRoot SHARED
    ${KOMMON_ROOT_SOURCE_FILES} ${KOMMON_ROOT_HEADER_FILES} ${KOMMON_ROOT_EXTRA_FILES} )

# get header paths from collected header files
foreach(HEADER ${KOMMON_ROOT_HEADER_FILES})
    get_filename_component(DIRNAME ${CMAKE_CURRENT_SOURCE_DIR}/${HEADER} DIRECTORY)
    target_include_directories(KommonRoot PUBLIC $<BUILD_INTERFACE:${DIRNAME}>)
endforeach(HEADER)
target_include_directories(KommonRoot PUBLIC $<INSTALL_INTERFACE:include>)

target_link_libraries (KommonRoot
    PRIVATE
        KommonBase
)

if( KASPER_USE_BOOST )
    target_link_libraries(KommonRoot
        PUBLIC
            KommonBoost
    )
endif( KASPER_USE_BOOST )

target_compile_definitions(KommonRoot PUBLIC KASPER_USE_ROOT)
target_link_libraries( KommonRoot
    PUBLIC
        ROOT::Core ROOT::Graf ROOT::Gpad ROOT::Hist
    PRIVATE
        ROOT::MathCore
)

kasper_install_libraries( KommonRoot )
kasper_install_headers( ${KOMMON_ROOT_HEADER_FILES} )
