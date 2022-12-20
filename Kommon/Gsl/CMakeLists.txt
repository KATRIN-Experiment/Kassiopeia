find_package( GSL REQUIRED )

# headers
set(KOMMON_GSL_HEADER_FILES
    Utility/KMathPolynomialSolver.h
    Utility/KMathBracketingSolver.h
    Utility/KGslErrorHandler.h
)

# sources
set( KOMMON_GSL_SOURCE_FILES
    Utility/KMathPolynomialSolver.cxx
    Utility/KMathBracketingSolver.cxx
    Utility/KGslErrorHandler.cxx
)

# # propagate path variable to parent scope
# kasper_append_paths( KOMMON_HEADER_FILES ${KOMMON_GSL_HEADER_FILES} )
# kasper_append_paths( KOMMON_SOURCE_FILES ${KOMMON_GSL_SOURCE_FILES} )

add_library( KommonGsl SHARED
    ${KOMMON_GSL_SOURCE_FILES} ${KOMMON_GSL_HEADER_FILES} )

# get header paths from collected header files
foreach(HEADER ${KOMMON_GSL_HEADER_FILES})
    get_filename_component(DIRNAME ${CMAKE_CURRENT_SOURCE_DIR}/${HEADER} DIRECTORY)
    target_include_directories(KommonGsl PUBLIC $<BUILD_INTERFACE:${DIRNAME}>)
endforeach(HEADER)
target_include_directories(KommonGsl PUBLIC $<INSTALL_INTERFACE:include>)

target_compile_definitions(KommonGsl PUBLIC KASPER_USE_GSL)
target_link_libraries(KommonGsl
    PUBLIC
        KommonBase
        GSL::gsl
)

if( KASPER_USE_BOOST )
    target_link_libraries(KommonGsl
        PRIVATE
            KommonBoost
    )

    # see: https://github.com/boostorg/stacktrace/issues/88
    if(UNIX OR APPLE OR MINGW)
        target_compile_definitions(KommonGsl PRIVATE _GNU_SOURCE)
    endif()
endif( KASPER_USE_BOOST )

kasper_install_libraries( KommonGsl )
kasper_install_headers( ${KOMMON_GSL_HEADER_FILES} )
