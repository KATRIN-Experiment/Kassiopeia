# internal
kasper_internal_include_directories(
    Utility
)

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


# propagate path variable to parent scope
kasper_append_paths( KOMMON_HEADER_FILES ${KOMMON_GSL_HEADER_FILES} )
kasper_append_paths( KOMMON_SOURCE_FILES ${KOMMON_GSL_SOURCE_FILES} )
