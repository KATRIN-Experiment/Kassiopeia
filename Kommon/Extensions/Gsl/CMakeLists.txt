# headers
set( KOMMON_GSL_HEADER_FILENAMES
    Utility/KMathPolynomialSolver.h
    Utility/KMathBracketingSolver.h
)

# sources
set( KOMMON_GSL_SOURCE_FILENAMES
    Utility/KMathPolynomialSolver.cxx
    Utility/KMathBracketingSolver.cxx
)

# external
find_package( GSL REQUIRED )
kasper_external_include_directories(
	${GSL_INCLUDE_DIRS}
)
# internal
kasper_internal_include_directories(
	${CMAKE_CURRENT_SOURCE_DIR}/Utility
)

# link
add_library( KommonGsl SHARED ${KOMMON_GSL_SOURCE_FILENAMES} )
target_link_libraries( KommonGsl KommonCore ${GSL_LIBRARIES})

# install
kasper_install_libraries( KommonGsl )
kasper_install_headers( ${KOMMON_GSL_HEADER_FILENAMES} )
