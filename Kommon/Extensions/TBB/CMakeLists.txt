# headers
set( KOMMON_TBB_HEADER_FILENAMES
	Utility/KMathIntegratorThreaded.h
)

# source
set( KOMMON_TBB_SOURCE_FILENAMES
    Utility/KMathIntegratorThreaded.cxx
)

# external
find_package( TBB REQUIRED )

kasper_external_include_directories( ${TBB_INCLUDE_DIR} )
kasper_internal_include_directories( Utility )

# link
add_library( KommonTBB SHARED ${KOMMON_TBB_SOURCE_FILENAMES} )
target_link_libraries( KommonTBB KommonCore ${TBB_LIBRARY} )

# install
kasper_install_libraries( KommonTBB )
kasper_install_headers( ${KOMMON_TBB_HEADER_FILENAMES} )
