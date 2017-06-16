# include directories
kasper_internal_include_directories( Utility )

# headers
set(KOMMON_BOOST_HEADER_FILES
	Utility/KAccumulators.h
	Utility/KFunctionCache.h
	Utility/KStringUtils.h
	Utility/KFastMath.h
)

# propagate path variable to parent scope
kasper_append_paths( KOMMON_HEADER_FILES ${KOMMON_BOOST_HEADER_FILES} )
