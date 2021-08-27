# include directories

# headers
set(KOMMON_BOOST_HEADER_FILES
    Utility/KAccumulators.h
    Utility/KFastMath.h
    Utility/KFunctionCache.h
    Utility/KPathUtils.h
    Utility/KStringUtils.h
)

# propagate path variable to parent scope
kasper_append_paths( KOMMON_HEADER_FILES ${KOMMON_BOOST_HEADER_FILES} )
