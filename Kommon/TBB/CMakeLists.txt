# headers
set( KOMMON_TBB_HEADER_FILENAMES
    Utility/KMathIntegratorThreaded.h
)

# source
#set( KOMMON_TBB_SOURCE_FILENAMES
#    Utility/KMathIntegratorThreaded.cxx
#)

# propagate path variable to parent scope
kasper_append_paths( KOMMON_HEADER_FILES ${KOMMON_TBB_HEADER_FILENAMES} )
#kasper_append_paths( KOMMON_SOURCE_FILES ${KOMMON_TBB_SOURCE_FILENAMES} )
