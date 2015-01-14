# headers
set( KOMMON_VTK_HEADER_FILENAMES
    Utility/KVTKWindow.h
    Utility/KVTKPainter.h
)

# source
set( KOMMON_VTK_SOURCE_FILENAMES
    Utility/KVTKWindow.cxx
    Utility/KVTKPainter.cxx
)

# external
find_package( VTK REQUIRED )
include( ${VTK_USE_FILE} )
kasper_external_include_directories(
	${VTK_INCLUDE_DIRS}
)
if( VTK_MAJOR_VERSION GREATER 5 )
	set_property( SOURCE
	    Utility/KVTKWindow.cxx
	    Utility/KVTKPainter.cxx
    	APPEND PROPERTY COMPILE_DEFINITIONS VTK6
	)
endif()

# internal
kasper_internal_include_directories(
	${CMAKE_CURRENT_SOURCE_DIR}/Utility
)

# link
add_library( KommonVtk SHARED ${KOMMON_VTK_SOURCE_FILENAMES} )
target_link_libraries( KommonVtk KommonCore ${VTK_LIBRARIES} )

# install
kasper_install_libraries( KommonVtk )
kasper_install_headers( ${Kommon_USE_VTK_HEADERS} )