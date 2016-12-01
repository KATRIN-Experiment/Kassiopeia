kasper_find_vtk()
if( VTK_MAJOR_VERSION GREATER 5 )
    add_definitions(-DVTK6)
endif()

# internal
kasper_internal_include_directories( Utility )

# headers
set( KOMMON_VTK_HEADER_FILENAMES
        Utility/KVTKWindow.h
        Utility/KVTKPainter.h
        Utility/KVTKWindowBuilder.h
)

# source
set( KOMMON_VTK_SOURCE_FILENAMES
        Utility/KVTKWindow.cxx
        Utility/KVTKPainter.cxx
        Utility/KVTKWindowBuilder.cxx
)

# propagate path variable to parent scope
kasper_install_headers( ${KOMMON_VTK_HEADER_FILENAMES} )

add_library( KommonVtk SHARED ${KOMMON_VTK_SOURCE_FILENAMES} )
target_link_libraries( KommonVtk ${VTK_LIBRARIES} )
kasper_install_libraries( KommonVtk )

