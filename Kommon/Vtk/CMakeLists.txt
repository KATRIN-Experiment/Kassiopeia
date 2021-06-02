find_package(VTK CONFIG REQUIRED)

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

add_library( KommonVtk SHARED
    ${KOMMON_VTK_SOURCE_FILENAMES} ${KOMMON_VTK_HEADER_FILENAMES})

# get header paths from collected header files
foreach(HEADER ${KOMMON_VTK_HEADER_FILENAMES})
    get_filename_component(DIRNAME ${CMAKE_CURRENT_SOURCE_DIR}/${HEADER} DIRECTORY)
    target_include_directories(KommonVtk PUBLIC $<BUILD_INTERFACE:${DIRNAME}>)
endforeach(HEADER)
target_include_directories(KommonVtk PUBLIC $<INSTALL_INTERFACE:include>)

# VTK should *really* use proper CMake targets at this point :-/
target_include_directories(KommonVtk PUBLIC ${VTK_INCLUDE_DIRS})

target_compile_definitions(KommonVtk PUBLIC KASPER_USE_VTK)

target_link_libraries( KommonVtk
    PUBLIC
        Kommon
        ${VTK_LIBRARIES}
)

if( VTK_MAJOR_VERSION GREATER 6 )
    if( CMAKE_CXX_COMPILER_ID STREQUAL "Clang" )
        target_compile_options(KommonVtk PUBLIC -Wno-inconsistent-missing-override)  # FIXME: vtk doesn't care about warnings, it seems
    endif()
endif()

if( VTK_MAJOR_VERSION GREATER 5 )
    target_compile_definitions(KommonVtk PRIVATE VTK6)
endif()

kasper_install_libraries( KommonVtk )
kasper_install_headers( ${KOMMON_VTK_HEADER_FILENAMES} )
