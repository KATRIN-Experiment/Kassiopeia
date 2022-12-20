# Setup VTK modules for use in CMake
# For details, see: https://vtk.org/doc/nightly/html/md__builds_gitlab-kitware-sciviz-ci_Documentation_Doxygen_ModuleMigration.html

find_package(VTK COMPONENTS vtkCommonCore)

if( VTK_MAJOR_VERSION GREATER 8 )
    set(KOMMON_VTK_COMPONENTS
        ChartsCore
        CommonCore
        CommonDataModel
        FiltersVerdict
        InteractionStyle
        IOGeometry
        IOXML
        RenderingContext2D
        RenderingContextOpenGL2
        RenderingCore
        RenderingFreeType
        RenderingOpenGL2
        ViewsContext2D
    )
else()
    set(KOMMON_VTK_COMPONENTS
        vtkChartsCore
        vtkCommonCore
        vtkCommonDataModel
        vtkFiltersVerdict
        vtkInteractionStyle
        vtkIOGeometry
        vtkIOXML
        vtkRenderingContext2D
        vtkRenderingContextOpenGL2
        vtkRenderingCore
        vtkRenderingFreeType
        vtkRenderingOpenGL2
        vtkViewsContext2D
    )
endif()

# needed in parent scope to write the KommonConfig.cmake file
set(KOMMON_VTK_COMPONENTS ${KOMMON_VTK_COMPONENTS} PARENT_SCOPE)

# Requires VTK 7 or above
find_package(VTK CONFIG COMPONENTS ${KOMMON_VTK_COMPONENTS} REQUIRED)
message(STATUS "VTK: Found version ${VTK_VERSION}.")

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
# The lines below are adapted from the `UseVTK.cmake` file, located at ${VTK_USE_FILE}

# Add compiler flags needed to use VTK.
separate_arguments(_VTK_REQUIRED_CXX_FLAGS UNIX_COMMAND "${VTK_REQUIRED_CXX_FLAGS}")
target_compile_options(KommonVtk PUBLIC ${_VTK_REQUIRED_CXX_FLAGS})
set(_target_type "$<TARGET_PROPERTY:TYPE>")
add_link_options( $<$<STREQUAL:"${_target_type}","EXECUTABLE">:${VTK_REQUIRED_EXE_LINKER_FLAGS}> )
add_link_options( $<$<STREQUAL:"${_target_type}","SHARED_LIBRARY">:${VTK_REQUIRED_SHARED_LINKER_FLAGS}> )
add_link_options( $<$<STREQUAL:"${_target_type}","MODULE_LIBRARY">:${VTK_REQUIRED_MODULE_LINKER_FLAGS}> )

# Add preprocessor definitions needed to use VTK.
set_property(DIRECTORY APPEND PROPERTY COMPILE_DEFINITIONS ${VTK_DEFINITIONS})

# Add include directories needed to use VTK.
target_include_directories(KommonVtk PUBLIC ${VTK_INCLUDE_DIRS})

# Add link directories needed to use VTK.
target_link_directories(KommonVtk PUBLIC ${VTK_LIBRARY_DIRS})

target_compile_definitions(KommonVtk PUBLIC KASPER_USE_VTK)
target_link_libraries( KommonVtk
    PUBLIC
        KommonCore
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

if( VTK_MAJOR_VERSION GREATER 8 )
    vtk_module_autoinit(TARGETS KommonVtk MODULES ${VTK_LIBRARIES})
endif()

kasper_install_libraries( KommonVtk )
kasper_install_headers( ${KOMMON_VTK_HEADER_FILENAMES} )
