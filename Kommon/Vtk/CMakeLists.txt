find_package(VTK CONFIG REQUIRED)

if( VTK_MAJOR_VERSION GREATER 6 )
    find_package(VTK CONFIG COMPONENTS
        vtkChartsCore
        vtkCommonCore
        vtkCommonDataModel
        vtkInteractionStyle
        vtkIOGeometry
        vtkIOXML
        vtkRenderingContext2D
        vtkRenderingContextOpenGL2
        vtkRenderingCore
        vtkRenderingFreeType
        vtkRenderingOpenGL2
        vtkViewsContext2D
        QUIET
    )
else()
    find_package(VTK CONFIG COMPONENTS
        vtkChartsCore
        vtkCommonCore
        vtkCommonDataModel
        vtkInteractionStyle
        vtkIOGeometry
        vtkIOXML
        vtkRenderingContext2D
        #vtkRenderingContextOpenGL
        vtkRenderingCore
        vtkRenderingFreeType
        vtkRenderingOpenGL
        vtkViewsContext2D
        QUIET
    )
endif()
# find_package(VTK CONFIG COMPONENTS
#     vtkChartsCore
#     vtkCommonCore
#     vtkCommonDataModel
#     vtkInteractionStyle
#     vtkIOGeometry
#     vtkIOXML
#     vtkRenderingContext2D
#     vtkRenderingContextOpenGL2
#     vtkRenderingCore
#     vtkRenderingFreeType
#     vtkRenderingGL2PSOpenGL2
#     vtkRenderingOpenGL2
#     vtkViewsContext2D
#     QUIET
# )

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
