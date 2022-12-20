# CMakeLists for KEMField/Plugins/VTKPart2
# Author: W. Gosda

# This library is the second part of the vtk plugin. This split is necessary to
# avoid circular dependencies.
# The library is build even with VTK disabled to provide a function less empty
# version of the KVTKViewerAsBoundaryFieldVisitor to avoid parser errors.

#header files

set( VTK-PART2_HEADERFILES
    ${CMAKE_CURRENT_SOURCE_DIR}/include/KVTKViewerAsBoundaryFieldVisitor.hh
)

if( KEMField_USE_VTK )
    list( APPEND VTK-PART2_HEADERFILES
        ${CMAKE_CURRENT_SOURCE_DIR}/include/KElectrostaticPotentialmap.hh
        ${CMAKE_CURRENT_SOURCE_DIR}/include/KMagnetostaticFieldmap.hh
    )
endif( KEMField_USE_VTK )

#source files

set( VTK-PART2_SOURCEFILES
    ${CMAKE_CURRENT_SOURCE_DIR}/src/KVTKViewerAsBoundaryFieldVisitor.cc
)

if( KEMField_USE_VTK )
    list( APPEND VTK-PART2_SOURCEFILES
        ${CMAKE_CURRENT_SOURCE_DIR}/src/KElectrostaticPotentialmap.cc
        ${CMAKE_CURRENT_SOURCE_DIR}/src/KMagnetostaticFieldmap.cc
    )
endif( KEMField_USE_VTK )

#link libraries

add_library (KEMVTKPart2 SHARED
    ${VTK-PART2_SOURCEFILES} ${VTK-PART2_HEADERFILES})
target_include_directories(KEMVTKPart2
    PUBLIC $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include> $<INSTALL_INTERFACE:include>)
target_link_libraries (KEMVTKPart2
    PUBLIC
        KommonGsl
        KEMFieldsElectric
        KEMFieldsMagnetic
)

if( KEMField_USE_VTK )
    target_link_libraries(KEMVTKPart2
        PUBLIC
            KEMVTKPlugin
    )
endif( KEMField_USE_VTK )

kasper_install_headers ( ${VTK-PART2_HEADERFILES} )
kasper_install_libraries (KEMVTKPart2)
