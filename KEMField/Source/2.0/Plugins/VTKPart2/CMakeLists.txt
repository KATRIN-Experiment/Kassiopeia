# CMakeLists for KEMField/Plugins/VTKPart2
# Author: W. Gosda
# This library is the second part of the vtk plugin. This split is necessary to avoid
#circular dependencies.

# The library is build even with VTK disabled to provide a function less empty
# version of the KVTKViewerAsBoundaryFieldVisitor to avoid parser errors.

kasper_internal_include_directories(${CMAKE_CURRENT_SOURCE_DIR}/include)

#header files

set( VTK-PART2_HEADERFILES
  ${CMAKE_CURRENT_SOURCE_DIR}/include/KVTKViewerAsBoundaryFieldVisitor.hh
)

if( @PROJECT_NAME@_USE_VTK AND @PROJECT_NAME@_USE_KGEOBAG)
    list( APPEND VTK-PART2_HEADERFILES 
        ${CMAKE_CURRENT_SOURCE_DIR}/include/KElectrostaticPotentialMap.hh
    )
endif( @PROJECT_NAME@_USE_VTK AND @PROJECT_NAME@_USE_KGEOBAG)

#source files

set( VTK-PART2_SOURCEFILES
  ${CMAKE_CURRENT_SOURCE_DIR}/src/KVTKViewerAsBoundaryFieldVisitor.cc
)

if( @PROJECT_NAME@_USE_VTK AND @PROJECT_NAME@_USE_KGEOBAG)
    list( APPEND VTK-PART2_SOURCEFILES 
        ${CMAKE_CURRENT_SOURCE_DIR}/src/KElectrostaticPotentialMap.cc
    )
endif( @PROJECT_NAME@_USE_VTK AND @PROJECT_NAME@_USE_KGEOBAG)

#link libraries

set( VTK-PART2_LINK_LIBRARIES
    KEMFieldsElectric
)
if( @PROJECT_NAME@_USE_VTK )
      list( APPEND VTK-PART2_LINK_LIBRARIES
        KEMVTKPlugin
          
      )
endif( @PROJECT_NAME@_USE_VTK )

if( @PROJECT_NAME@_USE_VTK AND @PROJECT_NAME@_USE_KGEOBAG )
      list( APPEND VTK-PART2_LINK_LIBRARIES
          ${KGEOBAG_LIBRARIES}
          ${VTK_LIBRARIES}
          
      )
endif( @PROJECT_NAME@_USE_VTK AND @PROJECT_NAME@_USE_KGEOBAG )

add_library (KEMVTKPart2 SHARED 
  ${VTK-PART2_SOURCEFILES} 
)

target_link_libraries ( KEMVTKPart2 ${VTK-PART2_LINK_LIBRARIES})
  
kasper_install_headers ( ${VTK-PART2_HEADERFILES} )
kasper_install_libraries (KEMVTKPart2)
