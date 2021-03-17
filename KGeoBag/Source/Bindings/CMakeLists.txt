# headers
set( BINDINGS_HEADER_FILES
  # math
  Math/Include/KGTransformationBuilder.hh
  
  # core
  Core/Include/KGSurfaceBuilder.hh
  Core/Include/KGSpaceBuilder.hh
  Core/Include/KGInterfaceBuilder.hh
  
  # complex shapes
  Shapes/Complex/Include/KGBeamBuilder.hh
  Shapes/Complex/Include/KGConicalWireArrayBuilder.hh
  Shapes/Complex/Include/KGConicSectPortHousingBuilder.hh
  Shapes/Complex/Include/KGExtrudedObjectBuilder.hh
  Shapes/Complex/Include/KGPortHousingBuilder.hh
  Shapes/Complex/Include/KGRodBuilder.hh
  Shapes/Complex/Include/KGRotatedObjectBuilder.hh
  Shapes/Complex/Include/KGComplexAnnulusBuilder.hh
  Shapes/Complex/Include/KGLinearWireGridBuilder.hh
  Shapes/Complex/Include/KGQuadraticWireGridBuilder.hh
  Shapes/Complex/Include/KGCircleWireBuilder.hh
  Shapes/Complex/Include/KGCircularWirePinsBuilder.hh

  # planar paths
  Shapes/PlanarShapes/Include/KGPlanarLineSegmentBuilder.hh
  Shapes/PlanarShapes/Include/KGPlanarArcSegmentBuilder.hh
  Shapes/PlanarShapes/Include/KGPlanarPolyLineBuilder.hh
  Shapes/PlanarShapes/Include/KGPlanarCircleBuilder.hh
  Shapes/PlanarShapes/Include/KGPlanarPolyLoopBuilder.hh
  
  # flattened closed path surfaces
  Shapes/FlattenedAreas/Include/KGFlattenedCircleSurfaceBuilder.hh
  Shapes/FlattenedAreas/Include/KGFlattenedPolyLoopSurfaceBuilder.hh
  
  # rotated path surfaces
  Shapes/RotatedAreas/Include/KGRotatedLineSegmentSurfaceBuilder.hh
  Shapes/RotatedAreas/Include/KGDiskSurfaceBuilder.hh
  Shapes/RotatedAreas/Include/KGAnnulusSurfaceBuilder.hh
  Shapes/RotatedAreas/Include/KGCylinderSurfaceBuilder.hh
  Shapes/RotatedAreas/Include/KGConeSurfaceBuilder.hh
  Shapes/RotatedAreas/Include/KGCutConeSurfaceBuilder.hh
  Shapes/RotatedAreas/Include/KGRotatedArcSegmentSurfaceBuilder.hh
  Shapes/RotatedAreas/Include/KGCutTorusSurfaceBuilder.hh
  Shapes/RotatedAreas/Include/KGRotatedPolyLineSurfaceBuilder.hh
  Shapes/RotatedAreas/Include/KGRotatedCircleSurfaceBuilder.hh
  Shapes/RotatedAreas/Include/KGTorusSurfaceBuilder.hh
  Shapes/RotatedAreas/Include/KGRotatedPolyLoopSurfaceBuilder.hh
  
  # shell path surfaces
  Shapes/ShellAreas/Include/KGShellLineSegmentSurfaceBuilder.hh
  Shapes/ShellAreas/Include/KGShellArcSegmentSurfaceBuilder.hh
  Shapes/ShellAreas/Include/KGShellPolyLineSurfaceBuilder.hh
  Shapes/ShellAreas/Include/KGShellPolyLoopSurfaceBuilder.hh
  Shapes/ShellAreas/Include/KGShellCircleSurfaceBuilder.hh
  
  # extruded path surfaces
  Shapes/ExtrudedAreas/Include/KGExtrudedLineSegmentSurfaceBuilder.hh
  Shapes/ExtrudedAreas/Include/KGExtrudedArcSegmentSurfaceBuilder.hh
  Shapes/ExtrudedAreas/Include/KGExtrudedPolyLineSurfaceBuilder.hh
  Shapes/ExtrudedAreas/Include/KGExtrudedCircleSurfaceBuilder.hh
  Shapes/ExtrudedAreas/Include/KGExtrudedPolyLoopSurfaceBuilder.hh
  
  # rotated open path spaces
  Shapes/RotatedVolumes/Include/KGRotatedLineSegmentSpaceBuilder.hh
  Shapes/RotatedVolumes/Include/KGCylinderSpaceBuilder.hh
  Shapes/RotatedVolumes/Include/KGConeSpaceBuilder.hh
  Shapes/RotatedVolumes/Include/KGCutConeSpaceBuilder.hh
  Shapes/RotatedVolumes/Include/KGRotatedArcSegmentSpaceBuilder.hh
  Shapes/RotatedVolumes/Include/KGRotatedPolyLineSpaceBuilder.hh
  
  # rotated closed path spaces  
  Shapes/RotatedVolumes/Include/KGRotatedCircleSpaceBuilder.hh
  Shapes/RotatedVolumes/Include/KGTorusSpaceBuilder.hh
  Shapes/RotatedVolumes/Include/KGRotatedPolyLoopSpaceBuilder.hh
  Shapes/RotatedVolumes/Include/KGCylinderTubeSpaceBuilder.hh
  Shapes/RotatedVolumes/Include/KGCutConeTubeSpaceBuilder.hh
  
  # extruded closed path spaces
  Shapes/ExtrudedVolumes/Include/KGExtrudedCircleSpaceBuilder.hh
  Shapes/ExtrudedVolumes/Include/KGExtrudedPolyLoopSpaceBuilder.hh
  Shapes/ExtrudedVolumes/Include/KGBoxSpaceBuilder.hh
  
  # appearance
  Extensions/Appearance/Include/KGAppearanceBuilder.hh

  # deformation
  Extensions/Deformation/Include/KGMeshDeformerBuilder.hh

  # mesh
  Extensions/Mesh/Include/KGMeshBuilder.hh
  
  # axial mesh
  Extensions/AxialMesh/Include/KGAxialMeshBuilder.hh

  # discrete rotational mesh
  Extensions/DiscreteRotationalMesh/Include/KGDiscreteRotationalMeshBuilder.hh

  Visualization/Basic/Include/KGGeometryPrinterBuilder.hh
)

if( KGeoBag_USE_VTK )
  	list(APPEND BINDINGS_HEADER_FILES
    	Visualization/Vtk/Include/KGVTKGeometryPainterBuilder.hh
    	Visualization/Vtk/Include/KGVTKMeshPainterBuilder.hh
    	Visualization/Vtk/Include/KGVTKAxialMeshPainterBuilder.hh
    	Visualization/Vtk/Include/KGVTKPointTesterBuilder.hh
    	Visualization/Vtk/Include/KGVTKNormalTesterBuilder.hh
    	Visualization/Vtk/Include/KGVTKDistanceTesterBuilder.hh
    	Visualization/Vtk/Include/KGVTKOutsideTesterBuilder.hh
    	Visualization/Vtk/Include/KGVTKRandomPointTesterBuilder.hh
	)
endif( KGeoBag_USE_VTK )

if( KGeoBag_USE_ROOT )
	list( APPEND BINDINGS_HEADER_FILES
		Visualization/Root/Include/KGROOTGeometryPainterBuilder.hh
	)
endif( KGeoBag_USE_ROOT )

# sources
set( BINDINGS_SOURCE_FILES
    # math
    Math/Source/KGTransformationBuilder.cc
  
    # core
    Core/Source/KGSurfaceBuilder.cc
    Core/Source/KGSpaceBuilder.cc
    Core/Source/KGInterfaceBuilder.cc
  
    # complex shapes
    Shapes/Complex/Source/KGBeamBuilder.cc
    Shapes/Complex/Source/KGConicalWireArrayBuilder.cc
    Shapes/Complex/Source/KGConicSectPortHousingBuilder.cc
    Shapes/Complex/Source/KGExtrudedObjectBuilder.cc
    Shapes/Complex/Source/KGPortHousingBuilder.cc
    Shapes/Complex/Source/KGRodBuilder.cc
    Shapes/Complex/Source/KGRotatedObjectBuilder.cc
    Shapes/Complex/Source/KGComplexAnnulusBuilder.cc
    Shapes/Complex/Source/KGLinearWireGridBuilder.cc
    Shapes/Complex/Source/KGQuadraticWireGridBuilder.cc
    Shapes/Complex/Source/KGCircleWireBuilder.cc
    Shapes/Complex/Source/KGCircularWirePinsBuilder.cc

    # planar paths
    Shapes/PlanarShapes/Source/KGPlanarLineSegmentBuilder.cc
    Shapes/PlanarShapes/Source/KGPlanarArcSegmentBuilder.cc
    Shapes/PlanarShapes/Source/KGPlanarPolyLineBuilder.cc
    Shapes/PlanarShapes/Source/KGPlanarCircleBuilder.cc
    Shapes/PlanarShapes/Source/KGPlanarPolyLoopBuilder.cc

    # flattened closed path surfaces  
    Shapes/FlattenedAreas/Source/KGFlattenedCircleSurfaceBuilder.cc
    Shapes/FlattenedAreas/Source/KGFlattenedPolyLoopSurfaceBuilder.cc
  
    # rotated path surfaces
    Shapes/RotatedAreas/Source/KGRotatedLineSegmentSurfaceBuilder.cc
    Shapes/RotatedAreas/Source/KGDiskSurfaceBuilder.cc
    Shapes/RotatedAreas/Source/KGAnnulusSurfaceBuilder.cc
    Shapes/RotatedAreas/Source/KGCylinderSurfaceBuilder.cc
    Shapes/RotatedAreas/Source/KGConeSurfaceBuilder.cc
    Shapes/RotatedAreas/Source/KGCutConeSurfaceBuilder.cc
    Shapes/RotatedAreas/Source/KGRotatedArcSegmentSurfaceBuilder.cc
    Shapes/RotatedAreas/Source/KGCutTorusSurfaceBuilder.cc
    Shapes/RotatedAreas/Source/KGRotatedPolyLineSurfaceBuilder.cc
    Shapes/RotatedAreas/Source/KGRotatedCircleSurfaceBuilder.cc
    Shapes/RotatedAreas/Source/KGTorusSurfaceBuilder.cc
    Shapes/RotatedAreas/Source/KGRotatedPolyLoopSurfaceBuilder.cc
  
    # shell path surfaces
    Shapes/ShellAreas/Source/KGShellLineSegmentSurfaceBuilder.cc
    Shapes/ShellAreas/Source/KGShellArcSegmentSurfaceBuilder.cc
    Shapes/ShellAreas/Source/KGShellPolyLineSurfaceBuilder.cc
    Shapes/ShellAreas/Source/KGShellPolyLoopSurfaceBuilder.cc
    Shapes/ShellAreas/Source/KGShellCircleSurfaceBuilder.cc
  
    # extruded path surfaces
    Shapes/ExtrudedAreas/Source/KGExtrudedLineSegmentSurfaceBuilder.cc
    Shapes/ExtrudedAreas/Source/KGExtrudedArcSegmentSurfaceBuilder.cc
    Shapes/ExtrudedAreas/Source/KGExtrudedPolyLineSurfaceBuilder.cc
    Shapes/ExtrudedAreas/Source/KGExtrudedCircleSurfaceBuilder.cc
    Shapes/ExtrudedAreas/Source/KGExtrudedPolyLoopSurfaceBuilder.cc
  
    # rotated open path spaces
    Shapes/RotatedVolumes/Source/KGRotatedLineSegmentSpaceBuilder.cc  
    Shapes/RotatedVolumes/Source/KGCylinderSpaceBuilder.cc
    Shapes/RotatedVolumes/Source/KGConeSpaceBuilder.cc
    Shapes/RotatedVolumes/Source/KGCutConeSpaceBuilder.cc
    Shapes/RotatedVolumes/Source/KGRotatedArcSegmentSpaceBuilder.cc
    Shapes/RotatedVolumes/Source/KGRotatedPolyLineSpaceBuilder.cc
  
    # rotated closed path spaces
    Shapes/RotatedVolumes/Source/KGRotatedCircleSpaceBuilder.cc
    Shapes/RotatedVolumes/Source/KGTorusSpaceBuilder.cc
    Shapes/RotatedVolumes/Source/KGRotatedPolyLoopSpaceBuilder.cc
    Shapes/RotatedVolumes/Source/KGCylinderTubeSpaceBuilder.cc
    Shapes/RotatedVolumes/Source/KGCutConeTubeSpaceBuilder.cc
  
    # extruded closed path spaces
    Shapes/ExtrudedVolumes/Source/KGExtrudedCircleSpaceBuilder.cc
    Shapes/ExtrudedVolumes/Source/KGExtrudedPolyLoopSpaceBuilder.cc
    Shapes/ExtrudedVolumes/Source/KGBoxSpaceBuilder.cc
  
    # appearance
    Extensions/Appearance/Source/KGAppearanceBuilder.cc

    # deformation
    Extensions/Deformation/Source/KGMeshDeformerBuilder.cc

    # mesh
    Extensions/Mesh/Source/KGMeshBuilder.cc
    
    # axial mesh
    Extensions/AxialMesh/Source/KGAxialMeshBuilder.cc

    # discrete rotational mesh
    Extensions/DiscreteRotationalMesh/Source/KGDiscreteRotationalMeshBuilder.cc

    Visualization/Basic/Source/KGGeometryPrinterBuilder.cc
)

if( KGeoBag_USE_VTK )
	list( APPEND BINDINGS_SOURCE_FILES
	    Visualization/Vtk/Source/KGVTKGeometryPainterBuilder.cc
	    Visualization/Vtk/Source/KGVTKMeshPainterBuilder.cc
	    Visualization/Vtk/Source/KGVTKAxialMeshPainterBuilder.cc
	    Visualization/Vtk/Source/KGVTKPointTesterBuilder.cc
	    Visualization/Vtk/Source/KGVTKNormalTesterBuilder.cc
	    Visualization/Vtk/Source/KGVTKDistanceTesterBuilder.cc
	    Visualization/Vtk/Source/KGVTKOutsideTesterBuilder.cc
	    Visualization/Vtk/Source/KGVTKRandomPointTesterBuilder.cc
	)
endif( KGeoBag_USE_VTK )

if( KGeoBag_USE_ROOT )
	list( APPEND BINDINGS_SOURCE_FILES
		Visualization/Root/Source/KGROOTGeometryPainterBuilder.cc
	)
endif( KGeoBag_USE_ROOT )

# internal
kasper_internal_include_directories(
    ${CMAKE_CURRENT_SOURCE_DIR}/Math/Include
    ${CMAKE_CURRENT_SOURCE_DIR}/Core/Include
    ${CMAKE_CURRENT_SOURCE_DIR}/Shapes/Complex/Include
    ${CMAKE_CURRENT_SOURCE_DIR}/Shapes/PlanarShapes/Include
    ${CMAKE_CURRENT_SOURCE_DIR}/Shapes/FlattenedAreas/Include
    ${CMAKE_CURRENT_SOURCE_DIR}/Shapes/RotatedAreas/Include
    ${CMAKE_CURRENT_SOURCE_DIR}/Shapes/RotatedVolumes/Include
    ${CMAKE_CURRENT_SOURCE_DIR}/Shapes/ExtrudedAreas/Include
    ${CMAKE_CURRENT_SOURCE_DIR}/Shapes/ExtrudedVolumes/Include
    ${CMAKE_CURRENT_SOURCE_DIR}/Shapes/ShellAreas/Include
    ${CMAKE_CURRENT_SOURCE_DIR}/Extensions/Appearance/Include
    ${CMAKE_CURRENT_SOURCE_DIR}/Extensions/Deformation/Include
    ${CMAKE_CURRENT_SOURCE_DIR}/Extensions/Mesh/Include
    ${CMAKE_CURRENT_SOURCE_DIR}/Extensions/AxialMesh/Include
    ${CMAKE_CURRENT_SOURCE_DIR}/Extensions/DiscreteRotationalMesh/Include
    ${CMAKE_CURRENT_SOURCE_DIR}/Visualization/Basic/Include
)

if( KGeoBag_USE_VTK )
	kasper_internal_include_directories(
		${CMAKE_CURRENT_SOURCE_DIR}/Visualization/Vtk/Include
	)
endif( KGeoBag_USE_VTK )

if( KGeoBag_USE_ROOT )
	kasper_internal_include_directories(
		${CMAKE_CURRENT_SOURCE_DIR}/Visualization/Root/Include
	)
endif( KGeoBag_USE_ROOT )

# target
add_library( KGeoBagBindings SHARED
	${BINDINGS_SOURCE_FILES} ${BINDINGS_HEADER_FILES}
)
set( BINDINGS_LIBRARIES
	${Kommon_LIBRARIES}
    KGeoBagCore
    KGeoBagShapes
    KGeoBagAppearance
    KGeoBagDeformation
    KGeoBagMesh
    KGeoBagAxialMesh
    KGeoBagDiscreteRotationalMesh
    KGeoBagBasicVisualization
)

if( KGeoBag_USE_VTK )
	list( APPEND BINDINGS_LIBRARIES
		KGeoBagVtkVisualization
	)
endif( KGeoBag_USE_VTK )

if( KGeoBag_USE_ROOT )
	list( APPEND BINDINGS_LIBRARIES
		KGeoBagROOTVisualization
	)
endif( KGeoBag_USE_ROOT )

target_link_libraries( KGeoBagBindings
	${BINDINGS_LIBRARIES}
)

# install
kasper_install_headers( ${BINDINGS_HEADER_FILES} )
kasper_install_libraries( KGeoBagBindings )
