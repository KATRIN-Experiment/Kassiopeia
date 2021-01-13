# headers
set( SHAPES_HEADER_FILES
	# utilities
	Utility/Include/KGShapeMessage.hh
	  
	# basic shapes
	Basic/Include/KGBox.hh
	Basic/Include/KGRectangle.hh
	Basic/Include/KGDisk.hh
	Basic/Include/KGCylinder.hh
	
	# planar paths
	PlanarShapes/Include/KGPlanarPath.hh
	PlanarShapes/Include/KGPlanarOpenPath.hh
	PlanarShapes/Include/KGPlanarLineSegment.hh
	PlanarShapes/Include/KGPlanarArcSegment.hh
	PlanarShapes/Include/KGPlanarPolyLine.hh
	PlanarShapes/Include/KGPlanarClosedPath.hh
	PlanarShapes/Include/KGPlanarCircle.hh
	PlanarShapes/Include/KGPlanarPolyLoop.hh
	  
	# flattened closed path surfaces
	FlattenedAreas/Include/KGFlattenedClosedPathSurface.hh
	FlattenedAreas/Include/KGFlattenedCircleSurface.hh
	FlattenedAreas/Include/KGFlattenedPolyLoopSurface.hh
	  
	# rotated path surfaces
	RotatedAreas/Include/KGRotatedPathSurface.hh
	RotatedAreas/Include/KGRotatedLineSegmentSurface.hh
	RotatedAreas/Include/KGDiskSurface.hh
	RotatedAreas/Include/KGAnnulusSurface.hh
	RotatedAreas/Include/KGCylinderSurface.hh
	RotatedAreas/Include/KGConeSurface.hh
	RotatedAreas/Include/KGCutConeSurface.hh
	RotatedAreas/Include/KGRotatedArcSegmentSurface.hh
	RotatedAreas/Include/KGCutTorusSurface.hh
	RotatedAreas/Include/KGRotatedPolyLineSurface.hh
	RotatedAreas/Include/KGRotatedCircleSurface.hh
	RotatedAreas/Include/KGTorusSurface.hh
	RotatedAreas/Include/KGRotatedPolyLoopSurface.hh
	
    # shell path surfaces
    ShellAreas/Include/KGShellPathSurface.hh
    ShellAreas/Include/KGShellLineSegmentSurface.hh
    ShellAreas/Include/KGShellArcSegmentSurface.hh
    ShellAreas/Include/KGShellPolyLineSurface.hh
    ShellAreas/Include/KGShellPolyLoopSurface.hh
    ShellAreas/Include/KGShellCircleSurface.hh
	  
	# extruded path surfaces
	ExtrudedAreas/Include/KGExtrudedPathSurface.hh
	ExtrudedAreas/Include/KGExtrudedLineSegmentSurface.hh
	ExtrudedAreas/Include/KGExtrudedArcSegmentSurface.hh
	ExtrudedAreas/Include/KGExtrudedPolyLineSurface.hh
	ExtrudedAreas/Include/KGExtrudedCircleSurface.hh
	ExtrudedAreas/Include/KGExtrudedPolyLoopSurface.hh
	  
	# rotated open path spaces
	RotatedVolumes/Include/KGRotatedOpenPathSpace.hh
	RotatedVolumes/Include/KGRotatedLineSegmentSpace.hh
	RotatedVolumes/Include/KGCylinderSpace.hh
	RotatedVolumes/Include/KGConeSpace.hh
	RotatedVolumes/Include/KGCutConeSpace.hh
	RotatedVolumes/Include/KGRotatedArcSegmentSpace.hh
	RotatedVolumes/Include/KGRotatedPolyLineSpace.hh
	  
	# rotated closed path spaces  
	RotatedVolumes/Include/KGRotatedClosedPathSpace.hh
	RotatedVolumes/Include/KGRotatedCircleSpace.hh
	RotatedVolumes/Include/KGTorusSpace.hh
	RotatedVolumes/Include/KGRotatedPolyLoopSpace.hh
	RotatedVolumes/Include/KGCylinderTubeSpace.hh
	RotatedVolumes/Include/KGCutConeTubeSpace.hh
	  
	# extruded closed path spaces
	ExtrudedVolumes/Include/KGExtrudedClosedPathSpace.hh
	ExtrudedVolumes/Include/KGExtrudedCircleSpace.hh
	ExtrudedVolumes/Include/KGExtrudedPolyLoopSpace.hh
	ExtrudedVolumes/Include/KGBoxSpace.hh
	  
	# complex shapes
	Complex/Include/KGWrappedSurface.hh
	Complex/Include/KGWrappedSpace.hh
	Complex/Include/KGExtrudedObject.hh
	Complex/Include/KGExtrudedSurface.hh
	Complex/Include/KGExtrudedSpace.hh
	Complex/Include/KGRotatedObject.hh
	Complex/Include/KGRotatedSurface.hh
	Complex/Include/KGRotatedSpace.hh
	Complex/Include/KGConicalWireArray.hh
	Complex/Include/KGConicalWireArraySurface.hh
	Complex/Include/KGConicalWireArraySpace.hh
	Complex/Include/KGConicSectPortHousing.hh
	Complex/Include/KGConicSectPortHousingSurface.hh
	Complex/Include/KGConicSectPortHousingSpace.hh
	Complex/Include/KGPortHousing.hh
	Complex/Include/KGPortHousingSurface.hh
	Complex/Include/KGPortHousingSpace.hh
	Complex/Include/KGBeam.hh
	Complex/Include/KGBeamSurface.hh
	Complex/Include/KGBeamSpace.hh
	Complex/Include/KGRod.hh
	Complex/Include/KGRodSurface.hh
	Complex/Include/KGRodSpace.hh
	Complex/Include/KGComplexAnnulus.hh
	Complex/Include/KGComplexAnnulusSurface.hh
	Complex/Include/KGLinearWireGrid.hh
	Complex/Include/KGLinearWireGridSurface.hh
	Complex/Include/KGLinearWireGridSpace.hh
	Complex/Include/KGQuadraticWireGrid.hh
    Complex/Include/KGQuadraticWireGridSurface.hh
    Complex/Include/KGQuadraticWireGridSpace.hh
    Complex/Include/KGCircleWire.hh
    Complex/Include/KGCircleWireSurface.hh
    Complex/Include/KGCircleWireSpace.hh
    Complex/Include/KGCircularWirePins.hh
    Complex/Include/KGCircularWirePinsSurface.hh
    Complex/Include/KGCircularWirePinsSpace.hh
)

# source
set( SHAPES_SOURCE_FILES
    # utilities
    Utility/Source/KGShapeMessage.cc

    # basic shapes
    Basic/Source/KGBox.cc
    Basic/Source/KGRectangle.cc
    Basic/Source/KGDisk.cc
    Basic/Source/KGCylinder.cc

    # planar paths
    PlanarShapes/Source/KGPlanarPath.cc
    PlanarShapes/Source/KGPlanarOpenPath.cc
    PlanarShapes/Source/KGPlanarLineSegment.cc
    PlanarShapes/Source/KGPlanarArcSegment.cc
    PlanarShapes/Source/KGPlanarPolyLine.cc
    PlanarShapes/Source/KGPlanarClosedPath.cc
    PlanarShapes/Source/KGPlanarCircle.cc
    PlanarShapes/Source/KGPlanarPolyLoop.cc

    # flattened closed path surfaces  
    FlattenedAreas/Source/KGFlattenedCircleSurface.cc
    FlattenedAreas/Source/KGFlattenedPolyLoopSurface.cc
  
    # rotated path surfaces
    RotatedAreas/Source/KGRotatedLineSegmentSurface.cc
    RotatedAreas/Source/KGDiskSurface.cc
    RotatedAreas/Source/KGAnnulusSurface.cc
    RotatedAreas/Source/KGCylinderSurface.cc
    RotatedAreas/Source/KGConeSurface.cc
    RotatedAreas/Source/KGCutConeSurface.cc
    RotatedAreas/Source/KGRotatedArcSegmentSurface.cc
    RotatedAreas/Source/KGCutTorusSurface.cc
    RotatedAreas/Source/KGRotatedPolyLineSurface.cc
    RotatedAreas/Source/KGRotatedCircleSurface.cc
    RotatedAreas/Source/KGTorusSurface.cc
    RotatedAreas/Source/KGRotatedPolyLoopSurface.cc

    # shell path surfaces
    ShellAreas/Source/KGShellLineSegmentSurface.cc
    ShellAreas/Source/KGShellArcSegmentSurface.cc
    ShellAreas/Source/KGShellPolyLineSurface.cc
    ShellAreas/Source/KGShellPolyLoopSurface.cc
    ShellAreas/Source/KGShellCircleSurface.cc
  
    # extruded path surfaces
    ExtrudedAreas/Source/KGExtrudedLineSegmentSurface.cc
    ExtrudedAreas/Source/KGExtrudedArcSegmentSurface.cc
    ExtrudedAreas/Source/KGExtrudedPolyLineSurface.cc
    ExtrudedAreas/Source/KGExtrudedCircleSurface.cc
    ExtrudedAreas/Source/KGExtrudedPolyLoopSurface.cc
  
    # rotated open path spaces
    RotatedVolumes/Source/KGRotatedLineSegmentSpace.cc  
    RotatedVolumes/Source/KGCylinderSpace.cc
    RotatedVolumes/Source/KGConeSpace.cc
    RotatedVolumes/Source/KGCutConeSpace.cc
    RotatedVolumes/Source/KGRotatedArcSegmentSpace.cc
    RotatedVolumes/Source/KGRotatedPolyLineSpace.cc
  
    # rotated closed path spaces
    RotatedVolumes/Source/KGRotatedCircleSpace.cc
    RotatedVolumes/Source/KGTorusSpace.cc
    RotatedVolumes/Source/KGRotatedPolyLoopSpace.cc
    RotatedVolumes/Source/KGCylinderTubeSpace.cc
    RotatedVolumes/Source/KGCutConeTubeSpace.cc
  
    # extruded closed path spaces
    ExtrudedVolumes/Source/KGExtrudedCircleSpace.cc
    ExtrudedVolumes/Source/KGExtrudedPolyLoopSpace.cc
    ExtrudedVolumes/Source/KGBoxSpace.cc
  
    
    Complex/Source/KGExtrudedObject.cc
    Complex/Source/KGRotatedObject.cc
    Complex/Source/KGPortHousing.cc
    Complex/Source/KGConicalWireArray.cc
    Complex/Source/KGConicSectPortHousing.cc
    Complex/Source/KGBeam.cc
    Complex/Source/KGRod.cc
    Complex/Source/KGLinearWireGrid.cc
    Complex/Source/KGQuadraticWireGrid.cc
    Complex/Source/KGCircleWire.cc
    Complex/Source/KGCircularWirePins.cc

    Complex/Source/KGRotatedSpace.cc
    Complex/Source/KGPortHousingSpace.cc
    Complex/Source/KGConicSectPortHousingSpace.cc
    Complex/Source/KGComplexAnnulus.cc
)

# internal
kasper_internal_include_directories(
 	${CMAKE_CURRENT_SOURCE_DIR}/Basic/Include
    ${CMAKE_CURRENT_SOURCE_DIR}/Complex/Include
    ${CMAKE_CURRENT_SOURCE_DIR}/Utility/Include
    ${CMAKE_CURRENT_SOURCE_DIR}/PlanarShapes/Include
    ${CMAKE_CURRENT_SOURCE_DIR}/FlattenedAreas/Include
    ${CMAKE_CURRENT_SOURCE_DIR}/RotatedAreas/Include
    ${CMAKE_CURRENT_SOURCE_DIR}/RotatedVolumes/Include
    ${CMAKE_CURRENT_SOURCE_DIR}/ExtrudedAreas/Include
    ${CMAKE_CURRENT_SOURCE_DIR}/ExtrudedVolumes/Include
    ${CMAKE_CURRENT_SOURCE_DIR}/ShellAreas/Include
)

# target
add_library( KGeoBagShapes SHARED
    ${SHAPES_SOURCE_FILES} ${SHAPES_HEADER_FILES}
)
target_link_libraries( KGeoBagShapes
    ${Kommon_LIBRARIES}
    KGeoBagMath
    KGeoBagCore
)

# install
kasper_install_headers(
	${SHAPES_HEADER_FILES}
)
kasper_install_libraries(
	KGeoBagShapes
)
