
# KRoot

|element name|source files|child elements|child types|attributes|attribute types|
|-----|-----|-----|-----|-----|-----|
|—    |—    |[`NamedRandomGenerator`](#kommonnamedrandomgenerator)<br>[`Random`](#kdummyrandom)<br>[`geometry`](#kginterface)<br>[`geometry_printer`](#kggeometryprinter)<br>[`kassiopeia`](#ksroot)<br>[`kemfield`](#kemroot)<br>[`messages`](#kmessagetable)<br>[`root_window`](#krootwindow)<br>[`run`](#kapplicationrunner)<br>[`vtk_window`](#kvtkwindow)|*`KommonNamedRandomGenerator`*<br>*`KDummyRandom`*<br>*`KGInterface`*<br>*`KGGeometryPrinter`*<br>*`KSRoot`*<br>*`KEMRoot`*<br>*`KMessageTable`*<br>*`KROOTWindow`*<br>*`KApplicationRunner`*<br>*`KVTKWindow`*|—    |—    |

## KommonNamedRandomGenerator

|element name|source files|child elements|child types|attributes|attribute types|
|-----|-----|-----|-----|-----|-----|
|<a name="kommonnamedrandomgenerator">`NamedRandomGenerator`</a>|[*KNamedRandomGeneratorBuilder.cxx*](https://github.com/KATRIN-Experiment/Kassiopeia/tree/main/Kommon/Core/Bindings/KNamedRandomGeneratorBuilder.cxx)|—    |—    |`Name`<br>`Seed`|*`string`*<br>*`SeedType`*|

## KDummyRandom

|element name|source files|child elements|child types|attributes|attribute types|
|-----|-----|-----|-----|-----|-----|
|<a name="kdummyrandom">`Random`</a>|[*KRandomBuilder.cxx*](https://github.com/KATRIN-Experiment/Kassiopeia/tree/main/Kommon/Core/Bindings/KRandomBuilder.cxx)|—    |—    |`Seed`|*`int32_t`*|

## KGInterface

|element name|source files|child elements|child types|attributes|attribute types|
|-----|-----|-----|-----|-----|-----|
|<a name="kginterface">`geometry`</a>|[*KGInterfaceBuilder.cc*](https://github.com/KATRIN-Experiment/Kassiopeia/tree/main/KGeoBag/Source/Bindings/Core/Source/KGInterfaceBuilder.cc)|[`annulus_surface`](#kgannulussurface)<br>[`appearance`](#kgappearanceattributor)<br>[`axial_mesh`](#kgaxialmeshattributor)<br>[`beam_space`](#kgwrappedspace\<kgbeam\>)<br>[`beam_surface`](#kgwrappedsurface\<kgbeam\>)<br>[`box_space`](#kgboxspace)<br>[`circle_wire_space`](#kgwrappedspace\<kgcirclewire\>)<br>[`circle_wire_surface`](#kgwrappedsurface\<kgcirclewire\>)<br>[`circular_wire_pins_space`](#kgwrappedspace\<kgcircularwirepins\>)<br>[`circular_wire_pins_surface`](#kgwrappedsurface\<kgcircularwirepins\>)<br>[`complex_annulus_surface`](#kgwrappedsurface\<kgcomplexannulus\>)<br>[`cone_space`](#kgconespace)<br>[`cone_surface`](#kgconesurface)<br>[`conic_section_port_housing_space`](#kgwrappedspace\<kgconicsectporthousing\>)<br>[`conic_section_port_housing_surface`](#kgwrappedsurface\<kgconicsectporthousing\>)<br>[`conical_wire_array_space`](#kgwrappedspace\<kgconicalwirearray\>)<br>[`conical_wire_array_surface`](#kgwrappedsurface\<kgconicalwirearray\>)<br>[`cut_cone_space`](#kgcutconespace)<br>[`cut_cone_surface`](#kgcutconesurface)<br>[`cut_cone_tube_space`](#kgcutconetubespace)<br>[`cut_torus_surface`](#kgcuttorussurface)<br>[`cylinder_space`](#kgcylinderspace)<br>[`cylinder_surface`](#kgcylindersurface)<br>[`cylinder_tube_space`](#kgcylindertubespace)<br>[`discrete_rotational_mesh`](#kgdiscreterotationalmeshattributor)<br>[`disk_surface`](#kgdisksurface)<br>[`electromagnet`](#kgelectromagnetattributor)<br>[`extruded_arc_segment_surface`](#kgextrudedarcsegmentsurface)<br>[`extruded_circle_space`](#kgextrudedcirclespace)<br>[`extruded_circle_surface`](#kgextrudedcirclesurface)<br>[`extruded_line_segment_surface`](#kgextrudedlinesegmentsurface)<br>[`extruded_poly_line_surface`](#kgextrudedpolylinesurface)<br>[`extruded_poly_loop_space`](#kgextrudedpolyloopspace)<br>[`extruded_poly_loop_surface`](#kgextrudedpolyloopsurface)<br>[`extruded_space`](#kgwrappedspace\<kgextrudedobject\>)<br>[`extruded_surface`](#kgwrappedsurface\<kgextrudedobject\>)<br>[`flattened_circle_surface`](#kgflattenedcirclesurface)<br>[`flattened_poly_loop_surface`](#kgflattenedpolyloopsurface)<br>[`linear_wire_grid_space`](#kgwrappedspace\<kglinearwiregrid\>)<br>[`linear_wire_grid_surface`](#kgwrappedsurface\<kglinearwiregrid\>)<br>[`mesh`](#kgmeshattributor)<br>[`mesh_deformer`](#kgmeshdeformer)<br>[`mesh_refiner`](#kgmeshrefiner)<br>[`ply_file_space`](#kgwrappedspace\<kgplyfile\>)<br>[`ply_file_surface`](#kgwrappedsurface\<kgplyfile\>)<br>[`port_housing_space`](#kgwrappedspace\<kgporthousing\>)<br>[`port_housing_surface`](#kgwrappedsurface\<kgporthousing\>)<br>[`quadratic_wire_grid_space`](#kgwrappedspace\<kgquadraticwiregrid\>)<br>[`quadratic_wire_grid_surface`](#kgwrappedsurface\<kgquadraticwiregrid\>)<br>[`rod_space`](#kgwrappedspace\<kgrod\>)<br>[`rod_surface`](#kgwrappedsurface\<kgrod\>)<br>[`rotated_arc_segment_space`](#kgrotatedarcsegmentspace)<br>[`rotated_arc_segment_surface`](#kgrotatedarcsegmentsurface)<br>[`rotated_circle_space`](#kgrotatedcirclespace)<br>[`rotated_circle_surface`](#kgrotatedcirclesurface)<br>[`rotated_line_segment_space`](#kgrotatedlinesegmentspace)<br>[`rotated_line_segment_surface`](#kgrotatedlinesegmentsurface)<br>[`rotated_poly_line_space`](#kgrotatedpolylinespace)<br>[`rotated_poly_line_surface`](#kgrotatedpolylinesurface)<br>[`rotated_poly_loop_space`](#kgrotatedpolyloopspace)<br>[`rotated_poly_loop_surface`](#kgrotatedpolyloopsurface)<br>[`rotated_space`](#kgwrappedspace\<kgrotatedobject\>)<br>[`rotated_surface`](#kgwrappedsurface\<kgrotatedobject\>)<br>[`shell_arc_segment_surface`](#kgshellarcsegmentsurface)<br>[`shell_circle_surface`](#kgshellcirclesurface)<br>[`shell_line_segment_surface`](#kgshelllinesegmentsurface)<br>[`shell_poly_line_surface`](#kgshellpolylinesurface)<br>[`shell_poly_loop_surface`](#kgshellpolyloopsurface)<br>[`space`](#kgspace)<br>[`stl_file_space`](#kgwrappedspace\<kgstlfile\>)<br>[`stl_file_surface`](#kgwrappedsurface\<kgstlfile\>)<br>[`surface`](#kgsurface)<br>[`torus_space`](#kgtorusspace)<br>[`torus_surface`](#kgtorussurface)|*`KGAnnulusSurface`*<br>*`KGAppearanceAttributor`*<br>*`KGAxialMeshAttributor`*<br>*`KGWrappedSpace<KGBeam>`*<br>*`KGWrappedSurface<KGBeam>`*<br>*`KGBoxSpace`*<br>*`KGWrappedSpace<KGCircleWire>`*<br>*`KGWrappedSurface<KGCircleWire>`*<br>*`KGWrappedSpace<KGCircularWirePins>`*<br>*`KGWrappedSurface<KGCircularWirePins>`*<br>*`KGWrappedSurface<KGComplexAnnulus>`*<br>*`KGConeSpace`*<br>*`KGConeSurface`*<br>*`KGWrappedSpace<KGConicSectPortHousing>`*<br>*`KGWrappedSurface<KGConicSectPortHousing>`*<br>*`KGWrappedSpace<KGConicalWireArray>`*<br>*`KGWrappedSurface<KGConicalWireArray>`*<br>*`KGCutConeSpace`*<br>*`KGCutConeSurface`*<br>*`KGCutConeTubeSpace`*<br>*`KGCutTorusSurface`*<br>*`KGCylinderSpace`*<br>*`KGCylinderSurface`*<br>*`KGCylinderTubeSpace`*<br>*`KGDiscreteRotationalMeshAttributor`*<br>*`KGDiskSurface`*<br>*`KGElectromagnetAttributor`*<br>*`KGExtrudedArcSegmentSurface`*<br>*`KGExtrudedCircleSpace`*<br>*`KGExtrudedCircleSurface`*<br>*`KGExtrudedLineSegmentSurface`*<br>*`KGExtrudedPolyLineSurface`*<br>*`KGExtrudedPolyLoopSpace`*<br>*`KGExtrudedPolyLoopSurface`*<br>*`KGWrappedSpace<KGExtrudedObject>`*<br>*`KGWrappedSurface<KGExtrudedObject>`*<br>*`KGFlattenedCircleSurface`*<br>*`KGFlattenedPolyLoopSurface`*<br>*`KGWrappedSpace<KGLinearWireGrid>`*<br>*`KGWrappedSurface<KGLinearWireGrid>`*<br>*`KGMeshAttributor`*<br>*`KGMeshDeformer`*<br>*`KGMeshRefiner`*<br>*`KGWrappedSpace<KGPlyFile>`*<br>*`KGWrappedSurface<KGPlyFile>`*<br>*`KGWrappedSpace<KGPortHousing>`*<br>*`KGWrappedSurface<KGPortHousing>`*<br>*`KGWrappedSpace<KGQuadraticWireGrid>`*<br>*`KGWrappedSurface<KGQuadraticWireGrid>`*<br>*`KGWrappedSpace<KGRod>`*<br>*`KGWrappedSurface<KGRod>`*<br>*`KGRotatedArcSegmentSpace`*<br>*`KGRotatedArcSegmentSurface`*<br>*`KGRotatedCircleSpace`*<br>*`KGRotatedCircleSurface`*<br>*`KGRotatedLineSegmentSpace`*<br>*`KGRotatedLineSegmentSurface`*<br>*`KGRotatedPolyLineSpace`*<br>*`KGRotatedPolyLineSurface`*<br>*`KGRotatedPolyLoopSpace`*<br>*`KGRotatedPolyLoopSurface`*<br>*`KGWrappedSpace<KGRotatedObject>`*<br>*`KGWrappedSurface<KGRotatedObject>`*<br>*`KGShellArcSegmentSurface`*<br>*`KGShellCircleSurface`*<br>*`KGShellLineSegmentSurface`*<br>*`KGShellPolyLineSurface`*<br>*`KGShellPolyLoopSurface`*<br>*`KGSpace`*<br>*`KGWrappedSpace<KGStlFile>`*<br>*`KGWrappedSurface<KGStlFile>`*<br>*`KGSurface`*<br>*`KGTorusSpace`*<br>*`KGTorusSurface`*|`reset`|*`bool`*|

### KGAnnulusSurface

|element name|source files|child elements|child types|attributes|attribute types|
|-----|-----|-----|-----|-----|-----|
|<a name="kgannulussurface">`annulus_surface`</a>|[*KGAnnulusSurfaceBuilder.cc*](https://github.com/KATRIN-Experiment/Kassiopeia/tree/main/KGeoBag/Source/Bindings/Shapes/RotatedAreas/Source/KGAnnulusSurfaceBuilder.cc)|—    |—    |`axial_mesh_count`<br>`name`<br>`r1`<br>`r2`<br>`radial_mesh_count`<br>`radial_mesh_power`<br>`z`|*`unsigned int`*<br>*`string`*<br>*`double`*<br>*`double`*<br>*`unsigned int`*<br>*`double`*<br>*`double`*|

### KGAppearanceAttributor

|element name|source files|child elements|child types|attributes|attribute types|
|-----|-----|-----|-----|-----|-----|
|<a name="kgappearanceattributor">`appearance`</a>|[*KGAppearanceBuilder.cc*](https://github.com/KATRIN-Experiment/Kassiopeia/tree/main/KGeoBag/Source/Bindings/Extensions/Appearance/Source/KGAppearanceBuilder.cc)|—    |—    |`arc`<br>`color`<br>`name`<br>`spaces`<br>`surfaces`|*`unsigned int`*<br>*`KGRGBAColor`*<br>*`string`*<br>*`string`*<br>*`string`*|

### KGAxialMeshAttributor

|element name|source files|child elements|child types|attributes|attribute types|
|-----|-----|-----|-----|-----|-----|
|<a name="kgaxialmeshattributor">`axial_mesh`</a>|[*KGAxialMeshBuilder.cc*](https://github.com/KATRIN-Experiment/Kassiopeia/tree/main/KGeoBag/Source/Bindings/Extensions/AxialMesh/Source/KGAxialMeshBuilder.cc)|—    |—    |`name`<br>`spaces`<br>`surfaces`|*`string`*<br>*`string`*<br>*`string`*|

### KGWrappedSpace\<KGBeam\>

|element name|source files|child elements|child types|attributes|attribute types|
|-----|-----|-----|-----|-----|-----|
|<a name="kgwrappedspace\<kgbeam\>">`beam_space`</a>|[*KGBeamBuilder.cc*](https://github.com/KATRIN-Experiment/Kassiopeia/tree/main/KGeoBag/Source/Bindings/Shapes/Complex/Source/KGBeamBuilder.cc)|[`beam`](#kgbeam)|*`KGBeam`*|`name`|*`string`*|

#### KGBeam

|element name|source files|child elements|child types|attributes|attribute types|
|-----|-----|-----|-----|-----|-----|
|<a name="kgbeam">`beam`</a>|[*KGBeamBuilder.cc*](https://github.com/KATRIN-Experiment/Kassiopeia/tree/main/KGeoBag/Source/Bindings/Shapes/Complex/Source/KGBeamBuilder.cc)|[`end_line`](#kgbeamline)<br>[`start_line`](#kgbeamline)|*`KGBeamLine`*<br>*`KGBeamLine`*|`axial_mesh_count`<br>`longitudinal_mesh_count`|*`int`*<br>*`int`*|

##### KGBeamLine

|element name|source files|child elements|child types|attributes|attribute types|
|-----|-----|-----|-----|-----|-----|
|<a name="kgbeamline">`end_line`</a>|[*KGBeamBuilder.cc*](https://github.com/KATRIN-Experiment/Kassiopeia/tree/main/KGeoBag/Source/Bindings/Shapes/Complex/Source/KGBeamBuilder.cc)|—    |—    |`x1`<br>`x2`<br>`y1`<br>`y2`<br>`z1`<br>`z2`|*`double`*<br>*`double`*<br>*`double`*<br>*`double`*<br>*`double`*<br>*`double`*|

### KGWrappedSurface\<KGBeam\>

|element name|source files|child elements|child types|attributes|attribute types|
|-----|-----|-----|-----|-----|-----|
|<a name="kgwrappedsurface\<kgbeam\>">`beam_surface`</a>|[*KGBeamBuilder.cc*](https://github.com/KATRIN-Experiment/Kassiopeia/tree/main/KGeoBag/Source/Bindings/Shapes/Complex/Source/KGBeamBuilder.cc)|[`beam`](#kgbeam)|*`KGBeam`*|`name`|*`string`*|

### KGBoxSpace

|element name|source files|child elements|child types|attributes|attribute types|
|-----|-----|-----|-----|-----|-----|
|<a name="kgboxspace">`box_space`</a>|[*KGBoxSpaceBuilder.cc*](https://github.com/KATRIN-Experiment/Kassiopeia/tree/main/KGeoBag/Source/Bindings/Shapes/ExtrudedVolumes/Source/KGBoxSpaceBuilder.cc)|—    |—    |`name`<br>`x_mesh_count`<br>`x_mesh_power`<br>`xa`<br>`xb`<br>`y_mesh_count`<br>`y_mesh_power`<br>`ya`<br>`yb`<br>`z_mesh_count`<br>`z_mesh_power`<br>`za`<br>`zb`|*`string`*<br>*`unsigned int`*<br>*`double`*<br>*`double`*<br>*`double`*<br>*`unsigned int`*<br>*`double`*<br>*`double`*<br>*`double`*<br>*`unsigned int`*<br>*`double`*<br>*`double`*<br>*`double`*|

### KGWrappedSpace\<KGCircleWire\>

|element name|source files|child elements|child types|attributes|attribute types|
|-----|-----|-----|-----|-----|-----|
|<a name="kgwrappedspace\<kgcirclewire\>">`circle_wire_space`</a>|[*KGCircleWireBuilder.cc*](https://github.com/KATRIN-Experiment/Kassiopeia/tree/main/KGeoBag/Source/Bindings/Shapes/Complex/Source/KGCircleWireBuilder.cc)|[`circle_wire`](#kgcirclewire)|*`KGCircleWire`*|`name`|*`string`*|

#### KGCircleWire

|element name|source files|child elements|child types|attributes|attribute types|
|-----|-----|-----|-----|-----|-----|
|<a name="kgcirclewire">`circle_wire`</a>|[*KGCircleWireBuilder.cc*](https://github.com/KATRIN-Experiment/Kassiopeia/tree/main/KGeoBag/Source/Bindings/Shapes/Complex/Source/KGCircleWireBuilder.cc)|—    |—    |`diameter`<br>`mesh_count`<br>`radius`|*`double`*<br>*`unsigned int`*<br>*`double`*|

### KGWrappedSurface\<KGCircleWire\>

|element name|source files|child elements|child types|attributes|attribute types|
|-----|-----|-----|-----|-----|-----|
|<a name="kgwrappedsurface\<kgcirclewire\>">`circle_wire_surface`</a>|[*KGCircleWireBuilder.cc*](https://github.com/KATRIN-Experiment/Kassiopeia/tree/main/KGeoBag/Source/Bindings/Shapes/Complex/Source/KGCircleWireBuilder.cc)|[`circle_wire`](#kgcirclewire)|*`KGCircleWire`*|`name`|*`string`*|

### KGWrappedSpace\<KGCircularWirePins\>

|element name|source files|child elements|child types|attributes|attribute types|
|-----|-----|-----|-----|-----|-----|
|<a name="kgwrappedspace\<kgcircularwirepins\>">`circular_wire_pins_space`</a>|[*KGCircularWirePinsBuilder.cc*](https://github.com/KATRIN-Experiment/Kassiopeia/tree/main/KGeoBag/Source/Bindings/Shapes/Complex/Source/KGCircularWirePinsBuilder.cc)|[`circular_wire_pins`](#kgcircularwirepins)|*`KGCircularWirePins`*|`name`|*`string`*|

#### KGCircularWirePins

|element name|source files|child elements|child types|attributes|attribute types|
|-----|-----|-----|-----|-----|-----|
|<a name="kgcircularwirepins">`circular_wire_pins`</a>|[*KGCircularWirePinsBuilder.cc*](https://github.com/KATRIN-Experiment/Kassiopeia/tree/main/KGeoBag/Source/Bindings/Shapes/Complex/Source/KGCircularWirePinsBuilder.cc)|—    |—    |`diameter`<br>`inner_radius`<br>`mesh_count`<br>`mesh_power`<br>`n_pins`<br>`outer_radius`<br>`rotation_angle`|*`double`*<br>*`double`*<br>*`unsigned int`*<br>*`double`*<br>*`unsigned int`*<br>*`double`*<br>*`double`*|

### KGWrappedSurface\<KGCircularWirePins\>

|element name|source files|child elements|child types|attributes|attribute types|
|-----|-----|-----|-----|-----|-----|
|<a name="kgwrappedsurface\<kgcircularwirepins\>">`circular_wire_pins_surface`</a>|[*KGCircularWirePinsBuilder.cc*](https://github.com/KATRIN-Experiment/Kassiopeia/tree/main/KGeoBag/Source/Bindings/Shapes/Complex/Source/KGCircularWirePinsBuilder.cc)|[`circular_wire_pins`](#kgcircularwirepins)|*`KGCircularWirePins`*|`name`|*`string`*|

### KGWrappedSurface\<KGComplexAnnulus\>

|element name|source files|child elements|child types|attributes|attribute types|
|-----|-----|-----|-----|-----|-----|
|<a name="kgwrappedsurface\<kgcomplexannulus\>">`complex_annulus_surface`</a>|[*KGComplexAnnulusBuilder.cc*](https://github.com/KATRIN-Experiment/Kassiopeia/tree/main/KGeoBag/Source/Bindings/Shapes/Complex/Source/KGComplexAnnulusBuilder.cc)<br>[*KGComplexAnnulusBuilder.cc*](https://github.com/KATRIN-Experiment/Kassiopeia/tree/main/KGeoBag/Source/Shapes/Complex/Source/KGComplexAnnulusBuilder.cc)|[`complex_annulus`](#kgcomplexannulus)|*`KGComplexAnnulus`*|`name`|*`string`*|

#### KGComplexAnnulus

|element name|source files|child elements|child types|attributes|attribute types|
|-----|-----|-----|-----|-----|-----|
|<a name="kgcomplexannulus">`complex_annulus`</a>|[*KGComplexAnnulusBuilder.cc*](https://github.com/KATRIN-Experiment/Kassiopeia/tree/main/KGeoBag/Source/Bindings/Shapes/Complex/Source/KGComplexAnnulusBuilder.cc)<br>[*KGComplexAnnulusBuilder.cc*](https://github.com/KATRIN-Experiment/Kassiopeia/tree/main/KGeoBag/Source/Shapes/Complex/Source/KGComplexAnnulusBuilder.cc)|[`ring`](#kgcomplexannulusring)|*`KGComplexAnnulusRing`*|`axial_mesh_count`<br>`radial_mesh_count`<br>`radius`|*`int`*<br>*`int`*<br>*`double`*|

##### KGComplexAnnulusRing

|element name|source files|child elements|child types|attributes|attribute types|
|-----|-----|-----|-----|-----|-----|
|<a name="kgcomplexannulusring">`ring`</a>|[*KGComplexAnnulusBuilder.cc*](https://github.com/KATRIN-Experiment/Kassiopeia/tree/main/KGeoBag/Source/Bindings/Shapes/Complex/Source/KGComplexAnnulusBuilder.cc)<br>[*KGComplexAnnulusBuilder.cc*](https://github.com/KATRIN-Experiment/Kassiopeia/tree/main/KGeoBag/Source/Shapes/Complex/Source/KGComplexAnnulusBuilder.cc)|—    |—    |`radius`<br>`x`<br>`y`|*`double`*<br>*`double`*<br>*`double`*|

### KGConeSpace

|element name|source files|child elements|child types|attributes|attribute types|
|-----|-----|-----|-----|-----|-----|
|<a name="kgconespace">`cone_space`</a>|[*KGConeSpaceBuilder.cc*](https://github.com/KATRIN-Experiment/Kassiopeia/tree/main/KGeoBag/Source/Bindings/Shapes/RotatedVolumes/Source/KGConeSpaceBuilder.cc)|—    |—    |`axial_mesh_count`<br>`longitudinal_mesh_count`<br>`longitudinal_mesh_power`<br>`name`<br>`radial_mesh_count`<br>`radial_mesh_power`<br>`rb`<br>`za`<br>`zb`|*`unsigned int`*<br>*`unsigned int`*<br>*`double`*<br>*`string`*<br>*`unsigned int`*<br>*`double`*<br>*`double`*<br>*`double`*<br>*`double`*|

### KGConeSurface

|element name|source files|child elements|child types|attributes|attribute types|
|-----|-----|-----|-----|-----|-----|
|<a name="kgconesurface">`cone_surface`</a>|[*KGConeSurfaceBuilder.cc*](https://github.com/KATRIN-Experiment/Kassiopeia/tree/main/KGeoBag/Source/Bindings/Shapes/RotatedAreas/Source/KGConeSurfaceBuilder.cc)|—    |—    |`axial_mesh_count`<br>`longitudinal_mesh_count`<br>`longitudinal_mesh_power`<br>`name`<br>`rb`<br>`za`<br>`zb`|*`unsigned int`*<br>*`unsigned int`*<br>*`double`*<br>*`string`*<br>*`double`*<br>*`double`*<br>*`double`*|

### KGWrappedSpace\<KGConicSectPortHousing\>

|element name|source files|child elements|child types|attributes|attribute types|
|-----|-----|-----|-----|-----|-----|
|<a name="kgwrappedspace\<kgconicsectporthousing\>">`conic_section_port_housing_space`</a>|[*KGConicSectPortHousingBuilder.cc*](https://github.com/KATRIN-Experiment/Kassiopeia/tree/main/KGeoBag/Source/Bindings/Shapes/Complex/Source/KGConicSectPortHousingBuilder.cc)|[`conic_section_port_housing`](#kgconicsectporthousing)|*`KGConicSectPortHousing`*|`name`|*`string`*|

#### KGConicSectPortHousing

|element name|source files|child elements|child types|attributes|attribute types|
|-----|-----|-----|-----|-----|-----|
|<a name="kgconicsectporthousing">`conic_section_port_housing`</a>|[*KGConicSectPortHousingBuilder.cc*](https://github.com/KATRIN-Experiment/Kassiopeia/tree/main/KGeoBag/Source/Bindings/Shapes/Complex/Source/KGConicSectPortHousingBuilder.cc)|[`orthogonal_port`](#kgconicsectporthousingorthogonalport)<br>[`paraxial_port`](#kgconicsectporthousingparaxialport)|*`KGConicSectPortHousingOrthogonalPort`*<br>*`KGConicSectPortHousingParaxialPort`*|`axial_mesh_count`<br>`longitudinal_mesh_count`<br>`r1`<br>`r2`<br>`z1`<br>`z2`|*`int`*<br>*`int`*<br>*`double`*<br>*`double`*<br>*`double`*<br>*`double`*|

##### KGConicSectPortHousingOrthogonalPort

|element name|source files|child elements|child types|attributes|attribute types|
|-----|-----|-----|-----|-----|-----|
|<a name="kgconicsectporthousingorthogonalport">`orthogonal_port`</a>|[*KGConicSectPortHousingBuilder.cc*](https://github.com/KATRIN-Experiment/Kassiopeia/tree/main/KGeoBag/Source/Bindings/Shapes/Complex/Source/KGConicSectPortHousingBuilder.cc)|—    |—    |`box_curve_mesh_count`<br>`box_radial_mesh_count`<br>`cylinder_axial_mesh_count`<br>`cylinder_longitudinal_mesh_count`<br>`radius`<br>`x`<br>`y`<br>`z`|*`int`*<br>*`int`*<br>*`int`*<br>*`int`*<br>*`double`*<br>*`double`*<br>*`double`*<br>*`double`*|

##### KGConicSectPortHousingParaxialPort

|element name|source files|child elements|child types|attributes|attribute types|
|-----|-----|-----|-----|-----|-----|
|<a name="kgconicsectporthousingparaxialport">`paraxial_port`</a>|[*KGConicSectPortHousingBuilder.cc*](https://github.com/KATRIN-Experiment/Kassiopeia/tree/main/KGeoBag/Source/Bindings/Shapes/Complex/Source/KGConicSectPortHousingBuilder.cc)|—    |—    |`box_curve_mesh_count`<br>`box_radial_mesh_count`<br>`cylinder_axial_mesh_count`<br>`cylinder_longitudinal_mesh_count`<br>`radius`<br>`x`<br>`y`<br>`z`|*`int`*<br>*`int`*<br>*`int`*<br>*`int`*<br>*`double`*<br>*`double`*<br>*`double`*<br>*`double`*|

### KGWrappedSurface\<KGConicSectPortHousing\>

|element name|source files|child elements|child types|attributes|attribute types|
|-----|-----|-----|-----|-----|-----|
|<a name="kgwrappedsurface\<kgconicsectporthousing\>">`conic_section_port_housing_surface`</a>|[*KGConicSectPortHousingBuilder.cc*](https://github.com/KATRIN-Experiment/Kassiopeia/tree/main/KGeoBag/Source/Bindings/Shapes/Complex/Source/KGConicSectPortHousingBuilder.cc)|[`conic_section_port_housing`](#kgconicsectporthousing)|*`KGConicSectPortHousing`*|`name`|*`string`*|

### KGWrappedSpace\<KGConicalWireArray\>

|element name|source files|child elements|child types|attributes|attribute types|
|-----|-----|-----|-----|-----|-----|
|<a name="kgwrappedspace\<kgconicalwirearray\>">`conical_wire_array_space`</a>|[*KGConicalWireArrayBuilder.cc*](https://github.com/KATRIN-Experiment/Kassiopeia/tree/main/KGeoBag/Source/Bindings/Shapes/Complex/Source/KGConicalWireArrayBuilder.cc)|[`conical_wire_array`](#kgconicalwirearray)|*`KGConicalWireArray`*|`name`|*`string`*|

#### KGConicalWireArray

|element name|source files|child elements|child types|attributes|attribute types|
|-----|-----|-----|-----|-----|-----|
|<a name="kgconicalwirearray">`conical_wire_array`</a>|[*KGConicalWireArrayBuilder.cc*](https://github.com/KATRIN-Experiment/Kassiopeia/tree/main/KGeoBag/Source/Bindings/Shapes/Complex/Source/KGConicalWireArrayBuilder.cc)|—    |—    |`diameter`<br>`longitudinal_mesh_count`<br>`longitudinal_mesh_power`<br>`radius1`<br>`radius2`<br>`theta_start`<br>`wire_count`<br>`z1`<br>`z2`|*`double`*<br>*`unsigned int`*<br>*`double`*<br>*`double`*<br>*`double`*<br>*`double`*<br>*`unsigned int`*<br>*`double`*<br>*`double`*|

### KGWrappedSurface\<KGConicalWireArray\>

|element name|source files|child elements|child types|attributes|attribute types|
|-----|-----|-----|-----|-----|-----|
|<a name="kgwrappedsurface\<kgconicalwirearray\>">`conical_wire_array_surface`</a>|[*KGConicalWireArrayBuilder.cc*](https://github.com/KATRIN-Experiment/Kassiopeia/tree/main/KGeoBag/Source/Bindings/Shapes/Complex/Source/KGConicalWireArrayBuilder.cc)|[`conical_wire_array`](#kgconicalwirearray)|*`KGConicalWireArray`*|`name`|*`string`*|

### KGCutConeSpace

|element name|source files|child elements|child types|attributes|attribute types|
|-----|-----|-----|-----|-----|-----|
|<a name="kgcutconespace">`cut_cone_space`</a>|[*KGCutConeSpaceBuilder.cc*](https://github.com/KATRIN-Experiment/Kassiopeia/tree/main/KGeoBag/Source/Bindings/Shapes/RotatedVolumes/Source/KGCutConeSpaceBuilder.cc)|—    |—    |`axial_mesh_count`<br>`longitudinal_mesh_count`<br>`longitudinal_mesh_power`<br>`name`<br>`r1`<br>`r2`<br>`radial_mesh_count`<br>`radial_mesh_power`<br>`z1`<br>`z2`|*`unsigned int`*<br>*`unsigned int`*<br>*`double`*<br>*`string`*<br>*`double`*<br>*`double`*<br>*`unsigned int`*<br>*`double`*<br>*`double`*<br>*`double`*|

### KGCutConeSurface

|element name|source files|child elements|child types|attributes|attribute types|
|-----|-----|-----|-----|-----|-----|
|<a name="kgcutconesurface">`cut_cone_surface`</a>|[*KGCutConeSurfaceBuilder.cc*](https://github.com/KATRIN-Experiment/Kassiopeia/tree/main/KGeoBag/Source/Bindings/Shapes/RotatedAreas/Source/KGCutConeSurfaceBuilder.cc)|—    |—    |`axial_mesh_count`<br>`longitudinal_mesh_count`<br>`longitudinal_mesh_power`<br>`name`<br>`r1`<br>`r2`<br>`z1`<br>`z2`|*`unsigned int`*<br>*`unsigned int`*<br>*`double`*<br>*`string`*<br>*`double`*<br>*`double`*<br>*`double`*<br>*`double`*|

### KGCutConeTubeSpace

|element name|source files|child elements|child types|attributes|attribute types|
|-----|-----|-----|-----|-----|-----|
|<a name="kgcutconetubespace">`cut_cone_tube_space`</a>|[*KGCutConeTubeSpaceBuilder.cc*](https://github.com/KATRIN-Experiment/Kassiopeia/tree/main/KGeoBag/Source/Bindings/Shapes/RotatedVolumes/Source/KGCutConeTubeSpaceBuilder.cc)|—    |—    |`axial_mesh_count`<br>`longitudinal_mesh_count`<br>`longitudinal_mesh_power`<br>`name`<br>`r11`<br>`r12`<br>`r21`<br>`r22`<br>`radial_mesh_count`<br>`radial_mesh_power`<br>`z1`<br>`z2`|*`unsigned int`*<br>*`unsigned int`*<br>*`double`*<br>*`string`*<br>*`double`*<br>*`double`*<br>*`double`*<br>*`double`*<br>*`unsigned int`*<br>*`double`*<br>*`double`*<br>*`double`*|

### KGCutTorusSurface

|element name|source files|child elements|child types|attributes|attribute types|
|-----|-----|-----|-----|-----|-----|
|<a name="kgcuttorussurface">`cut_torus_surface`</a>|[*KGCutTorusSurfaceBuilder.cc*](https://github.com/KATRIN-Experiment/Kassiopeia/tree/main/KGeoBag/Source/Bindings/Shapes/RotatedAreas/Source/KGCutTorusSurfaceBuilder.cc)|—    |—    |`axial_mesh_count`<br>`name`<br>`r1`<br>`r2`<br>`radius`<br>`right`<br>`short`<br>`toroidal_mesh_count`<br>`z1`<br>`z2`|*`unsigned int`*<br>*`string`*<br>*`double`*<br>*`double`*<br>*`double`*<br>*`bool`*<br>*`bool`*<br>*`unsigned int`*<br>*`double`*<br>*`double`*|

### KGCylinderSpace

|element name|source files|child elements|child types|attributes|attribute types|
|-----|-----|-----|-----|-----|-----|
|<a name="kgcylinderspace">`cylinder_space`</a>|[*KGCylinderSpaceBuilder.cc*](https://github.com/KATRIN-Experiment/Kassiopeia/tree/main/KGeoBag/Source/Bindings/Shapes/RotatedVolumes/Source/KGCylinderSpaceBuilder.cc)|—    |—    |`axial_mesh_count`<br>`length`<br>`longitudinal_mesh_count`<br>`longitudinal_mesh_power`<br>`name`<br>`r`<br>`radial_mesh_count`<br>`radial_mesh_power`<br>`z1`<br>`z2`|*`unsigned int`*<br>*`double`*<br>*`unsigned int`*<br>*`double`*<br>*`string`*<br>*`double`*<br>*`unsigned int`*<br>*`double`*<br>*`double`*<br>*`double`*|

### KGCylinderSurface

|element name|source files|child elements|child types|attributes|attribute types|
|-----|-----|-----|-----|-----|-----|
|<a name="kgcylindersurface">`cylinder_surface`</a>|[*KGCylinderSurfaceBuilder.cc*](https://github.com/KATRIN-Experiment/Kassiopeia/tree/main/KGeoBag/Source/Bindings/Shapes/RotatedAreas/Source/KGCylinderSurfaceBuilder.cc)|—    |—    |`axial_mesh_count`<br>`length`<br>`longitudinal_mesh_count`<br>`longitudinal_mesh_power`<br>`name`<br>`r`<br>`z1`<br>`z2`|*`unsigned int`*<br>*`double`*<br>*`unsigned int`*<br>*`double`*<br>*`string`*<br>*`double`*<br>*`double`*<br>*`double`*|

### KGCylinderTubeSpace

|element name|source files|child elements|child types|attributes|attribute types|
|-----|-----|-----|-----|-----|-----|
|<a name="kgcylindertubespace">`cylinder_tube_space`</a>|[*KGCylinderTubeSpaceBuilder.cc*](https://github.com/KATRIN-Experiment/Kassiopeia/tree/main/KGeoBag/Source/Bindings/Shapes/RotatedVolumes/Source/KGCylinderTubeSpaceBuilder.cc)|—    |—    |`axial_mesh_count`<br>`length`<br>`longitudinal_mesh_count`<br>`longitudinal_mesh_power`<br>`name`<br>`r1`<br>`r2`<br>`radial_mesh_count`<br>`radial_mesh_power`<br>`z1`<br>`z2`|*`unsigned int`*<br>*`double`*<br>*`unsigned int`*<br>*`double`*<br>*`string`*<br>*`double`*<br>*`double`*<br>*`unsigned int`*<br>*`double`*<br>*`double`*<br>*`double`*|

### KGDiscreteRotationalMeshAttributor

|element name|source files|child elements|child types|attributes|attribute types|
|-----|-----|-----|-----|-----|-----|
|<a name="kgdiscreterotationalmeshattributor">`discrete_rotational_mesh`</a>|[*KGDiscreteRotationalMeshBuilder.cc*](https://github.com/KATRIN-Experiment/Kassiopeia/tree/main/KGeoBag/Source/Bindings/Extensions/DiscreteRotationalMesh/Source/KGDiscreteRotationalMeshBuilder.cc)|—    |—    |`angle`<br>`count`<br>`name`<br>`spaces`<br>`surfaces`|*`double`*<br>*`int`*<br>*`string`*<br>*`string`*<br>*`string`*|

### KGDiskSurface

|element name|source files|child elements|child types|attributes|attribute types|
|-----|-----|-----|-----|-----|-----|
|<a name="kgdisksurface">`disk_surface`</a>|[*KGDiskSurfaceBuilder.cc*](https://github.com/KATRIN-Experiment/Kassiopeia/tree/main/KGeoBag/Source/Bindings/Shapes/RotatedAreas/Source/KGDiskSurfaceBuilder.cc)|—    |—    |`axial_mesh_count`<br>`name`<br>`r`<br>`radial_mesh_count`<br>`radial_mesh_power`<br>`z`|*`unsigned int`*<br>*`string`*<br>*`double`*<br>*`unsigned int`*<br>*`double`*<br>*`double`*|

### KGElectromagnetAttributor

|element name|source files|child elements|child types|attributes|attribute types|
|-----|-----|-----|-----|-----|-----|
|<a name="kgelectromagnetattributor">`electromagnet`</a>|[*KGElectromagnetBuilder.cc*](https://github.com/KATRIN-Experiment/Kassiopeia/tree/main/KEMField/Source/Bindings/KGeoBag/src/KGElectromagnetBuilder.cc)|—    |—    |`current`<br>`direction`<br>`name`<br>`num_turns`<br>`scaling_factor`<br>`spaces`<br>`surfaces`|*`double`*<br>*`string`*<br>*`string`*<br>*`double`*<br>*`double`*<br>*`string`*<br>*`string`*|

### KGExtrudedArcSegmentSurface

|element name|source files|child elements|child types|attributes|attribute types|
|-----|-----|-----|-----|-----|-----|
|<a name="kgextrudedarcsegmentsurface">`extruded_arc_segment_surface`</a>|[*KGExtrudedArcSegmentSurfaceBuilder.cc*](https://github.com/KATRIN-Experiment/Kassiopeia/tree/main/KGeoBag/Source/Bindings/Shapes/ExtrudedAreas/Source/KGExtrudedArcSegmentSurfaceBuilder.cc)|[`arc_segment`](#kgplanararcsegment)|*`KGPlanarArcSegment`*|`extruded_mesh_count`<br>`extruded_mesh_power`<br>`name`<br>`zmax`<br>`zmin`|*`unsigned int`*<br>*`double`*<br>*`string`*<br>*`double`*<br>*`double`*|

#### KGPlanarArcSegment

|element name|source files|child elements|child types|attributes|attribute types|
|-----|-----|-----|-----|-----|-----|
|<a name="kgplanararcsegment">`arc_segment`</a>|[*KGExtrudedArcSegmentSurfaceBuilder.cc*](https://github.com/KATRIN-Experiment/Kassiopeia/tree/main/KGeoBag/Source/Bindings/Shapes/ExtrudedAreas/Source/KGExtrudedArcSegmentSurfaceBuilder.cc)<br>[*KGRotatedArcSegmentSurfaceBuilder.cc*](https://github.com/KATRIN-Experiment/Kassiopeia/tree/main/KGeoBag/Source/Bindings/Shapes/RotatedAreas/Source/KGRotatedArcSegmentSurfaceBuilder.cc)<br>[*KGRotatedArcSegmentSpaceBuilder.cc*](https://github.com/KATRIN-Experiment/Kassiopeia/tree/main/KGeoBag/Source/Bindings/Shapes/RotatedVolumes/Source/KGRotatedArcSegmentSpaceBuilder.cc)<br>[*KGShellArcSegmentSurfaceBuilder.cc*](https://github.com/KATRIN-Experiment/Kassiopeia/tree/main/KGeoBag/Source/Bindings/Shapes/ShellAreas/Source/KGShellArcSegmentSurfaceBuilder.cc)|—    |—    |`arc_mesh_count`<br>`radius`<br>`right`<br>`short`<br>`x1`<br>`x2`<br>`y1`<br>`y2`|*`unsigned int`*<br>*`double`*<br>*`bool`*<br>*`bool`*<br>*`double`*<br>*`double`*<br>*`double`*<br>*`double`*|

### KGExtrudedCircleSpace

|element name|source files|child elements|child types|attributes|attribute types|
|-----|-----|-----|-----|-----|-----|
|<a name="kgextrudedcirclespace">`extruded_circle_space`</a>|[*KGExtrudedCircleSpaceBuilder.cc*](https://github.com/KATRIN-Experiment/Kassiopeia/tree/main/KGeoBag/Source/Bindings/Shapes/ExtrudedVolumes/Source/KGExtrudedCircleSpaceBuilder.cc)|[`circle`](#kgplanarcircle)|*`KGPlanarCircle`*|`extruded_mesh_count`<br>`extruded_mesh_power`<br>`flattened_mesh_count`<br>`flattened_mesh_power`<br>`name`<br>`zmax`<br>`zmin`|*`unsigned int`*<br>*`double`*<br>*`unsigned int`*<br>*`double`*<br>*`string`*<br>*`double`*<br>*`double`*|

#### KGPlanarCircle

|element name|source files|child elements|child types|attributes|attribute types|
|-----|-----|-----|-----|-----|-----|
|<a name="kgplanarcircle">`circle`</a>|[*KGExtrudedCircleSurfaceBuilder.cc*](https://github.com/KATRIN-Experiment/Kassiopeia/tree/main/KGeoBag/Source/Bindings/Shapes/ExtrudedAreas/Source/KGExtrudedCircleSurfaceBuilder.cc)<br>[*KGExtrudedCircleSpaceBuilder.cc*](https://github.com/KATRIN-Experiment/Kassiopeia/tree/main/KGeoBag/Source/Bindings/Shapes/ExtrudedVolumes/Source/KGExtrudedCircleSpaceBuilder.cc)<br>[*KGFlattenedCircleSurfaceBuilder.cc*](https://github.com/KATRIN-Experiment/Kassiopeia/tree/main/KGeoBag/Source/Bindings/Shapes/FlattenedAreas/Source/KGFlattenedCircleSurfaceBuilder.cc)<br>[*KGRotatedCircleSurfaceBuilder.cc*](https://github.com/KATRIN-Experiment/Kassiopeia/tree/main/KGeoBag/Source/Bindings/Shapes/RotatedAreas/Source/KGRotatedCircleSurfaceBuilder.cc)<br>[*KGRotatedCircleSpaceBuilder.cc*](https://github.com/KATRIN-Experiment/Kassiopeia/tree/main/KGeoBag/Source/Bindings/Shapes/RotatedVolumes/Source/KGRotatedCircleSpaceBuilder.cc)<br>[*KGShellCircleSurfaceBuilder.cc*](https://github.com/KATRIN-Experiment/Kassiopeia/tree/main/KGeoBag/Source/Bindings/Shapes/ShellAreas/Source/KGShellCircleSurfaceBuilder.cc)|—    |—    |`circle_mesh_count`<br>`radius`<br>`x`<br>`y`|*`unsigned int`*<br>*`double`*<br>*`double`*<br>*`double`*|

### KGExtrudedCircleSurface

|element name|source files|child elements|child types|attributes|attribute types|
|-----|-----|-----|-----|-----|-----|
|<a name="kgextrudedcirclesurface">`extruded_circle_surface`</a>|[*KGExtrudedCircleSurfaceBuilder.cc*](https://github.com/KATRIN-Experiment/Kassiopeia/tree/main/KGeoBag/Source/Bindings/Shapes/ExtrudedAreas/Source/KGExtrudedCircleSurfaceBuilder.cc)|[`circle`](#kgplanarcircle)|*`KGPlanarCircle`*|`extruded_mesh_count`<br>`extruded_mesh_power`<br>`name`<br>`zmax`<br>`zmin`|*`unsigned int`*<br>*`double`*<br>*`string`*<br>*`double`*<br>*`double`*|

### KGExtrudedLineSegmentSurface

|element name|source files|child elements|child types|attributes|attribute types|
|-----|-----|-----|-----|-----|-----|
|<a name="kgextrudedlinesegmentsurface">`extruded_line_segment_surface`</a>|[*KGExtrudedLineSegmentSurfaceBuilder.cc*](https://github.com/KATRIN-Experiment/Kassiopeia/tree/main/KGeoBag/Source/Bindings/Shapes/ExtrudedAreas/Source/KGExtrudedLineSegmentSurfaceBuilder.cc)|[`line_segment`](#kgplanarlinesegment)|*`KGPlanarLineSegment`*|`extruded_mesh_count`<br>`extruded_mesh_power`<br>`name`<br>`zmax`<br>`zmin`|*`unsigned int`*<br>*`double`*<br>*`string`*<br>*`double`*<br>*`double`*|

#### KGPlanarLineSegment

|element name|source files|child elements|child types|attributes|attribute types|
|-----|-----|-----|-----|-----|-----|
|<a name="kgplanarlinesegment">`line_segment`</a>|[*KGExtrudedLineSegmentSurfaceBuilder.cc*](https://github.com/KATRIN-Experiment/Kassiopeia/tree/main/KGeoBag/Source/Bindings/Shapes/ExtrudedAreas/Source/KGExtrudedLineSegmentSurfaceBuilder.cc)<br>[*KGRotatedLineSegmentSurfaceBuilder.cc*](https://github.com/KATRIN-Experiment/Kassiopeia/tree/main/KGeoBag/Source/Bindings/Shapes/RotatedAreas/Source/KGRotatedLineSegmentSurfaceBuilder.cc)<br>[*KGRotatedLineSegmentSpaceBuilder.cc*](https://github.com/KATRIN-Experiment/Kassiopeia/tree/main/KGeoBag/Source/Bindings/Shapes/RotatedVolumes/Source/KGRotatedLineSegmentSpaceBuilder.cc)<br>[*KGShellLineSegmentSurfaceBuilder.cc*](https://github.com/KATRIN-Experiment/Kassiopeia/tree/main/KGeoBag/Source/Bindings/Shapes/ShellAreas/Source/KGShellLineSegmentSurfaceBuilder.cc)|—    |—    |`line_mesh_count`<br>`line_mesh_power`<br>`x1`<br>`x2`<br>`y1`<br>`y2`|*`unsigned int`*<br>*`double`*<br>*`double`*<br>*`double`*<br>*`double`*<br>*`double`*|

### KGExtrudedPolyLineSurface

|element name|source files|child elements|child types|attributes|attribute types|
|-----|-----|-----|-----|-----|-----|
|<a name="kgextrudedpolylinesurface">`extruded_poly_line_surface`</a>|[*KGExtrudedPolyLineSurfaceBuilder.cc*](https://github.com/KATRIN-Experiment/Kassiopeia/tree/main/KGeoBag/Source/Bindings/Shapes/ExtrudedAreas/Source/KGExtrudedPolyLineSurfaceBuilder.cc)|[`poly_line`](#kgplanarpolyline)|*`KGPlanarPolyLine`*|`extruded_mesh_count`<br>`extruded_mesh_power`<br>`name`<br>`zmax`<br>`zmin`|*`unsigned int`*<br>*`double`*<br>*`string`*<br>*`double`*<br>*`double`*|

#### KGPlanarPolyLine

|element name|source files|child elements|child types|attributes|attribute types|
|-----|-----|-----|-----|-----|-----|
|<a name="kgplanarpolyline">`poly_line`</a>|[*KGExtrudedPolyLineSurfaceBuilder.cc*](https://github.com/KATRIN-Experiment/Kassiopeia/tree/main/KGeoBag/Source/Bindings/Shapes/ExtrudedAreas/Source/KGExtrudedPolyLineSurfaceBuilder.cc)<br>[*KGRotatedPolyLineSurfaceBuilder.cc*](https://github.com/KATRIN-Experiment/Kassiopeia/tree/main/KGeoBag/Source/Bindings/Shapes/RotatedAreas/Source/KGRotatedPolyLineSurfaceBuilder.cc)<br>[*KGRotatedPolyLineSpaceBuilder.cc*](https://github.com/KATRIN-Experiment/Kassiopeia/tree/main/KGeoBag/Source/Bindings/Shapes/RotatedVolumes/Source/KGRotatedPolyLineSpaceBuilder.cc)<br>[*KGShellPolyLineSurfaceBuilder.cc*](https://github.com/KATRIN-Experiment/Kassiopeia/tree/main/KGeoBag/Source/Bindings/Shapes/ShellAreas/Source/KGShellPolyLineSurfaceBuilder.cc)|[`next_arc`](#kgplanarpolylinearcarguments)<br>[`next_line`](#kgplanarpolylinelinearguments)<br>[`previous_arc`](#kgplanarpolylinearcarguments)<br>[`previous_line`](#kgplanarpolylinelinearguments)<br>[`start_point`](#kgplanarpolylinestartpointarguments)|*`KGPlanarPolyLineArcArguments`*<br>*`KGPlanarPolyLineLineArguments`*<br>*`KGPlanarPolyLineArcArguments`*<br>*`KGPlanarPolyLineLineArguments`*<br>*`KGPlanarPolyLineStartPointArguments`*|—    |—    |

##### KGPlanarPolyLineArcArguments

|element name|source files|child elements|child types|attributes|attribute types|
|-----|-----|-----|-----|-----|-----|
|<a name="kgplanarpolylinearcarguments">`next_arc`</a>|[*KGPlanarPolyLineBuilder.cc*](https://github.com/KATRIN-Experiment/Kassiopeia/tree/main/KGeoBag/Source/Bindings/Shapes/PlanarShapes/Source/KGPlanarPolyLineBuilder.cc)|—    |—    |`arc_mesh_count`<br>`radius`<br>`right`<br>`short`<br>`x`<br>`y`|*`unsigned int`*<br>*`double`*<br>*`bool`*<br>*`bool`*<br>*`double`*<br>*`double`*|

##### KGPlanarPolyLineLineArguments

|element name|source files|child elements|child types|attributes|attribute types|
|-----|-----|-----|-----|-----|-----|
|<a name="kgplanarpolylinelinearguments">`next_line`</a>|[*KGPlanarPolyLineBuilder.cc*](https://github.com/KATRIN-Experiment/Kassiopeia/tree/main/KGeoBag/Source/Bindings/Shapes/PlanarShapes/Source/KGPlanarPolyLineBuilder.cc)|—    |—    |`line_mesh_count`<br>`line_mesh_power`<br>`x`<br>`y`|*`unsigned int`*<br>*`double`*<br>*`double`*<br>*`double`*|

##### KGPlanarPolyLineStartPointArguments

|element name|source files|child elements|child types|attributes|attribute types|
|-----|-----|-----|-----|-----|-----|
|<a name="kgplanarpolylinestartpointarguments">`start_point`</a>|[*KGPlanarPolyLineBuilder.cc*](https://github.com/KATRIN-Experiment/Kassiopeia/tree/main/KGeoBag/Source/Bindings/Shapes/PlanarShapes/Source/KGPlanarPolyLineBuilder.cc)|—    |—    |`x`<br>`y`|*`double`*<br>*`double`*|

### KGExtrudedPolyLoopSpace

|element name|source files|child elements|child types|attributes|attribute types|
|-----|-----|-----|-----|-----|-----|
|<a name="kgextrudedpolyloopspace">`extruded_poly_loop_space`</a>|[*KGExtrudedPolyLoopSpaceBuilder.cc*](https://github.com/KATRIN-Experiment/Kassiopeia/tree/main/KGeoBag/Source/Bindings/Shapes/ExtrudedVolumes/Source/KGExtrudedPolyLoopSpaceBuilder.cc)|[`poly_loop`](#kgplanarpolyloop)|*`KGPlanarPolyLoop`*|`extruded_mesh_count`<br>`extruded_mesh_power`<br>`flattened_mesh_count`<br>`flattened_mesh_power`<br>`name`<br>`zmax`<br>`zmin`|*`unsigned int`*<br>*`double`*<br>*`unsigned int`*<br>*`double`*<br>*`string`*<br>*`double`*<br>*`double`*|

#### KGPlanarPolyLoop

|element name|source files|child elements|child types|attributes|attribute types|
|-----|-----|-----|-----|-----|-----|
|<a name="kgplanarpolyloop">`poly_loop`</a>|[*KGExtrudedPolyLoopSurfaceBuilder.cc*](https://github.com/KATRIN-Experiment/Kassiopeia/tree/main/KGeoBag/Source/Bindings/Shapes/ExtrudedAreas/Source/KGExtrudedPolyLoopSurfaceBuilder.cc)<br>[*KGExtrudedPolyLoopSpaceBuilder.cc*](https://github.com/KATRIN-Experiment/Kassiopeia/tree/main/KGeoBag/Source/Bindings/Shapes/ExtrudedVolumes/Source/KGExtrudedPolyLoopSpaceBuilder.cc)<br>[*KGFlattenedPolyLoopSurfaceBuilder.cc*](https://github.com/KATRIN-Experiment/Kassiopeia/tree/main/KGeoBag/Source/Bindings/Shapes/FlattenedAreas/Source/KGFlattenedPolyLoopSurfaceBuilder.cc)<br>[*KGRotatedPolyLoopSurfaceBuilder.cc*](https://github.com/KATRIN-Experiment/Kassiopeia/tree/main/KGeoBag/Source/Bindings/Shapes/RotatedAreas/Source/KGRotatedPolyLoopSurfaceBuilder.cc)<br>[*KGRotatedPolyLoopSpaceBuilder.cc*](https://github.com/KATRIN-Experiment/Kassiopeia/tree/main/KGeoBag/Source/Bindings/Shapes/RotatedVolumes/Source/KGRotatedPolyLoopSpaceBuilder.cc)<br>[*KGShellPolyLoopSurfaceBuilder.cc*](https://github.com/KATRIN-Experiment/Kassiopeia/tree/main/KGeoBag/Source/Bindings/Shapes/ShellAreas/Source/KGShellPolyLoopSurfaceBuilder.cc)|[`last_arc`](#kgplanarpolylooplastarcarguments)<br>[`last_line`](#kgplanarpolylooplastlinearguments)<br>[`next_arc`](#kgplanarpolylooparcarguments)<br>[`next_line`](#kgplanarpolylooplinearguments)<br>[`previous_arc`](#kgplanarpolylooparcarguments)<br>[`previous_line`](#kgplanarpolylooplinearguments)<br>[`start_point`](#kgplanarpolyloopstartpointarguments)|*`KGPlanarPolyLoopLastArcArguments`*<br>*`KGPlanarPolyLoopLastLineArguments`*<br>*`KGPlanarPolyLoopArcArguments`*<br>*`KGPlanarPolyLoopLineArguments`*<br>*`KGPlanarPolyLoopArcArguments`*<br>*`KGPlanarPolyLoopLineArguments`*<br>*`KGPlanarPolyLoopStartPointArguments`*|—    |—    |

##### KGPlanarPolyLoopLastArcArguments

|element name|source files|child elements|child types|attributes|attribute types|
|-----|-----|-----|-----|-----|-----|
|<a name="kgplanarpolylooplastarcarguments">`last_arc`</a>|[*KGPlanarPolyLoopBuilder.cc*](https://github.com/KATRIN-Experiment/Kassiopeia/tree/main/KGeoBag/Source/Bindings/Shapes/PlanarShapes/Source/KGPlanarPolyLoopBuilder.cc)|—    |—    |`arc_mesh_count`<br>`radius`<br>`right`<br>`short`|*`unsigned int`*<br>*`double`*<br>*`bool`*<br>*`bool`*|

##### KGPlanarPolyLoopLastLineArguments

|element name|source files|child elements|child types|attributes|attribute types|
|-----|-----|-----|-----|-----|-----|
|<a name="kgplanarpolylooplastlinearguments">`last_line`</a>|[*KGPlanarPolyLoopBuilder.cc*](https://github.com/KATRIN-Experiment/Kassiopeia/tree/main/KGeoBag/Source/Bindings/Shapes/PlanarShapes/Source/KGPlanarPolyLoopBuilder.cc)|—    |—    |`line_mesh_count`<br>`line_mesh_power`|*`unsigned int`*<br>*`double`*|

##### KGPlanarPolyLoopArcArguments

|element name|source files|child elements|child types|attributes|attribute types|
|-----|-----|-----|-----|-----|-----|
|<a name="kgplanarpolylooparcarguments">`next_arc`</a>|[*KGPlanarPolyLoopBuilder.cc*](https://github.com/KATRIN-Experiment/Kassiopeia/tree/main/KGeoBag/Source/Bindings/Shapes/PlanarShapes/Source/KGPlanarPolyLoopBuilder.cc)|—    |—    |`arc_mesh_count`<br>`radius`<br>`right`<br>`short`<br>`x`<br>`y`|*`unsigned int`*<br>*`double`*<br>*`bool`*<br>*`bool`*<br>*`double`*<br>*`double`*|

##### KGPlanarPolyLoopLineArguments

|element name|source files|child elements|child types|attributes|attribute types|
|-----|-----|-----|-----|-----|-----|
|<a name="kgplanarpolylooplinearguments">`next_line`</a>|[*KGPlanarPolyLoopBuilder.cc*](https://github.com/KATRIN-Experiment/Kassiopeia/tree/main/KGeoBag/Source/Bindings/Shapes/PlanarShapes/Source/KGPlanarPolyLoopBuilder.cc)|—    |—    |`line_mesh_count`<br>`line_mesh_power`<br>`x`<br>`y`|*`unsigned int`*<br>*`double`*<br>*`double`*<br>*`double`*|

##### KGPlanarPolyLoopStartPointArguments

|element name|source files|child elements|child types|attributes|attribute types|
|-----|-----|-----|-----|-----|-----|
|<a name="kgplanarpolyloopstartpointarguments">`start_point`</a>|[*KGPlanarPolyLoopBuilder.cc*](https://github.com/KATRIN-Experiment/Kassiopeia/tree/main/KGeoBag/Source/Bindings/Shapes/PlanarShapes/Source/KGPlanarPolyLoopBuilder.cc)|—    |—    |`x`<br>`y`|*`double`*<br>*`double`*|

### KGExtrudedPolyLoopSurface

|element name|source files|child elements|child types|attributes|attribute types|
|-----|-----|-----|-----|-----|-----|
|<a name="kgextrudedpolyloopsurface">`extruded_poly_loop_surface`</a>|[*KGExtrudedPolyLoopSurfaceBuilder.cc*](https://github.com/KATRIN-Experiment/Kassiopeia/tree/main/KGeoBag/Source/Bindings/Shapes/ExtrudedAreas/Source/KGExtrudedPolyLoopSurfaceBuilder.cc)|[`poly_loop`](#kgplanarpolyloop)|*`KGPlanarPolyLoop`*|`extruded_mesh_count`<br>`extruded_mesh_power`<br>`name`<br>`zmax`<br>`zmin`|*`unsigned int`*<br>*`double`*<br>*`string`*<br>*`double`*<br>*`double`*|

### KGWrappedSpace\<KGExtrudedObject\>

|element name|source files|child elements|child types|attributes|attribute types|
|-----|-----|-----|-----|-----|-----|
|<a name="kgwrappedspace\<kgextrudedobject\>">`extruded_space`</a>|[*KGExtrudedObjectBuilder.cc*](https://github.com/KATRIN-Experiment/Kassiopeia/tree/main/KGeoBag/Source/Bindings/Shapes/Complex/Source/KGExtrudedObjectBuilder.cc)|[`extruded_object`](#kgextrudedobject)|*`KGExtrudedObject`*|`name`|*`string`*|

#### KGExtrudedObject

|element name|source files|child elements|child types|attributes|attribute types|
|-----|-----|-----|-----|-----|-----|
|<a name="kgextrudedobject">`extruded_object`</a>|[*KGExtrudedObjectBuilder.cc*](https://github.com/KATRIN-Experiment/Kassiopeia/tree/main/KGeoBag/Source/Bindings/Shapes/Complex/Source/KGExtrudedObjectBuilder.cc)|[`inner_arc`](#kgextrudedobjectarc)<br>[`inner_line`](#kgextrudedobjectline)<br>[`outer_arc`](#kgextrudedobjectarc)<br>[`outer_line`](#kgextrudedobjectline)|*`KGExtrudedObjectArc`*<br>*`KGExtrudedObjectLine`*<br>*`KGExtrudedObjectArc`*<br>*`KGExtrudedObjectLine`*|`closed_form`<br>`extruded_mesh_count`<br>`extruded_mesh_power`<br>`longitudinal_mesh_count`<br>`longitudinal_mesh_power`<br>`mesh_merge_distance`<br>`refine_mesh`<br>`z_max`<br>`z_min`|*`bool`*<br>*`int`*<br>*`double`*<br>*`int`*<br>*`double`*<br>*`double`*<br>*`bool`*<br>*`double`*<br>*`double`*|

##### KGExtrudedObjectArc

|element name|source files|child elements|child types|attributes|attribute types|
|-----|-----|-----|-----|-----|-----|
|<a name="kgextrudedobjectarc">`inner_arc`</a>|[*KGExtrudedObjectBuilder.cc*](https://github.com/KATRIN-Experiment/Kassiopeia/tree/main/KGeoBag/Source/Bindings/Shapes/Complex/Source/KGExtrudedObjectBuilder.cc)|—    |—    |`positive_orientation`<br>`radius`<br>`x1`<br>`x2`<br>`y1`<br>`y2`|*`bool`*<br>*`double`*<br>*`double`*<br>*`double`*<br>*`double`*<br>*`double`*|

##### KGExtrudedObjectLine

|element name|source files|child elements|child types|attributes|attribute types|
|-----|-----|-----|-----|-----|-----|
|<a name="kgextrudedobjectline">`inner_line`</a>|[*KGExtrudedObjectBuilder.cc*](https://github.com/KATRIN-Experiment/Kassiopeia/tree/main/KGeoBag/Source/Bindings/Shapes/Complex/Source/KGExtrudedObjectBuilder.cc)|—    |—    |`x1`<br>`x2`<br>`y1`<br>`y2`|*`double`*<br>*`double`*<br>*`double`*<br>*`double`*|

### KGWrappedSurface\<KGExtrudedObject\>

|element name|source files|child elements|child types|attributes|attribute types|
|-----|-----|-----|-----|-----|-----|
|<a name="kgwrappedsurface\<kgextrudedobject\>">`extruded_surface`</a>|[*KGExtrudedObjectBuilder.cc*](https://github.com/KATRIN-Experiment/Kassiopeia/tree/main/KGeoBag/Source/Bindings/Shapes/Complex/Source/KGExtrudedObjectBuilder.cc)|[`extruded_object`](#kgextrudedobject)|*`KGExtrudedObject`*|`name`|*`string`*|

### KGFlattenedCircleSurface

|element name|source files|child elements|child types|attributes|attribute types|
|-----|-----|-----|-----|-----|-----|
|<a name="kgflattenedcirclesurface">`flattened_circle_surface`</a>|[*KGFlattenedCircleSurfaceBuilder.cc*](https://github.com/KATRIN-Experiment/Kassiopeia/tree/main/KGeoBag/Source/Bindings/Shapes/FlattenedAreas/Source/KGFlattenedCircleSurfaceBuilder.cc)|[`circle`](#kgplanarcircle)|*`KGPlanarCircle`*|`flattened_mesh_count`<br>`flattened_mesh_power`<br>`name`<br>`z`|*`unsigned int`*<br>*`double`*<br>*`string`*<br>*`double`*|

### KGFlattenedPolyLoopSurface

|element name|source files|child elements|child types|attributes|attribute types|
|-----|-----|-----|-----|-----|-----|
|<a name="kgflattenedpolyloopsurface">`flattened_poly_loop_surface`</a>|[*KGFlattenedPolyLoopSurfaceBuilder.cc*](https://github.com/KATRIN-Experiment/Kassiopeia/tree/main/KGeoBag/Source/Bindings/Shapes/FlattenedAreas/Source/KGFlattenedPolyLoopSurfaceBuilder.cc)|[`poly_loop`](#kgplanarpolyloop)|*`KGPlanarPolyLoop`*|`flattened_mesh_count`<br>`flattened_mesh_power`<br>`name`<br>`z`|*`unsigned int`*<br>*`double`*<br>*`string`*<br>*`double`*|

### KGWrappedSpace\<KGLinearWireGrid\>

|element name|source files|child elements|child types|attributes|attribute types|
|-----|-----|-----|-----|-----|-----|
|<a name="kgwrappedspace\<kglinearwiregrid\>">`linear_wire_grid_space`</a>|[*KGLinearWireGridBuilder.cc*](https://github.com/KATRIN-Experiment/Kassiopeia/tree/main/KGeoBag/Source/Bindings/Shapes/Complex/Source/KGLinearWireGridBuilder.cc)|[`linear_wire_grid`](#kglinearwiregrid)|*`KGLinearWireGrid`*|`name`|*`string`*|

#### KGLinearWireGrid

|element name|source files|child elements|child types|attributes|attribute types|
|-----|-----|-----|-----|-----|-----|
|<a name="kglinearwiregrid">`linear_wire_grid`</a>|[*KGLinearWireGridBuilder.cc*](https://github.com/KATRIN-Experiment/Kassiopeia/tree/main/KGeoBag/Source/Bindings/Shapes/Complex/Source/KGLinearWireGridBuilder.cc)|—    |—    |`add_outer_circle`<br>`diameter`<br>`longitudinal_mesh_count`<br>`longitudinal_mesh_power`<br>`pitch`<br>`radius`|*`bool`*<br>*`double`*<br>*`unsigned int`*<br>*`double`*<br>*`double`*<br>*`double`*|

### KGWrappedSurface\<KGLinearWireGrid\>

|element name|source files|child elements|child types|attributes|attribute types|
|-----|-----|-----|-----|-----|-----|
|<a name="kgwrappedsurface\<kglinearwiregrid\>">`linear_wire_grid_surface`</a>|[*KGLinearWireGridBuilder.cc*](https://github.com/KATRIN-Experiment/Kassiopeia/tree/main/KGeoBag/Source/Bindings/Shapes/Complex/Source/KGLinearWireGridBuilder.cc)|[`linear_wire_grid`](#kglinearwiregrid)|*`KGLinearWireGrid`*|`name`|*`string`*|

### KGMeshAttributor

|element name|source files|child elements|child types|attributes|attribute types|
|-----|-----|-----|-----|-----|-----|
|<a name="kgmeshattributor">`mesh`</a>|[*KGMeshBuilder.cc*](https://github.com/KATRIN-Experiment/Kassiopeia/tree/main/KGeoBag/Source/Bindings/Extensions/Mesh/Source/KGMeshBuilder.cc)|—    |—    |`name`<br>`spaces`<br>`surfaces`|*`string`*<br>*`string`*<br>*`string`*|

### KGMeshDeformer

|element name|source files|child elements|child types|attributes|attribute types|
|-----|-----|-----|-----|-----|-----|
|<a name="kgmeshdeformer">`mesh_deformer`</a>|[*KGMeshDeformerBuilder.cc*](https://github.com/KATRIN-Experiment/Kassiopeia/tree/main/KGeoBag/Source/Bindings/Extensions/Deformation/Source/KGMeshDeformerBuilder.cc)|—    |—    |`spaces`<br>`surfaces`|*`string`*<br>*`string`*|

### KGMeshRefiner

|element name|source files|child elements|child types|attributes|attribute types|
|-----|-----|-----|-----|-----|-----|
|<a name="kgmeshrefiner">`mesh_refiner`</a>|[*KGMeshRefinerBuilder.cc*](https://github.com/KATRIN-Experiment/Kassiopeia/tree/main/KGeoBag/Source/Bindings/Extensions/Refinement/Source/KGMeshRefinerBuilder.cc)|—    |—    |`max_area`<br>`max_aspect_ratio`<br>`max_length`<br>`max_refinement_steps`<br>`spaces`<br>`surfaces`|*`double`*<br>*`double`*<br>*`double`*<br>*`unsigned int`*<br>*`string`*<br>*`string`*|

### KGWrappedSpace\<KGPlyFile\>

|element name|source files|child elements|child types|attributes|attribute types|
|-----|-----|-----|-----|-----|-----|
|<a name="kgwrappedspace\<kgplyfile\>">`ply_file_space`</a>|[*KGPlyFileBuilder.cc*](https://github.com/KATRIN-Experiment/Kassiopeia/tree/main/KGeoBag/Source/Bindings/Shapes/External/Source/KGPlyFileBuilder.cc)|[`ply_file`](#kgplyfile)|*`KGPlyFile`*|`name`|*`string`*|

#### KGPlyFile

|element name|source files|child elements|child types|attributes|attribute types|
|-----|-----|-----|-----|-----|-----|
|<a name="kgplyfile">`ply_file`</a>|[*KGPlyFileBuilder.cc*](https://github.com/KATRIN-Experiment/Kassiopeia/tree/main/KGeoBag/Source/Bindings/Shapes/External/Source/KGPlyFileBuilder.cc)|—    |—    |`file`<br>`mesh_count`<br>`path`<br>`scale`<br>`selector`|*`string`*<br>*`int`*<br>*`string`*<br>*`double`*<br>*`string`*|

### KGWrappedSurface\<KGPlyFile\>

|element name|source files|child elements|child types|attributes|attribute types|
|-----|-----|-----|-----|-----|-----|
|<a name="kgwrappedsurface\<kgplyfile\>">`ply_file_surface`</a>|[*KGPlyFileBuilder.cc*](https://github.com/KATRIN-Experiment/Kassiopeia/tree/main/KGeoBag/Source/Bindings/Shapes/External/Source/KGPlyFileBuilder.cc)|[`ply_file`](#kgplyfile)|*`KGPlyFile`*|`name`|*`string`*|

### KGWrappedSpace\<KGPortHousing\>

|element name|source files|child elements|child types|attributes|attribute types|
|-----|-----|-----|-----|-----|-----|
|<a name="kgwrappedspace\<kgporthousing\>">`port_housing_space`</a>|[*KGPortHousingBuilder.cc*](https://github.com/KATRIN-Experiment/Kassiopeia/tree/main/KGeoBag/Source/Bindings/Shapes/Complex/Source/KGPortHousingBuilder.cc)|[`port_housing`](#kgporthousing)|*`KGPortHousing`*|`name`|*`string`*|

#### KGPortHousing

|element name|source files|child elements|child types|attributes|attribute types|
|-----|-----|-----|-----|-----|-----|
|<a name="kgporthousing">`port_housing`</a>|[*KGPortHousingBuilder.cc*](https://github.com/KATRIN-Experiment/Kassiopeia/tree/main/KGeoBag/Source/Bindings/Shapes/Complex/Source/KGPortHousingBuilder.cc)|[`circular_port`](#kgporthousingcircularport)<br>[`rectangular_port`](#kgporthousingrectangularport)|*`KGPortHousingCircularPort`*<br>*`KGPortHousingRectangularPort`*|`axial_mesh_count`<br>`longitudinal_mesh_count`<br>`radius`<br>`x1`<br>`x2`<br>`y1`<br>`y2`<br>`z1`<br>`z2`|*`int`*<br>*`int`*<br>*`double`*<br>*`double`*<br>*`double`*<br>*`double`*<br>*`double`*<br>*`double`*<br>*`double`*|

##### KGPortHousingCircularPort

|element name|source files|child elements|child types|attributes|attribute types|
|-----|-----|-----|-----|-----|-----|
|<a name="kgporthousingcircularport">`circular_port`</a>|[*KGPortHousingBuilder.cc*](https://github.com/KATRIN-Experiment/Kassiopeia/tree/main/KGeoBag/Source/Bindings/Shapes/Complex/Source/KGPortHousingBuilder.cc)|—    |—    |`radius`<br>`x`<br>`y`<br>`z`|*`double`*<br>*`double`*<br>*`double`*<br>*`double`*|

##### KGPortHousingRectangularPort

|element name|source files|child elements|child types|attributes|attribute types|
|-----|-----|-----|-----|-----|-----|
|<a name="kgporthousingrectangularport">`rectangular_port`</a>|[*KGPortHousingBuilder.cc*](https://github.com/KATRIN-Experiment/Kassiopeia/tree/main/KGeoBag/Source/Bindings/Shapes/Complex/Source/KGPortHousingBuilder.cc)|—    |—    |`length`<br>`width`<br>`x`<br>`y`<br>`z`|*`double`*<br>*`double`*<br>*`double`*<br>*`double`*<br>*`double`*|

### KGWrappedSurface\<KGPortHousing\>

|element name|source files|child elements|child types|attributes|attribute types|
|-----|-----|-----|-----|-----|-----|
|<a name="kgwrappedsurface\<kgporthousing\>">`port_housing_surface`</a>|[*KGPortHousingBuilder.cc*](https://github.com/KATRIN-Experiment/Kassiopeia/tree/main/KGeoBag/Source/Bindings/Shapes/Complex/Source/KGPortHousingBuilder.cc)|[`port_housing`](#kgporthousing)|*`KGPortHousing`*|`name`|*`string`*|

### KGWrappedSpace\<KGQuadraticWireGrid\>

|element name|source files|child elements|child types|attributes|attribute types|
|-----|-----|-----|-----|-----|-----|
|<a name="kgwrappedspace\<kgquadraticwiregrid\>">`quadratic_wire_grid_space`</a>|[*KGQuadraticWireGridBuilder.cc*](https://github.com/KATRIN-Experiment/Kassiopeia/tree/main/KGeoBag/Source/Bindings/Shapes/Complex/Source/KGQuadraticWireGridBuilder.cc)|[`quadratic_wire_grid`](#kgquadraticwiregrid)|*`KGQuadraticWireGrid`*|`name`|*`string`*|

#### KGQuadraticWireGrid

|element name|source files|child elements|child types|attributes|attribute types|
|-----|-----|-----|-----|-----|-----|
|<a name="kgquadraticwiregrid">`quadratic_wire_grid`</a>|[*KGQuadraticWireGridBuilder.cc*](https://github.com/KATRIN-Experiment/Kassiopeia/tree/main/KGeoBag/Source/Bindings/Shapes/Complex/Source/KGQuadraticWireGridBuilder.cc)|—    |—    |`add_outer_circle`<br>`diameter`<br>`mesh_count_per_pitch`<br>`pitch`<br>`radius`|*`bool`*<br>*`double`*<br>*`unsigned int`*<br>*`double`*<br>*`double`*|

### KGWrappedSurface\<KGQuadraticWireGrid\>

|element name|source files|child elements|child types|attributes|attribute types|
|-----|-----|-----|-----|-----|-----|
|<a name="kgwrappedsurface\<kgquadraticwiregrid\>">`quadratic_wire_grid_surface`</a>|[*KGQuadraticWireGridBuilder.cc*](https://github.com/KATRIN-Experiment/Kassiopeia/tree/main/KGeoBag/Source/Bindings/Shapes/Complex/Source/KGQuadraticWireGridBuilder.cc)|[`quadratic_wire_grid`](#kgquadraticwiregrid)|*`KGQuadraticWireGrid`*|`name`|*`string`*|

### KGWrappedSpace\<KGRod\>

|element name|source files|child elements|child types|attributes|attribute types|
|-----|-----|-----|-----|-----|-----|
|<a name="kgwrappedspace\<kgrod\>">`rod_space`</a>|[*KGRodBuilder.cc*](https://github.com/KATRIN-Experiment/Kassiopeia/tree/main/KGeoBag/Source/Bindings/Shapes/Complex/Source/KGRodBuilder.cc)|[`rod`](#kgrod)|*`KGRod`*|`name`|*`string`*|

#### KGRod

|element name|source files|child elements|child types|attributes|attribute types|
|-----|-----|-----|-----|-----|-----|
|<a name="kgrod">`rod`</a>|[*KGRodBuilder.cc*](https://github.com/KATRIN-Experiment/Kassiopeia/tree/main/KGeoBag/Source/Bindings/Shapes/Complex/Source/KGRodBuilder.cc)|[`vertex`](#kgrodvertex)|*`KGRodVertex`*|`axial_mesh_count`<br>`longitudinal_mesh_count`<br>`name`<br>`radius`|*`int`*<br>*`int`*<br>*`string`*<br>*`double`*|

##### KGRodVertex

|element name|source files|child elements|child types|attributes|attribute types|
|-----|-----|-----|-----|-----|-----|
|<a name="kgrodvertex">`vertex`</a>|[*KGRodBuilder.cc*](https://github.com/KATRIN-Experiment/Kassiopeia/tree/main/KGeoBag/Source/Bindings/Shapes/Complex/Source/KGRodBuilder.cc)|—    |—    |`x`<br>`y`<br>`z`|*`double`*<br>*`double`*<br>*`double`*|

### KGWrappedSurface\<KGRod\>

|element name|source files|child elements|child types|attributes|attribute types|
|-----|-----|-----|-----|-----|-----|
|<a name="kgwrappedsurface\<kgrod\>">`rod_surface`</a>|[*KGRodBuilder.cc*](https://github.com/KATRIN-Experiment/Kassiopeia/tree/main/KGeoBag/Source/Bindings/Shapes/Complex/Source/KGRodBuilder.cc)|[`rod`](#kgrod)|*`KGRod`*|`name`|*`string`*|

### KGRotatedArcSegmentSpace

|element name|source files|child elements|child types|attributes|attribute types|
|-----|-----|-----|-----|-----|-----|
|<a name="kgrotatedarcsegmentspace">`rotated_arc_segment_space`</a>|[*KGRotatedArcSegmentSpaceBuilder.cc*](https://github.com/KATRIN-Experiment/Kassiopeia/tree/main/KGeoBag/Source/Bindings/Shapes/RotatedVolumes/Source/KGRotatedArcSegmentSpaceBuilder.cc)|[`arc_segment`](#kgplanararcsegment)|*`KGPlanarArcSegment`*|`flattened_mesh_count`<br>`flattened_mesh_power`<br>`name`<br>`rotated_mesh_count`|*`unsigned int`*<br>*`double`*<br>*`string`*<br>*`unsigned int`*|

### KGRotatedArcSegmentSurface

|element name|source files|child elements|child types|attributes|attribute types|
|-----|-----|-----|-----|-----|-----|
|<a name="kgrotatedarcsegmentsurface">`rotated_arc_segment_surface`</a>|[*KGRotatedArcSegmentSurfaceBuilder.cc*](https://github.com/KATRIN-Experiment/Kassiopeia/tree/main/KGeoBag/Source/Bindings/Shapes/RotatedAreas/Source/KGRotatedArcSegmentSurfaceBuilder.cc)|[`arc_segment`](#kgplanararcsegment)|*`KGPlanarArcSegment`*|`name`<br>`rotated_mesh_count`|*`string`*<br>*`unsigned int`*|

### KGRotatedCircleSpace

|element name|source files|child elements|child types|attributes|attribute types|
|-----|-----|-----|-----|-----|-----|
|<a name="kgrotatedcirclespace">`rotated_circle_space`</a>|[*KGRotatedCircleSpaceBuilder.cc*](https://github.com/KATRIN-Experiment/Kassiopeia/tree/main/KGeoBag/Source/Bindings/Shapes/RotatedVolumes/Source/KGRotatedCircleSpaceBuilder.cc)|[`circle`](#kgplanarcircle)|*`KGPlanarCircle`*|`name`<br>`rotated_mesh_count`|*`string`*<br>*`unsigned int`*|

### KGRotatedCircleSurface

|element name|source files|child elements|child types|attributes|attribute types|
|-----|-----|-----|-----|-----|-----|
|<a name="kgrotatedcirclesurface">`rotated_circle_surface`</a>|[*KGRotatedCircleSurfaceBuilder.cc*](https://github.com/KATRIN-Experiment/Kassiopeia/tree/main/KGeoBag/Source/Bindings/Shapes/RotatedAreas/Source/KGRotatedCircleSurfaceBuilder.cc)|[`circle`](#kgplanarcircle)|*`KGPlanarCircle`*|`name`<br>`rotated_mesh_count`|*`string`*<br>*`unsigned int`*|

### KGRotatedLineSegmentSpace

|element name|source files|child elements|child types|attributes|attribute types|
|-----|-----|-----|-----|-----|-----|
|<a name="kgrotatedlinesegmentspace">`rotated_line_segment_space`</a>|[*KGRotatedLineSegmentSpaceBuilder.cc*](https://github.com/KATRIN-Experiment/Kassiopeia/tree/main/KGeoBag/Source/Bindings/Shapes/RotatedVolumes/Source/KGRotatedLineSegmentSpaceBuilder.cc)|[`line_segment`](#kgplanarlinesegment)|*`KGPlanarLineSegment`*|`flattened_mesh_count`<br>`flattened_mesh_power`<br>`name`<br>`rotated_mesh_count`|*`unsigned int`*<br>*`double`*<br>*`string`*<br>*`unsigned int`*|

### KGRotatedLineSegmentSurface

|element name|source files|child elements|child types|attributes|attribute types|
|-----|-----|-----|-----|-----|-----|
|<a name="kgrotatedlinesegmentsurface">`rotated_line_segment_surface`</a>|[*KGRotatedLineSegmentSurfaceBuilder.cc*](https://github.com/KATRIN-Experiment/Kassiopeia/tree/main/KGeoBag/Source/Bindings/Shapes/RotatedAreas/Source/KGRotatedLineSegmentSurfaceBuilder.cc)|[`line_segment`](#kgplanarlinesegment)|*`KGPlanarLineSegment`*|`name`<br>`rotated_mesh_count`|*`string`*<br>*`unsigned int`*|

### KGRotatedPolyLineSpace

|element name|source files|child elements|child types|attributes|attribute types|
|-----|-----|-----|-----|-----|-----|
|<a name="kgrotatedpolylinespace">`rotated_poly_line_space`</a>|[*KGRotatedPolyLineSpaceBuilder.cc*](https://github.com/KATRIN-Experiment/Kassiopeia/tree/main/KGeoBag/Source/Bindings/Shapes/RotatedVolumes/Source/KGRotatedPolyLineSpaceBuilder.cc)|[`poly_line`](#kgplanarpolyline)|*`KGPlanarPolyLine`*|`flattened_mesh_count`<br>`flattened_mesh_power`<br>`name`<br>`rotated_mesh_count`|*`unsigned int`*<br>*`double`*<br>*`string`*<br>*`unsigned int`*|

### KGRotatedPolyLineSurface

|element name|source files|child elements|child types|attributes|attribute types|
|-----|-----|-----|-----|-----|-----|
|<a name="kgrotatedpolylinesurface">`rotated_poly_line_surface`</a>|[*KGRotatedPolyLineSurfaceBuilder.cc*](https://github.com/KATRIN-Experiment/Kassiopeia/tree/main/KGeoBag/Source/Bindings/Shapes/RotatedAreas/Source/KGRotatedPolyLineSurfaceBuilder.cc)|[`poly_line`](#kgplanarpolyline)|*`KGPlanarPolyLine`*|`name`<br>`rotated_mesh_count`|*`string`*<br>*`unsigned int`*|

### KGRotatedPolyLoopSpace

|element name|source files|child elements|child types|attributes|attribute types|
|-----|-----|-----|-----|-----|-----|
|<a name="kgrotatedpolyloopspace">`rotated_poly_loop_space`</a>|[*KGRotatedPolyLoopSpaceBuilder.cc*](https://github.com/KATRIN-Experiment/Kassiopeia/tree/main/KGeoBag/Source/Bindings/Shapes/RotatedVolumes/Source/KGRotatedPolyLoopSpaceBuilder.cc)|[`poly_loop`](#kgplanarpolyloop)|*`KGPlanarPolyLoop`*|`name`<br>`rotated_mesh_count`|*`string`*<br>*`unsigned int`*|

### KGRotatedPolyLoopSurface

|element name|source files|child elements|child types|attributes|attribute types|
|-----|-----|-----|-----|-----|-----|
|<a name="kgrotatedpolyloopsurface">`rotated_poly_loop_surface`</a>|[*KGRotatedPolyLoopSurfaceBuilder.cc*](https://github.com/KATRIN-Experiment/Kassiopeia/tree/main/KGeoBag/Source/Bindings/Shapes/RotatedAreas/Source/KGRotatedPolyLoopSurfaceBuilder.cc)|[`poly_loop`](#kgplanarpolyloop)|*`KGPlanarPolyLoop`*|`name`<br>`rotated_mesh_count`|*`string`*<br>*`unsigned int`*|

### KGWrappedSpace\<KGRotatedObject\>

|element name|source files|child elements|child types|attributes|attribute types|
|-----|-----|-----|-----|-----|-----|
|<a name="kgwrappedspace\<kgrotatedobject\>">`rotated_space`</a>|[*KGRotatedObjectBuilder.cc*](https://github.com/KATRIN-Experiment/Kassiopeia/tree/main/KGeoBag/Source/Bindings/Shapes/Complex/Source/KGRotatedObjectBuilder.cc)|[`rotated_object`](#kgrotatedobject)|*`KGRotatedObject`*|`name`|*`string`*|

#### KGRotatedObject

|element name|source files|child elements|child types|attributes|attribute types|
|-----|-----|-----|-----|-----|-----|
|<a name="kgrotatedobject">`rotated_object`</a>|[*KGRotatedObjectBuilder.cc*](https://github.com/KATRIN-Experiment/Kassiopeia/tree/main/KGeoBag/Source/Bindings/Shapes/Complex/Source/KGRotatedObjectBuilder.cc)|[`arc`](#kgrotatedobjectarc)<br>[`line`](#kgrotatedobjectline)|*`KGRotatedObjectArc`*<br>*`KGRotatedObjectLine`*|`longitudinal_mesh_count`<br>`longitudinal_mesh_count_end`<br>`longitudinal_mesh_count_start`<br>`longitudinal_mesh_power`|*`int`*<br>*`int`*<br>*`int`*<br>*`double`*|

##### KGRotatedObjectArc

|element name|source files|child elements|child types|attributes|attribute types|
|-----|-----|-----|-----|-----|-----|
|<a name="kgrotatedobjectarc">`arc`</a>|[*KGRotatedObjectBuilder.cc*](https://github.com/KATRIN-Experiment/Kassiopeia/tree/main/KGeoBag/Source/Bindings/Shapes/Complex/Source/KGRotatedObjectBuilder.cc)|—    |—    |`positive_orientation`<br>`r1`<br>`r2`<br>`radius`<br>`z1`<br>`z2`|*`bool`*<br>*`double`*<br>*`double`*<br>*`double`*<br>*`double`*<br>*`double`*|

##### KGRotatedObjectLine

|element name|source files|child elements|child types|attributes|attribute types|
|-----|-----|-----|-----|-----|-----|
|<a name="kgrotatedobjectline">`line`</a>|[*KGRotatedObjectBuilder.cc*](https://github.com/KATRIN-Experiment/Kassiopeia/tree/main/KGeoBag/Source/Bindings/Shapes/Complex/Source/KGRotatedObjectBuilder.cc)|—    |—    |`r1`<br>`r2`<br>`z1`<br>`z2`|*`double`*<br>*`double`*<br>*`double`*<br>*`double`*|

### KGWrappedSurface\<KGRotatedObject\>

|element name|source files|child elements|child types|attributes|attribute types|
|-----|-----|-----|-----|-----|-----|
|<a name="kgwrappedsurface\<kgrotatedobject\>">`rotated_surface`</a>|[*KGRotatedObjectBuilder.cc*](https://github.com/KATRIN-Experiment/Kassiopeia/tree/main/KGeoBag/Source/Bindings/Shapes/Complex/Source/KGRotatedObjectBuilder.cc)|[`rotated_object`](#kgrotatedobject)|*`KGRotatedObject`*|`name`|*`string`*|

### KGShellArcSegmentSurface

|element name|source files|child elements|child types|attributes|attribute types|
|-----|-----|-----|-----|-----|-----|
|<a name="kgshellarcsegmentsurface">`shell_arc_segment_surface`</a>|[*KGShellArcSegmentSurfaceBuilder.cc*](https://github.com/KATRIN-Experiment/Kassiopeia/tree/main/KGeoBag/Source/Bindings/Shapes/ShellAreas/Source/KGShellArcSegmentSurfaceBuilder.cc)|[`arc_segment`](#kgplanararcsegment)|*`KGPlanarArcSegment`*|`angle_start`<br>`angle_stop`<br>`name`<br>`shell_mesh_count`<br>`shell_mesh_power`|*`double`*<br>*`double`*<br>*`string`*<br>*`unsigned int`*<br>*`double`*|

### KGShellCircleSurface

|element name|source files|child elements|child types|attributes|attribute types|
|-----|-----|-----|-----|-----|-----|
|<a name="kgshellcirclesurface">`shell_circle_surface`</a>|[*KGShellCircleSurfaceBuilder.cc*](https://github.com/KATRIN-Experiment/Kassiopeia/tree/main/KGeoBag/Source/Bindings/Shapes/ShellAreas/Source/KGShellCircleSurfaceBuilder.cc)|[`circle`](#kgplanarcircle)|*`KGPlanarCircle`*|`angle_start`<br>`angle_stop`<br>`name`<br>`shell_mesh_count`<br>`shell_mesh_power`|*`double`*<br>*`double`*<br>*`string`*<br>*`unsigned int`*<br>*`double`*|

### KGShellLineSegmentSurface

|element name|source files|child elements|child types|attributes|attribute types|
|-----|-----|-----|-----|-----|-----|
|<a name="kgshelllinesegmentsurface">`shell_line_segment_surface`</a>|[*KGShellLineSegmentSurfaceBuilder.cc*](https://github.com/KATRIN-Experiment/Kassiopeia/tree/main/KGeoBag/Source/Bindings/Shapes/ShellAreas/Source/KGShellLineSegmentSurfaceBuilder.cc)|[`line_segment`](#kgplanarlinesegment)|*`KGPlanarLineSegment`*|`angle_start`<br>`angle_stop`<br>`name`<br>`shell_mesh_count`<br>`shell_mesh_power`|*`double`*<br>*`double`*<br>*`string`*<br>*`unsigned int`*<br>*`double`*|

### KGShellPolyLineSurface

|element name|source files|child elements|child types|attributes|attribute types|
|-----|-----|-----|-----|-----|-----|
|<a name="kgshellpolylinesurface">`shell_poly_line_surface`</a>|[*KGShellPolyLineSurfaceBuilder.cc*](https://github.com/KATRIN-Experiment/Kassiopeia/tree/main/KGeoBag/Source/Bindings/Shapes/ShellAreas/Source/KGShellPolyLineSurfaceBuilder.cc)|[`poly_line`](#kgplanarpolyline)|*`KGPlanarPolyLine`*|`angle_start`<br>`angle_stop`<br>`name`<br>`shell_mesh_count`<br>`shell_mesh_power`|*`double`*<br>*`double`*<br>*`string`*<br>*`unsigned int`*<br>*`double`*|

### KGShellPolyLoopSurface

|element name|source files|child elements|child types|attributes|attribute types|
|-----|-----|-----|-----|-----|-----|
|<a name="kgshellpolyloopsurface">`shell_poly_loop_surface`</a>|[*KGShellPolyLoopSurfaceBuilder.cc*](https://github.com/KATRIN-Experiment/Kassiopeia/tree/main/KGeoBag/Source/Bindings/Shapes/ShellAreas/Source/KGShellPolyLoopSurfaceBuilder.cc)|[`poly_loop`](#kgplanarpolyloop)|*`KGPlanarPolyLoop`*|`angle_start`<br>`angle_stop`<br>`name`<br>`shell_mesh_count`<br>`shell_mesh_power`|*`double`*<br>*`double`*<br>*`string`*<br>*`unsigned int`*<br>*`double`*|

### KGSpace

|element name|source files|child elements|child types|attributes|attribute types|
|-----|-----|-----|-----|-----|-----|
|<a name="kgspace">`space`</a>|[*KGSpaceBuilder.cc*](https://github.com/KATRIN-Experiment/Kassiopeia/tree/main/KGeoBag/Source/Bindings/Core/Source/KGSpaceBuilder.cc)|[`space`](#kgspace)<br>[`surface`](#kgsurface)<br>[`transformation`](#ktransformation)|*`KGSpace`*<br>*`KGSurface`*<br>*`KTransformation`*|`name`<br>`node`<br>`tree`|*`string`*<br>*`string`*<br>*`string`*|

#### KGSurface

|element name|source files|child elements|child types|attributes|attribute types|
|-----|-----|-----|-----|-----|-----|
|<a name="kgsurface">`surface`</a>|[*KGSpaceBuilder.cc*](https://github.com/KATRIN-Experiment/Kassiopeia/tree/main/KGeoBag/Source/Bindings/Core/Source/KGSpaceBuilder.cc)<br>[*KGSurfaceBuilder.cc*](https://github.com/KATRIN-Experiment/Kassiopeia/tree/main/KGeoBag/Source/Bindings/Core/Source/KGSurfaceBuilder.cc)|[`transformation`](#ktransformation)|*`KTransformation`*|`name`<br>`node`|*`string`*<br>*`string`*|

##### KTransformation

|element name|source files|child elements|child types|attributes|attribute types|
|-----|-----|-----|-----|-----|-----|
|<a name="ktransformation">`transformation`</a>|[*KGSpaceBuilder.cc*](https://github.com/KATRIN-Experiment/Kassiopeia/tree/main/KGeoBag/Source/Bindings/Core/Source/KGSpaceBuilder.cc)<br>[*KGSurfaceBuilder.cc*](https://github.com/KATRIN-Experiment/Kassiopeia/tree/main/KGeoBag/Source/Bindings/Core/Source/KGSurfaceBuilder.cc)|—    |—    |`d`<br>`displacement`<br>`r_aa`<br>`r_eu`<br>`rotation_axis_angle`<br>`rotation_euler`|*`KThreeVector`*<br>*`KThreeVector`*<br>*`KThreeVector`*<br>*`KThreeVector`*<br>*`KThreeVector`*<br>*`KThreeVector`*|

### KGWrappedSpace\<KGStlFile\>

|element name|source files|child elements|child types|attributes|attribute types|
|-----|-----|-----|-----|-----|-----|
|<a name="kgwrappedspace\<kgstlfile\>">`stl_file_space`</a>|[*KGStlFileBuilder.cc*](https://github.com/KATRIN-Experiment/Kassiopeia/tree/main/KGeoBag/Source/Bindings/Shapes/External/Source/KGStlFileBuilder.cc)|[`stl_file`](#kgstlfile)|*`KGStlFile`*|`name`|*`string`*|

#### KGStlFile

|element name|source files|child elements|child types|attributes|attribute types|
|-----|-----|-----|-----|-----|-----|
|<a name="kgstlfile">`stl_file`</a>|[*KGStlFileBuilder.cc*](https://github.com/KATRIN-Experiment/Kassiopeia/tree/main/KGeoBag/Source/Bindings/Shapes/External/Source/KGStlFileBuilder.cc)|—    |—    |`file`<br>`mesh_count`<br>`path`<br>`scale`<br>`selector`|*`string`*<br>*`int`*<br>*`string`*<br>*`double`*<br>*`string`*|

### KGWrappedSurface\<KGStlFile\>

|element name|source files|child elements|child types|attributes|attribute types|
|-----|-----|-----|-----|-----|-----|
|<a name="kgwrappedsurface\<kgstlfile\>">`stl_file_surface`</a>|[*KGStlFileBuilder.cc*](https://github.com/KATRIN-Experiment/Kassiopeia/tree/main/KGeoBag/Source/Bindings/Shapes/External/Source/KGStlFileBuilder.cc)|[`stl_file`](#kgstlfile)|*`KGStlFile`*|`name`|*`string`*|

### KGTorusSpace

|element name|source files|child elements|child types|attributes|attribute types|
|-----|-----|-----|-----|-----|-----|
|<a name="kgtorusspace">`torus_space`</a>|[*KGTorusSpaceBuilder.cc*](https://github.com/KATRIN-Experiment/Kassiopeia/tree/main/KGeoBag/Source/Bindings/Shapes/RotatedVolumes/Source/KGTorusSpaceBuilder.cc)|—    |—    |`axial_mesh_count`<br>`name`<br>`r`<br>`radius`<br>`toroidal_mesh_count`<br>`z`|*`unsigned int`*<br>*`string`*<br>*`double`*<br>*`double`*<br>*`unsigned int`*<br>*`double`*|

### KGTorusSurface

|element name|source files|child elements|child types|attributes|attribute types|
|-----|-----|-----|-----|-----|-----|
|<a name="kgtorussurface">`torus_surface`</a>|[*KGTorusSurfaceBuilder.cc*](https://github.com/KATRIN-Experiment/Kassiopeia/tree/main/KGeoBag/Source/Bindings/Shapes/RotatedAreas/Source/KGTorusSurfaceBuilder.cc)|—    |—    |`axial_mesh_count`<br>`name`<br>`r`<br>`radius`<br>`toroidal_mesh_count`<br>`z`|*`unsigned int`*<br>*`string`*<br>*`double`*<br>*`double`*<br>*`unsigned int`*<br>*`double`*|

## KGGeometryPrinter

|element name|source files|child elements|child types|attributes|attribute types|
|-----|-----|-----|-----|-----|-----|
|<a name="kggeometryprinter">`geometry_printer`</a>|[*KGGeometryPrinterBuilder.cc*](https://github.com/KATRIN-Experiment/Kassiopeia/tree/main/KGeoBag/Source/Bindings/Visualization/Basic/Source/KGGeometryPrinterBuilder.cc)|—    |—    |`file`<br>`name`<br>`path`<br>`spaces`<br>`surfaces`<br>`write_json`<br>`write_xml`|*`string`*<br>*`string`*<br>*`string`*<br>*`string`*<br>*`string`*<br>*`bool`*<br>*`bool`*|

## KSRoot

|element name|source files|child elements|child types|attributes|attribute types|
|-----|-----|-----|-----|-----|-----|
|<a name="ksroot">`kassiopeia`</a>|[*KSRootBuilder.cxx*](https://github.com/KATRIN-Experiment/Kassiopeia/tree/main/Kassiopeia/Bindings/Simulation/Source/KSRootBuilder.cxx)|[`kess_elastic_elsepa`](#kesselasticelsepa)<br>[`kess_inelastic_bethefano`](#kessinelasticbethefano)<br>[`kess_inelastic_penn`](#kessinelasticpenn)<br>[`kess_surface_interaction`](#kesssurfaceinteraction)<br>[`ks_command_group`](#kscommandgroup)<br>[`ks_command_member`](#kscommandmemberdata)<br>[`ks_component_delta`](#kscomponentdeltadata)<br>[`ks_component_group`](#kscomponentgroup)<br>[`ks_component_integral`](#kscomponentintegraldata)<br>[`ks_component_math`](#kscomponentmathdata)<br>[`ks_component_maximum`](#kscomponentmaximumdata)<br>[`ks_component_maximum_at`](#kscomponentmaximumatdata)<br>[`ks_component_member`](#kscomponentmemberdata)<br>[`ks_component_minimum`](#kscomponentminimumdata)<br>[`ks_component_minimum_at`](#kscomponentminimumatdata)<br>[`ks_root_electric_field`](#ksrootelectricfield)<br>[`ks_root_event_modifier`](#ksrooteventmodifier)<br>[`ks_root_generator`](#ksrootgenerator)<br>[`ks_root_magnetic_field`](#ksrootmagneticfield)<br>[`ks_root_run_modifier`](#ksrootrunmodifier)<br>[`ks_root_space_interaction`](#ksrootspaceinteraction)<br>[`ks_root_space_navigator`](#ksrootspacenavigator)<br>[`ks_root_step_modifier`](#ksrootstepmodifier)<br>[`ks_root_surface_interaction`](#ksrootsurfaceinteraction)<br>[`ks_root_surface_navigator`](#ksrootsurfacenavigator)<br>[`ks_root_terminator`](#ksrootterminator)<br>[`ks_root_track_modifier`](#ksroottrackmodifier)<br>[`ks_root_trajectory`](#ksroottrajectory)<br>[`ks_root_writer`](#ksrootwriter)<br>[`ks_simulation`](#kssimulation)<br>[`ksfield_electric_constant`](#kelectrostaticconstantfield)<br>[`ksfield_electric_induced_azi`](#kinducedazimuthalelectricfield)<br>[`ksfield_electric_potentialmap`](#kelectrostaticpotentialmap)<br>[`ksfield_electric_potentialmap_calculator`](#kelectrostaticpotentialmapcalculator)<br>[`ksfield_electric_quadrupole`](#kelectricquadrupolefield)<br>[`ksfield_electric_ramped`](#krampedelectricfield)<br>[`ksfield_electric_ramped_2fields`](#krampedelectric2field)<br>[`ksfield_electromagnet`](#kgstaticelectromagnetfield)<br>[`ksfield_electrostatic`](#kgelectrostaticboundaryfield)<br>[`ksfield_magnetic_constant`](#kmagnetostaticconstantfield)<br>[`ksfield_magnetic_dipole`](#kmagneticdipolefield)<br>[`ksfield_magnetic_fieldmap`](#kmagnetostaticfieldmap)<br>[`ksfield_magnetic_fieldmap_calculator`](#kmagnetostaticfieldmapcalculator)<br>[`ksfield_magnetic_ramped`](#krampedmagneticfield)<br>[`ksfield_magnetic_super_position`](#kmagneticsuperpositionfield)<br>[`ksgen_direction_spherical_composite`](#ksgendirectionsphericalcomposite)<br>[`ksgen_direction_spherical_magnetic_field`](#ksgendirectionsphericalmagneticfield)<br>[`ksgen_direction_surface_composite`](#ksgendirectionsurfacecomposite)<br>[`ksgen_energy_beta_decay`](#ksgenenergybetadecay)<br>[`ksgen_energy_beta_recoil`](#ksgenenergybetarecoil)<br>[`ksgen_energy_composite`](#ksgenenergycomposite)<br>[`ksgen_energy_krypton_event`](#ksgenenergykryptonevent)<br>[`ksgen_energy_lead_event`](#ksgenenergyleadevent)<br>[`ksgen_energy_radon_event`](#ksgenenergyradonevent)<br>[`ksgen_energy_rydberg`](#ksgenenergyrydberg)<br>[`ksgen_generator_composite`](#ksgengeneratorcomposite)<br>[`ksgen_generator_file`](#ksgengeneratortextfile)<br>[`ksgen_generator_simulation`](#ksgengeneratorsimulation)<br>[`ksgen_l_composite`](#ksgenlcomposite)<br>[`ksgen_l_statistical`](#ksgenlstatistical)<br>[`ksgen_l_uniform_max_n`](#ksgenluniformmaxn)<br>[`ksgen_momentum_rectangular_composite`](#ksgenmomentumrectangularcomposite)<br>[`ksgen_n_composite`](#ksgenncomposite)<br>[`ksgen_position_cylindrical_composite`](#ksgenpositioncylindricalcomposite)<br>[`ksgen_position_flux_tube`](#ksgenpositionfluxtube)<br>[`ksgen_position_frustrum_composite`](#ksgenpositionfrustrumcomposite)<br>[`ksgen_position_homogeneous_flux_tube`](#ksgenpositionhomogeneousfluxtube)<br>[`ksgen_position_mask`](#ksgenpositionmask)<br>[`ksgen_position_mesh_surface_random`](#ksgenpositionmeshsurfacerandom)<br>[`ksgen_position_rectangular_composite`](#ksgenpositionrectangularcomposite)<br>[`ksgen_position_space_random`](#ksgenpositionspacerandom)<br>[`ksgen_position_spherical_composite`](#ksgenpositionsphericalcomposite)<br>[`ksgen_position_surface_adjustment_step`](#ksgenpositionsurfaceadjustmentstep)<br>[`ksgen_position_surface_random`](#ksgenpositionsurfacerandom)<br>[`ksgen_spin_composite`](#ksgenspincomposite)<br>[`ksgen_spin_composite_relative`](#ksgenspinrelativecomposite)<br>[`ksgen_time_composite`](#ksgentimecomposite)<br>[`ksgen_value_angle_cosine`](#ksgenvalueanglecosine)<br>[`ksgen_value_angle_spherical`](#ksgenvalueanglespherical)<br>[`ksgen_value_boltzmann`](#ksgenvalueboltzmann)<br>[`ksgen_value_fix`](#ksgenvaluefix)<br>[`ksgen_value_formula`](#ksgenvalueformula)<br>[`ksgen_value_gauss`](#ksgenvaluegauss)<br>[`ksgen_value_generalized_gauss`](#ksgenvaluegeneralizedgauss)<br>[`ksgen_value_histogram`](#ksgenvaluehistogram)<br>[`ksgen_value_list`](#ksgenvaluelist)<br>[`ksgen_value_pareto`](#ksgenvaluepareto)<br>[`ksgen_value_radius_cylindrical`](#ksgenvalueradiuscylindrical)<br>[`ksgen_value_radius_fraction`](#ksgenvalueradiusfraction)<br>[`ksgen_value_radius_spherical`](#ksgenvalueradiusspherical)<br>[`ksgen_value_set`](#ksgenvalueset)<br>[`ksgen_value_uniform`](#ksgenvalueuniform)<br>[`ksgen_value_z_frustrum`](#ksgenvaluezfrustrum)<br>[`ksgeo_side`](#ksgeoside)<br>[`ksgeo_space`](#ksgeospace)<br>[`ksgeo_surface`](#ksgeosurface)<br>[`ksint_calculator_constant`](#ksintcalculatorconstant)<br>[`ksint_calculator_ion`](#ksintcalculatorion)<br>[`ksint_decay`](#ksintdecay)<br>[`ksint_decay_calculator_death_const_rate`](#ksintdecaycalculatordeathconstrate)<br>[`ksint_decay_calculator_ferenc_bbr_transition`](#ksintdecaycalculatorferencbbrtransition)<br>[`ksint_decay_calculator_ferenc_ionisation`](#ksintdecaycalculatorferencionisation)<br>[`ksint_decay_calculator_ferenc_spontaneous`](#ksintdecaycalculatorferencspontaneous)<br>[`ksint_decay_calculator_glukhov_deexcitation`](#ksintdecaycalculatorglukhovdeexcitation)<br>[`ksint_decay_calculator_glukhov_excitation`](#ksintdecaycalculatorglukhovexcitation)<br>[`ksint_decay_calculator_glukhov_ionisation`](#ksintdecaycalculatorglukhovionisation)<br>[`ksint_decay_calculator_glukhov_spontaneous`](#ksintdecaycalculatorglukhovspontaneous)<br>[`ksint_density_constant`](#ksintdensityconstant)<br>[`ksint_scattering`](#ksintscattering)<br>[`ksint_spin_flip`](#ksintspinflip)<br>[`ksint_spin_flip_pulse`](#ksintspinflippulse)<br>[`ksint_spin_rotate_y_pulse`](#ksintspinrotateypulse)<br>[`ksint_surface_UCN`](#ksintsurfaceucn)<br>[`ksint_surface_diffuse`](#ksintsurfacediffuse)<br>[`ksint_surface_multiplication`](#ksintsurfacemultiplication)<br>[`ksint_surface_specular`](#ksintsurfacespecular)<br>[`ksint_surface_spin_flip`](#ksintsurfacespinflip)<br>[`ksmod_dynamic_enhancement`](#ksmoddynamicenhancement)<br>[`ksmod_event_report`](#ksmodeventreport)<br>[`ksmod_split_on_turn`](#ksmodsplitonturn)<br>[`ksnav_meshed_space`](#ksnavmeshedspace)<br>[`ksnav_space`](#ksnavspace)<br>[`ksnav_surface`](#ksnavsurface)<br>[`ksterm_death`](#kstermdeath)<br>[`ksterm_magnetron`](#kstermmagnetron)<br>[`ksterm_max_energy`](#kstermmaxenergy)<br>[`ksterm_max_length`](#kstermmaxlength)<br>[`ksterm_max_long_energy`](#kstermmaxlongenergy)<br>[`ksterm_max_r`](#kstermmaxr)<br>[`ksterm_max_step_time`](#kstermmaxsteptime)<br>[`ksterm_max_steps`](#kstermmaxsteps)<br>[`ksterm_max_time`](#kstermmaxtime)<br>[`ksterm_max_total_time`](#kstermmaxtotaltime)<br>[`ksterm_max_z`](#kstermmaxz)<br>[`ksterm_min_distance`](#kstermmindistance)<br>[`ksterm_min_energy`](#kstermminenergy)<br>[`ksterm_min_long_energy`](#kstermminlongenergy)<br>[`ksterm_min_r`](#kstermminr)<br>[`ksterm_min_z`](#kstermminz)<br>[`ksterm_output`](#kstermoutputdata)<br>[`ksterm_secondaries`](#kstermsecondaries)<br>[`ksterm_stepsize`](#kstermstepsize)<br>[`ksterm_trapped`](#kstermtrapped)<br>[`ksterm_zh_radius`](#kstermzhradius)<br>[`kstraj_control_B_change`](#kstrajcontrolbchange)<br>[`kstraj_control_cyclotron`](#kstrajcontrolcyclotron)<br>[`kstraj_control_energy`](#kstrajcontrolenergy)<br>[`kstraj_control_length`](#kstrajcontrollength)<br>[`kstraj_control_m_dot`](#kstrajcontrolmdot)<br>[`kstraj_control_magnetic_moment`](#kstrajcontrolmagneticmoment)<br>[`kstraj_control_momentum_numerical_error`](#kstrajcontrolmomentumnumericalerror)<br>[`kstraj_control_position_numerical_error`](#kstrajcontrolpositionnumericalerror)<br>[`kstraj_control_spin_precession`](#kstrajcontrolspinprecession)<br>[`kstraj_control_time`](#kstrajcontroltime)<br>[`kstraj_integrator_rk54`](#kstrajintegratorrk54)<br>[`kstraj_integrator_rk65`](#kstrajintegratorrk65)<br>[`kstraj_integrator_rk8`](#kstrajintegratorrk8)<br>[`kstraj_integrator_rk86`](#kstrajintegratorrk86)<br>[`kstraj_integrator_rk87`](#kstrajintegratorrk87)<br>[`kstraj_integrator_rkdp54`](#kstrajintegratorrkdp54)<br>[`kstraj_integrator_rkdp853`](#kstrajintegratorrkdp853)<br>[`kstraj_integrator_sym4`](#kstrajintegratorsym4)<br>[`kstraj_interpolator_crk`](#kstrajinterpolatorcontinuousrungekutta)<br>[`kstraj_interpolator_fast`](#kstrajinterpolatorfast)<br>[`kstraj_interpolator_hermite`](#kstrajinterpolatorhermite)<br>[`kstraj_term_constant_force_propagation`](#kstrajtermconstantforcepropagation)<br>[`kstraj_term_drift`](#kstrajtermdrift)<br>[`kstraj_term_gravity`](#kstrajtermgravity)<br>[`kstraj_term_gyration`](#kstrajtermgyration)<br>[`kstraj_term_propagation`](#kstrajtermpropagation)<br>[`kstraj_term_synchrotron`](#kstrajtermsynchrotron)<br>[`kstraj_trajectory_adiabatic`](#kstrajtrajectoryadiabatic)<br>[`kstraj_trajectory_adiabatic_spin`](#kstrajtrajectoryadiabaticspin)<br>[`kstraj_trajectory_electric`](#kstrajtrajectoryelectric)<br>[`kstraj_trajectory_exact`](#kstrajtrajectoryexact)<br>[`kstraj_trajectory_exact_spin`](#kstrajtrajectoryexactspin)<br>[`kstraj_trajectory_exact_trapped`](#kstrajtrajectoryexacttrapped)<br>[`kstraj_trajectory_linear`](#kstrajtrajectorylinear)<br>[`kstraj_trajectory_magnetic`](#kstrajtrajectorymagnetic)<br>[`kswrite_ascii`](#kswriteascii)<br>[`kswrite_root`](#kswriteroot)<br>[`kswrite_root_condition_output`](#kswriterootconditionoutputdata)<br>[`kswrite_root_condition_periodic`](#kswriterootconditionperiodicdata)<br>[`kswrite_root_condition_step`](#kswriterootconditionstepdata)<br>[`kswrite_root_condition_terminator`](#kswriterootconditionterminatordata)<br>[`kswrite_vtk`](#kswritevtk)<br>[`output`](#kscomponentmemberdata)<br>[`output_delta`](#kscomponentdeltadata)<br>[`output_group`](#kscomponentgroup)<br>[`output_integral`](#kscomponentintegraldata)<br>[`output_math`](#kscomponentmathdata)<br>[`output_maximum`](#kscomponentmaximumdata)<br>[`output_maximum_at`](#kscomponentmaximumatdata)<br>[`output_minimum`](#kscomponentminimumdata)<br>[`output_minimum_at`](#kscomponentminimumatdata)|*`KESSElasticElsepa`*<br>*`KESSInelasticBetheFano`*<br>*`KESSInelasticPenn`*<br>*`KESSSurfaceInteraction`*<br>*`KSCommandGroup`*<br>*`KSCommandMemberData`*<br>*`KSComponentDeltaData`*<br>*`KSComponentGroup`*<br>*`KSComponentIntegralData`*<br>*`KSComponentMathData`*<br>*`KSComponentMaximumData`*<br>*`KSComponentMaximumAtData`*<br>*`KSComponentMemberData`*<br>*`KSComponentMinimumData`*<br>*`KSComponentMinimumAtData`*<br>*`KSRootElectricField`*<br>*`KSRootEventModifier`*<br>*`KSRootGenerator`*<br>*`KSRootMagneticField`*<br>*`KSRootRunModifier`*<br>*`KSRootSpaceInteraction`*<br>*`KSRootSpaceNavigator`*<br>*`KSRootStepModifier`*<br>*`KSRootSurfaceInteraction`*<br>*`KSRootSurfaceNavigator`*<br>*`KSRootTerminator`*<br>*`KSRootTrackModifier`*<br>*`KSRootTrajectory`*<br>*`KSRootWriter`*<br>*`KSSimulation`*<br>*`KElectrostaticConstantField`*<br>*`KInducedAzimuthalElectricField`*<br>*`KElectrostaticPotentialmap`*<br>*`KElectrostaticPotentialmapCalculator`*<br>*`KElectricQuadrupoleField`*<br>*`KRampedElectricField`*<br>*`KRampedElectric2Field`*<br>*`KGStaticElectromagnetField`*<br>*`KGElectrostaticBoundaryField`*<br>*`KMagnetostaticConstantField`*<br>*`KMagneticDipoleField`*<br>*`KMagnetostaticFieldmap`*<br>*`KMagnetostaticFieldmapCalculator`*<br>*`KRampedMagneticField`*<br>*`KMagneticSuperpositionField`*<br>*`KSGenDirectionSphericalComposite`*<br>*`KSGenDirectionSphericalMagneticField`*<br>*`KSGenDirectionSurfaceComposite`*<br>*`KSGenEnergyBetaDecay`*<br>*`KSGenEnergyBetaRecoil`*<br>*`KSGenEnergyComposite`*<br>*`KSGenEnergyKryptonEvent`*<br>*`KSGenEnergyLeadEvent`*<br>*`KSGenEnergyRadonEvent`*<br>*`KSGenEnergyRydberg`*<br>*`KSGenGeneratorComposite`*<br>*`KSGenGeneratorTextFile`*<br>*`KSGenGeneratorSimulation`*<br>*`KSGenLComposite`*<br>*`KSGenLStatistical`*<br>*`KSGenLUniformMaxN`*<br>*`KSGenMomentumRectangularComposite`*<br>*`KSGenNComposite`*<br>*`KSGenPositionCylindricalComposite`*<br>*`KSGenPositionFluxTube`*<br>*`KSGenPositionFrustrumComposite`*<br>*`KSGenPositionHomogeneousFluxTube`*<br>*`KSGenPositionMask`*<br>*`KSGenPositionMeshSurfaceRandom`*<br>*`KSGenPositionRectangularComposite`*<br>*`KSGenPositionSpaceRandom`*<br>*`KSGenPositionSphericalComposite`*<br>*`KSGenPositionSurfaceAdjustmentStep`*<br>*`KSGenPositionSurfaceRandom`*<br>*`KSGenSpinComposite`*<br>*`KSGenSpinRelativeComposite`*<br>*`KSGenTimeComposite`*<br>*`KSGenValueAngleCosine`*<br>*`KSGenValueAngleSpherical`*<br>*`KSGenValueBoltzmann`*<br>*`KSGenValueFix`*<br>*`KSGenValueFormula`*<br>*`KSGenValueGauss`*<br>*`KSGenValueGeneralizedGauss`*<br>*`KSGenValueHistogram`*<br>*`KSGenValueList`*<br>*`KSGenValuePareto`*<br>*`KSGenValueRadiusCylindrical`*<br>*`KSGenValueRadiusFraction`*<br>*`KSGenValueRadiusSpherical`*<br>*`KSGenValueSet`*<br>*`KSGenValueUniform`*<br>*`KSGenValueZFrustrum`*<br>*`KSGeoSide`*<br>*`KSGeoSpace`*<br>*`KSGeoSurface`*<br>*`KSIntCalculatorConstant`*<br>*`KSIntCalculatorIon`*<br>*`KSIntDecay`*<br>*`KSIntDecayCalculatorDeathConstRate`*<br>*`KSIntDecayCalculatorFerencBBRTransition`*<br>*`KSIntDecayCalculatorFerencIonisation`*<br>*`KSIntDecayCalculatorFerencSpontaneous`*<br>*`KSIntDecayCalculatorGlukhovDeExcitation`*<br>*`KSIntDecayCalculatorGlukhovExcitation`*<br>*`KSIntDecayCalculatorGlukhovIonisation`*<br>*`KSIntDecayCalculatorGlukhovSpontaneous`*<br>*`KSIntDensityConstant`*<br>*`KSIntScattering`*<br>*`KSIntSpinFlip`*<br>*`KSIntSpinFlipPulse`*<br>*`KSIntSpinRotateYPulse`*<br>*`KSIntSurfaceUCN`*<br>*`KSIntSurfaceDiffuse`*<br>*`KSIntSurfaceMultiplication`*<br>*`KSIntSurfaceSpecular`*<br>*`KSIntSurfaceSpinFlip`*<br>*`KSModDynamicEnhancement`*<br>*`KSModEventReport`*<br>*`KSModSplitOnTurn`*<br>*`KSNavMeshedSpace`*<br>*`KSNavSpace`*<br>*`KSNavSurface`*<br>*`KSTermDeath`*<br>*`KSTermMagnetron`*<br>*`KSTermMaxEnergy`*<br>*`KSTermMaxLength`*<br>*`KSTermMaxLongEnergy`*<br>*`KSTermMaxR`*<br>*`KSTermMaxStepTime`*<br>*`KSTermMaxSteps`*<br>*`KSTermMaxTime`*<br>*`KSTermMaxTotalTime`*<br>*`KSTermMaxZ`*<br>*`KSTermMinDistance`*<br>*`KSTermMinEnergy`*<br>*`KSTermMinLongEnergy`*<br>*`KSTermMinR`*<br>*`KSTermMinZ`*<br>*`KSTermOutputData`*<br>*`KSTermSecondaries`*<br>*`KSTermStepsize`*<br>*`KSTermTrapped`*<br>*`KSTermZHRadius`*<br>*`KSTrajControlBChange`*<br>*`KSTrajControlCyclotron`*<br>*`KSTrajControlEnergy`*<br>*`KSTrajControlLength`*<br>*`KSTrajControlMDot`*<br>*`KSTrajControlMagneticMoment`*<br>*`KSTrajControlMomentumNumericalError`*<br>*`KSTrajControlPositionNumericalError`*<br>*`KSTrajControlSpinPrecession`*<br>*`KSTrajControlTime`*<br>*`KSTrajIntegratorRK54`*<br>*`KSTrajIntegratorRK65`*<br>*`KSTrajIntegratorRK8`*<br>*`KSTrajIntegratorRK86`*<br>*`KSTrajIntegratorRK87`*<br>*`KSTrajIntegratorRKDP54`*<br>*`KSTrajIntegratorRKDP853`*<br>*`KSTrajIntegratorSym4`*<br>*`KSTrajInterpolatorContinuousRungeKutta`*<br>*`KSTrajInterpolatorFast`*<br>*`KSTrajInterpolatorHermite`*<br>*`KSTrajTermConstantForcePropagation`*<br>*`KSTrajTermDrift`*<br>*`KSTrajTermGravity`*<br>*`KSTrajTermGyration`*<br>*`KSTrajTermPropagation`*<br>*`KSTrajTermSynchrotron`*<br>*`KSTrajTrajectoryAdiabatic`*<br>*`KSTrajTrajectoryAdiabaticSpin`*<br>*`KSTrajTrajectoryElectric`*<br>*`KSTrajTrajectoryExact`*<br>*`KSTrajTrajectoryExactSpin`*<br>*`KSTrajTrajectoryExactTrapped`*<br>*`KSTrajTrajectoryLinear`*<br>*`KSTrajTrajectoryMagnetic`*<br>*`KSWriteASCII`*<br>*`KSWriteROOT`*<br>*`KSWriteROOTConditionOutputData`*<br>*`KSWriteROOTConditionPeriodicData`*<br>*`KSWriteROOTConditionStepData`*<br>*`KSWriteROOTConditionTerminatorData`*<br>*`KSWriteVTK`*<br>*`KSComponentMemberData`*<br>*`KSComponentDeltaData`*<br>*`KSComponentGroup`*<br>*`KSComponentIntegralData`*<br>*`KSComponentMathData`*<br>*`KSComponentMaximumData`*<br>*`KSComponentMaximumAtData`*<br>*`KSComponentMinimumData`*<br>*`KSComponentMinimumAtData`*|`random_seed`|*`unsigned int`*|

### KESSElasticElsepa

|element name|source files|child elements|child types|attributes|attribute types|
|-----|-----|-----|-----|-----|-----|
|<a name="kesselasticelsepa">`kess_elastic_elsepa`</a>|[*KESSElasticElsepaBuilder.cxx*](https://github.com/KATRIN-Experiment/Kassiopeia/tree/main/Kassiopeia/Bindings/Interactions/Source/KESSElasticElsepaBuilder.cxx)|—    |—    |`name`|*`string`*|

### KESSInelasticBetheFano

|element name|source files|child elements|child types|attributes|attribute types|
|-----|-----|-----|-----|-----|-----|
|<a name="kessinelasticbethefano">`kess_inelastic_bethefano`</a>|[*KESSInelasticBetheFanoBuilder.cxx*](https://github.com/KATRIN-Experiment/Kassiopeia/tree/main/Kassiopeia/Bindings/Interactions/Source/KESSInelasticBetheFanoBuilder.cxx)|—    |—    |`AugerRelaxation`<br>`PhotoAbsorption`<br>`name`|*`bool`*<br>*`bool`*<br>*`string`*|

### KESSInelasticPenn

|element name|source files|child elements|child types|attributes|attribute types|
|-----|-----|-----|-----|-----|-----|
|<a name="kessinelasticpenn">`kess_inelastic_penn`</a>|[*KESSInelasticPennBuilder.cxx*](https://github.com/KATRIN-Experiment/Kassiopeia/tree/main/Kassiopeia/Bindings/Interactions/Source/KESSInelasticPennBuilder.cxx)|—    |—    |`AugerRelaxation`<br>`PhotoAbsorption`<br>`name`|*`bool`*<br>*`bool`*<br>*`string`*|

### KESSSurfaceInteraction

|element name|source files|child elements|child types|attributes|attribute types|
|-----|-----|-----|-----|-----|-----|
|<a name="kesssurfaceinteraction">`kess_surface_interaction`</a>|[*KESSSurfaceInteractionBuilder.cxx*](https://github.com/KATRIN-Experiment/Kassiopeia/tree/main/Kassiopeia/Bindings/Interactions/Source/KESSSurfaceInteractionBuilder.cxx)|—    |—    |`name`<br>`siliconside`|*`string`*<br>*`string`*|

### KSCommandGroup

|element name|source files|child elements|child types|attributes|attribute types|
|-----|-----|-----|-----|-----|-----|
|<a name="kscommandgroup">`ks_command_group`</a>|[*KSCommandGroupBuilder.cxx*](https://github.com/KATRIN-Experiment/Kassiopeia/tree/main/Kassiopeia/Bindings/Objects/Source/KSCommandGroupBuilder.cxx)|[`command_group`](#kscommandgroup)|*`KSCommandGroup`*|`command`<br>`name`|*`string`*<br>*`string`*|

### KSCommandMemberData

|element name|source files|child elements|child types|attributes|attribute types|
|-----|-----|-----|-----|-----|-----|
|<a name="kscommandmemberdata">`ks_command_member`</a>|[*KSGeoSideBuilder.cxx*](https://github.com/KATRIN-Experiment/Kassiopeia/tree/main/Kassiopeia/Bindings/Geometry/Source/KSGeoSideBuilder.cxx)<br>[*KSGeoSpaceBuilder.cxx*](https://github.com/KATRIN-Experiment/Kassiopeia/tree/main/Kassiopeia/Bindings/Geometry/Source/KSGeoSpaceBuilder.cxx)<br>[*KSGeoSurfaceBuilder.cxx*](https://github.com/KATRIN-Experiment/Kassiopeia/tree/main/Kassiopeia/Bindings/Geometry/Source/KSGeoSurfaceBuilder.cxx)<br>[*KSCommandMemberBuilder.cxx*](https://github.com/KATRIN-Experiment/Kassiopeia/tree/main/Kassiopeia/Bindings/Objects/Source/KSCommandMemberBuilder.cxx)|—    |—    |`child`<br>`field`<br>`name`<br>`parent`|*`string`*<br>*`string`*<br>*`string`*<br>*`string`*|

### KSComponentDeltaData

|element name|source files|child elements|child types|attributes|attribute types|
|-----|-----|-----|-----|-----|-----|
|<a name="kscomponentdeltadata">`ks_component_delta`</a>|[*KSComponentDeltaBuilder.cxx*](https://github.com/KATRIN-Experiment/Kassiopeia/tree/main/Kassiopeia/Bindings/Objects/Source/KSComponentDeltaBuilder.cxx)|—    |—    |`component`<br>`group`<br>`name`<br>`parent`|*`string`*<br>*`string`*<br>*`string`*<br>*`string`*|

### KSComponentGroup

|element name|source files|child elements|child types|attributes|attribute types|
|-----|-----|-----|-----|-----|-----|
|<a name="kscomponentgroup">`ks_component_group`</a>|[*KSComponentGroupBuilder.cxx*](https://github.com/KATRIN-Experiment/Kassiopeia/tree/main/Kassiopeia/Bindings/Objects/Source/KSComponentGroupBuilder.cxx)|—    |—    |—    |—    |

### KSComponentIntegralData

|element name|source files|child elements|child types|attributes|attribute types|
|-----|-----|-----|-----|-----|-----|
|<a name="kscomponentintegraldata">`ks_component_integral`</a>|[*KSComponentIntegralBuilder.cxx*](https://github.com/KATRIN-Experiment/Kassiopeia/tree/main/Kassiopeia/Bindings/Objects/Source/KSComponentIntegralBuilder.cxx)|—    |—    |`component`<br>`group`<br>`name`<br>`parent`|*`string`*<br>*`string`*<br>*`string`*<br>*`string`*|

### KSComponentMathData

|element name|source files|child elements|child types|attributes|attribute types|
|-----|-----|-----|-----|-----|-----|
|<a name="kscomponentmathdata">`ks_component_math`</a>|[*KSComponentMathBuilder.cxx*](https://github.com/KATRIN-Experiment/Kassiopeia/tree/main/Kassiopeia/Bindings/Objects/Source/KSComponentMathBuilder.cxx)|—    |—    |`component`<br>`group`<br>`name`<br>`parent`<br>`term`|*`string`*<br>*`string`*<br>*`string`*<br>*`string`*<br>*`string`*|

### KSComponentMaximumData

|element name|source files|child elements|child types|attributes|attribute types|
|-----|-----|-----|-----|-----|-----|
|<a name="kscomponentmaximumdata">`ks_component_maximum`</a>|[*KSComponentMaximumBuilder.cxx*](https://github.com/KATRIN-Experiment/Kassiopeia/tree/main/Kassiopeia/Bindings/Objects/Source/KSComponentMaximumBuilder.cxx)|—    |—    |`component`<br>`group`<br>`name`<br>`parent`|*`string`*<br>*`string`*<br>*`string`*<br>*`string`*|

### KSComponentMaximumAtData

|element name|source files|child elements|child types|attributes|attribute types|
|-----|-----|-----|-----|-----|-----|
|<a name="kscomponentmaximumatdata">`ks_component_maximum_at`</a>|[*KSComponentMaximumAtBuilder.cxx*](https://github.com/KATRIN-Experiment/Kassiopeia/tree/main/Kassiopeia/Bindings/Objects/Source/KSComponentMaximumAtBuilder.cxx)|—    |—    |`component`<br>`group`<br>`name`<br>`parent`<br>`source`|*`string`*<br>*`string`*<br>*`string`*<br>*`string`*<br>*`string`*|

### KSComponentMemberData

|element name|source files|child elements|child types|attributes|attribute types|
|-----|-----|-----|-----|-----|-----|
|<a name="kscomponentmemberdata">`ks_component_member`</a>|[*KSComponentMemberBuilder.cxx*](https://github.com/KATRIN-Experiment/Kassiopeia/tree/main/Kassiopeia/Bindings/Objects/Source/KSComponentMemberBuilder.cxx)|—    |—    |`field`<br>`name`<br>`parent`|*`string`*<br>*`string`*<br>*`string`*|

### KSComponentMinimumData

|element name|source files|child elements|child types|attributes|attribute types|
|-----|-----|-----|-----|-----|-----|
|<a name="kscomponentminimumdata">`ks_component_minimum`</a>|[*KSComponentMinimumBuilder.cxx*](https://github.com/KATRIN-Experiment/Kassiopeia/tree/main/Kassiopeia/Bindings/Objects/Source/KSComponentMinimumBuilder.cxx)|—    |—    |`component`<br>`group`<br>`name`<br>`parent`|*`string`*<br>*`string`*<br>*`string`*<br>*`string`*|

### KSComponentMinimumAtData

|element name|source files|child elements|child types|attributes|attribute types|
|-----|-----|-----|-----|-----|-----|
|<a name="kscomponentminimumatdata">`ks_component_minimum_at`</a>|[*KSComponentMinimumAtBuilder.cxx*](https://github.com/KATRIN-Experiment/Kassiopeia/tree/main/Kassiopeia/Bindings/Objects/Source/KSComponentMinimumAtBuilder.cxx)|—    |—    |`component`<br>`group`<br>`name`<br>`parent`<br>`source`|*`string`*<br>*`string`*<br>*`string`*<br>*`string`*<br>*`string`*|

### KSRootElectricField

|element name|source files|child elements|child types|attributes|attribute types|
|-----|-----|-----|-----|-----|-----|
|<a name="ksrootelectricfield">`ks_root_electric_field`</a>|[*KSRootElectricFieldBuilder.cxx*](https://github.com/KATRIN-Experiment/Kassiopeia/tree/main/Kassiopeia/Bindings/Simulation/Source/KSRootElectricFieldBuilder.cxx)|—    |—    |`add_electric_field`<br>`name`|*`string`*<br>*`string`*|

### KSRootEventModifier

|element name|source files|child elements|child types|attributes|attribute types|
|-----|-----|-----|-----|-----|-----|
|<a name="ksrooteventmodifier">`ks_root_event_modifier`</a>|[*KSRootEventModifierBuilder.cxx*](https://github.com/KATRIN-Experiment/Kassiopeia/tree/main/Kassiopeia/Bindings/Simulation/Source/KSRootEventModifierBuilder.cxx)|—    |—    |`add_modifier`<br>`name`|*`string`*<br>*`string`*|

### KSRootGenerator

|element name|source files|child elements|child types|attributes|attribute types|
|-----|-----|-----|-----|-----|-----|
|<a name="ksrootgenerator">`ks_root_generator`</a>|[*KSRootGeneratorBuilder.cxx*](https://github.com/KATRIN-Experiment/Kassiopeia/tree/main/Kassiopeia/Bindings/Simulation/Source/KSRootGeneratorBuilder.cxx)|—    |—    |`name`<br>`set_generator`|*`string`*<br>*`string`*|

### KSRootMagneticField

|element name|source files|child elements|child types|attributes|attribute types|
|-----|-----|-----|-----|-----|-----|
|<a name="ksrootmagneticfield">`ks_root_magnetic_field`</a>|[*KSRootMagneticFieldBuilder.cxx*](https://github.com/KATRIN-Experiment/Kassiopeia/tree/main/Kassiopeia/Bindings/Simulation/Source/KSRootMagneticFieldBuilder.cxx)|—    |—    |`add_magnetic_field`<br>`name`|*`string`*<br>*`string`*|

### KSRootRunModifier

|element name|source files|child elements|child types|attributes|attribute types|
|-----|-----|-----|-----|-----|-----|
|<a name="ksrootrunmodifier">`ks_root_run_modifier`</a>|[*KSRootRunModifierBuilder.cxx*](https://github.com/KATRIN-Experiment/Kassiopeia/tree/main/Kassiopeia/Bindings/Simulation/Source/KSRootRunModifierBuilder.cxx)|—    |—    |`add_modifier`<br>`name`|*`string`*<br>*`string`*|

### KSRootSpaceInteraction

|element name|source files|child elements|child types|attributes|attribute types|
|-----|-----|-----|-----|-----|-----|
|<a name="ksrootspaceinteraction">`ks_root_space_interaction`</a>|[*KSRootSpaceInteractionBuilder.cxx*](https://github.com/KATRIN-Experiment/Kassiopeia/tree/main/Kassiopeia/Bindings/Simulation/Source/KSRootSpaceInteractionBuilder.cxx)|—    |—    |`add_space_interaction`<br>`name`|*`string`*<br>*`string`*|

### KSRootSpaceNavigator

|element name|source files|child elements|child types|attributes|attribute types|
|-----|-----|-----|-----|-----|-----|
|<a name="ksrootspacenavigator">`ks_root_space_navigator`</a>|[*KSRootSpaceNavigatorBuilder.cxx*](https://github.com/KATRIN-Experiment/Kassiopeia/tree/main/Kassiopeia/Bindings/Simulation/Source/KSRootSpaceNavigatorBuilder.cxx)|—    |—    |`name`<br>`set_space_navigator`|*`string`*<br>*`string`*|

### KSRootStepModifier

|element name|source files|child elements|child types|attributes|attribute types|
|-----|-----|-----|-----|-----|-----|
|<a name="ksrootstepmodifier">`ks_root_step_modifier`</a>|[*KSRootStepModifierBuilder.cxx*](https://github.com/KATRIN-Experiment/Kassiopeia/tree/main/Kassiopeia/Bindings/Simulation/Source/KSRootStepModifierBuilder.cxx)|—    |—    |`add_modifier`<br>`name`|*`string`*<br>*`string`*|

### KSRootSurfaceInteraction

|element name|source files|child elements|child types|attributes|attribute types|
|-----|-----|-----|-----|-----|-----|
|<a name="ksrootsurfaceinteraction">`ks_root_surface_interaction`</a>|[*KSRootSurfaceInteractionBuilder.cxx*](https://github.com/KATRIN-Experiment/Kassiopeia/tree/main/Kassiopeia/Bindings/Simulation/Source/KSRootSurfaceInteractionBuilder.cxx)|—    |—    |`name`<br>`set_surface_interaction`|*`string`*<br>*`string`*|

### KSRootSurfaceNavigator

|element name|source files|child elements|child types|attributes|attribute types|
|-----|-----|-----|-----|-----|-----|
|<a name="ksrootsurfacenavigator">`ks_root_surface_navigator`</a>|[*KSRootSurfaceNavigatorBuilder.cxx*](https://github.com/KATRIN-Experiment/Kassiopeia/tree/main/Kassiopeia/Bindings/Simulation/Source/KSRootSurfaceNavigatorBuilder.cxx)|—    |—    |`name`<br>`set_surface_navigator`|*`string`*<br>*`string`*|

### KSRootTerminator

|element name|source files|child elements|child types|attributes|attribute types|
|-----|-----|-----|-----|-----|-----|
|<a name="ksrootterminator">`ks_root_terminator`</a>|[*KSRootTerminatorBuilder.cxx*](https://github.com/KATRIN-Experiment/Kassiopeia/tree/main/Kassiopeia/Bindings/Simulation/Source/KSRootTerminatorBuilder.cxx)|—    |—    |`add_terminator`<br>`name`|*`string`*<br>*`string`*|

### KSRootTrackModifier

|element name|source files|child elements|child types|attributes|attribute types|
|-----|-----|-----|-----|-----|-----|
|<a name="ksroottrackmodifier">`ks_root_track_modifier`</a>|[*KSRootTrackModifierBuilder.cxx*](https://github.com/KATRIN-Experiment/Kassiopeia/tree/main/Kassiopeia/Bindings/Simulation/Source/KSRootTrackModifierBuilder.cxx)|—    |—    |`add_modifier`<br>`name`|*`string`*<br>*`string`*|

### KSRootTrajectory

|element name|source files|child elements|child types|attributes|attribute types|
|-----|-----|-----|-----|-----|-----|
|<a name="ksroottrajectory">`ks_root_trajectory`</a>|[*KSRootTrajectoryBuilder.cxx*](https://github.com/KATRIN-Experiment/Kassiopeia/tree/main/Kassiopeia/Bindings/Simulation/Source/KSRootTrajectoryBuilder.cxx)|—    |—    |`name`<br>`set_trajectory`|*`string`*<br>*`string`*|

### KSRootWriter

|element name|source files|child elements|child types|attributes|attribute types|
|-----|-----|-----|-----|-----|-----|
|<a name="ksrootwriter">`ks_root_writer`</a>|[*KSRootWriterBuilder.cxx*](https://github.com/KATRIN-Experiment/Kassiopeia/tree/main/Kassiopeia/Bindings/Simulation/Source/KSRootWriterBuilder.cxx)|—    |—    |`add_writer`<br>`name`|*`string`*<br>*`string`*|

### KSSimulation

|element name|source files|child elements|child types|attributes|attribute types|
|-----|-----|-----|-----|-----|-----|
|<a name="kssimulation">`ks_simulation`</a>|[*KSSimulationBuilder.cxx*](https://github.com/KATRIN-Experiment/Kassiopeia/tree/main/Kassiopeia/Bindings/Simulation/Source/KSSimulationBuilder.cxx)|—    |—    |`add_static_event_modifier`<br>`add_static_run_modifier`<br>`add_static_step_modifier`<br>`add_static_track_modifier`<br>`command`<br>`electric_field`<br>`events`<br>`generator`<br>`magnetic_field`<br>`name`<br>`run`<br>`seed`<br>`space`<br>`space_interaction`<br>`space_navigator`<br>`step_report_iteration`<br>`surface`<br>`surface_interaction`<br>`surface_navigator`<br>`terminator`<br>`trajectory`<br>`writer`|*`string`*<br>*`string`*<br>*`string`*<br>*`string`*<br>*`string`*<br>*`string`*<br>*`unsigned int`*<br>*`string`*<br>*`string`*<br>*`string`*<br>*`unsigned int`*<br>*`unsigned int`*<br>*`string`*<br>*`string`*<br>*`string`*<br>*`unsigned int`*<br>*`string`*<br>*`string`*<br>*`string`*<br>*`string`*<br>*`string`*<br>*`string`*|

### KElectrostaticConstantField

|element name|source files|child elements|child types|attributes|attribute types|
|-----|-----|-----|-----|-----|-----|
|<a name="kelectrostaticconstantfield">`ksfield_electric_constant`</a>|[*KElectrostaticConstantFieldBuilder.cc*](https://github.com/KATRIN-Experiment/Kassiopeia/tree/main/KEMField/Source/Bindings/Fields/Electric/src/KElectrostaticConstantFieldBuilder.cc)<br>[*KElectrostaticPotentialmapBuilder.cc*](https://github.com/KATRIN-Experiment/Kassiopeia/tree/main/KEMField/Source/Bindings/Fields/Electric/src/KElectrostaticPotentialmapBuilder.cc)<br>[*KSFieldKEMFieldObjectsBuilder.cxx*](https://github.com/KATRIN-Experiment/Kassiopeia/tree/main/Kassiopeia/Bindings/Fields/Source/KSFieldKEMFieldObjectsBuilder.cxx)|—    |—    |`field`<br>`location`<br>`name`<br>`offset_potential`|*`KEMStreamableThreeVector`*<br>*`KEMStreamableThreeVector`*<br>*`string`*<br>*`double`*|

### KInducedAzimuthalElectricField

|element name|source files|child elements|child types|attributes|attribute types|
|-----|-----|-----|-----|-----|-----|
|<a name="kinducedazimuthalelectricfield">`ksfield_electric_induced_azi`</a>|[*KInducedAzimuthalElectricFieldBuilder.cc*](https://github.com/KATRIN-Experiment/Kassiopeia/tree/main/KEMField/Source/Bindings/Fields/Electric/src/KInducedAzimuthalElectricFieldBuilder.cc)<br>[*KSFieldKEMFieldObjectsBuilder.cxx*](https://github.com/KATRIN-Experiment/Kassiopeia/tree/main/Kassiopeia/Bindings/Fields/Source/KSFieldKEMFieldObjectsBuilder.cxx)|—    |—    |`name`<br>`root_field`|*`string`*<br>*`string`*|

### KElectrostaticPotentialmap

|element name|source files|child elements|child types|attributes|attribute types|
|-----|-----|-----|-----|-----|-----|
|<a name="kelectrostaticpotentialmap">`ksfield_electric_potentialmap`</a>|[*KElectrostaticPotentialmapBuilder.cc*](https://github.com/KATRIN-Experiment/Kassiopeia/tree/main/KEMField/Source/Bindings/Fields/Electric/src/KElectrostaticPotentialmapBuilder.cc)<br>[*KSFieldKEMFieldObjectsBuilder.cxx*](https://github.com/KATRIN-Experiment/Kassiopeia/tree/main/Kassiopeia/Bindings/Fields/Source/KSFieldKEMFieldObjectsBuilder.cxx)|—    |—    |`directory`<br>`file`<br>`interpolation`<br>`name`|*`string`*<br>*`string`*<br>*`string`*<br>*`string`*|

### KElectrostaticPotentialmapCalculator

|element name|source files|child elements|child types|attributes|attribute types|
|-----|-----|-----|-----|-----|-----|
|<a name="kelectrostaticpotentialmapcalculator">`ksfield_electric_potentialmap_calculator`</a>|[*KElectrostaticPotentialmapBuilder.cc*](https://github.com/KATRIN-Experiment/Kassiopeia/tree/main/KEMField/Source/Bindings/Fields/Electric/src/KElectrostaticPotentialmapBuilder.cc)<br>[*KSFieldKEMFieldObjectsBuilder.cxx*](https://github.com/KATRIN-Experiment/Kassiopeia/tree/main/Kassiopeia/Bindings/Fields/Source/KSFieldKEMFieldObjectsBuilder.cxx)|[`field_electric_constant`](#kelectrostaticconstantfield)<br>[`field_electric_quadrupole`](#kelectricquadrupolefield)<br>[`field_electrostatic`](#kgelectrostaticboundaryfield)|*`KElectrostaticConstantField`*<br>*`KElectricQuadrupoleField`*<br>*`KGElectrostaticBoundaryField`*|`center`<br>`compute_field`<br>`directory`<br>`field`<br>`file`<br>`force_update`<br>`length`<br>`mirror_x`<br>`mirror_y`<br>`mirror_z`<br>`name`<br>`spaces`<br>`spacing`|*`KEMStreamableThreeVector`*<br>*`bool`*<br>*`string`*<br>*`string`*<br>*`string`*<br>*`bool`*<br>*`KEMStreamableThreeVector`*<br>*`bool`*<br>*`bool`*<br>*`bool`*<br>*`string`*<br>*`string`*<br>*`double`*|

#### KElectricQuadrupoleField

|element name|source files|child elements|child types|attributes|attribute types|
|-----|-----|-----|-----|-----|-----|
|<a name="kelectricquadrupolefield">`field_electric_quadrupole`</a>|[*KElectricQuadrupoleFieldBuilder.cc*](https://github.com/KATRIN-Experiment/Kassiopeia/tree/main/KEMField/Source/Bindings/Fields/Electric/src/KElectricQuadrupoleFieldBuilder.cc)<br>[*KElectrostaticPotentialmapBuilder.cc*](https://github.com/KATRIN-Experiment/Kassiopeia/tree/main/KEMField/Source/Bindings/Fields/Electric/src/KElectrostaticPotentialmapBuilder.cc)<br>[*KSFieldKEMFieldObjectsBuilder.cxx*](https://github.com/KATRIN-Experiment/Kassiopeia/tree/main/Kassiopeia/Bindings/Fields/Source/KSFieldKEMFieldObjectsBuilder.cxx)|—    |—    |`length`<br>`location`<br>`name`<br>`radius`<br>`strength`|*`double`*<br>*`KEMStreamableThreeVector`*<br>*`string`*<br>*`double`*<br>*`double`*|

#### KGElectrostaticBoundaryField

|element name|source files|child elements|child types|attributes|attribute types|
|-----|-----|-----|-----|-----|-----|
|<a name="kgelectrostaticboundaryfield">`field_electrostatic`</a>|[*KElectrostaticBoundaryFieldBuilder.cc*](https://github.com/KATRIN-Experiment/Kassiopeia/tree/main/KEMField/Source/Bindings/Fields/Electric/src/KElectrostaticBoundaryFieldBuilder.cc)<br>[*KElectrostaticPotentialmapBuilder.cc*](https://github.com/KATRIN-Experiment/Kassiopeia/tree/main/KEMField/Source/Bindings/Fields/Electric/src/KElectrostaticPotentialmapBuilder.cc)<br>[*KSFieldKEMFieldObjectsBuilder.cxx*](https://github.com/KATRIN-Experiment/Kassiopeia/tree/main/Kassiopeia/Bindings/Fields/Source/KSFieldKEMFieldObjectsBuilder.cxx)|[`boundary_element_info`](#kboundaryelementinfodisplay)<br>[`cached_bem_solver`](#kcachedchargedensitysolver)<br>[`cached_charge_density_solver`](#kcachedchargedensitysolver)<br>[`explicit_superposition_cached_bem_solver`](#kexplicitsuperpositioncachedchargedensitysolver)<br>[`explicit_superposition_cached_charge_density_solver`](#kexplicitsuperpositioncachedchargedensitysolver)<br>[`fast_multipole_field_solver`](#kelectricfastmultipolefieldsolver)<br>[`gauss_seidel_bem_solver`](#kgaussseidelchargedensitysolver)<br>[`gauss_seidel_charge_density_solver`](#kgaussseidelchargedensitysolver)<br>[`gaussian_elimination_bem_solver`](#kgaussianeliminationchargedensitysolver)<br>[`gaussian_elimination_charge_density_solver`](#kgaussianeliminationchargedensitysolver)<br>[`integrating_field_solver`](#kintegratingelectrostaticfieldsolver)<br>[`krylov_bem_solver`](#kkrylovchargedensitysolverold)<br>[`krylov_bem_solver_new`](#kkrylovchargedensitysolver)<br>[`krylov_bem_solver_old`](#kkrylovchargedensitysolverold)<br>[`krylov_charge_density_solver`](#kkrylovchargedensitysolver)<br>[`krylov_charge_density_solver_old`](#kkrylovchargedensitysolverold)<br>[`robin_hood_bem_solver`](#krobinhoodchargedensitysolver)<br>[`robin_hood_charge_density_solver`](#krobinhoodchargedensitysolver)<br>[`timer`](#kelectrostaticboundaryfieldtimer)<br>[`viewer`](#kvtkviewerasboundaryfieldvisitor)<br>[`zonal_harmonic_field_solver`](#kelectriczhfieldsolver)|*`KBoundaryElementInfoDisplay`*<br>*`KCachedChargeDensitySolver`*<br>*`KCachedChargeDensitySolver`*<br>*`KExplicitSuperpositionCachedChargeDensitySolver`*<br>*`KExplicitSuperpositionCachedChargeDensitySolver`*<br>*`KElectricFastMultipoleFieldSolver`*<br>*`KGaussSeidelChargeDensitySolver`*<br>*`KGaussSeidelChargeDensitySolver`*<br>*`KGaussianEliminationChargeDensitySolver`*<br>*`KGaussianEliminationChargeDensitySolver`*<br>*`KIntegratingElectrostaticFieldSolver`*<br>*`KKrylovChargeDensitySolverOld`*<br>*`KKrylovChargeDensitySolver`*<br>*`KKrylovChargeDensitySolverOld`*<br>*`KKrylovChargeDensitySolver`*<br>*`KKrylovChargeDensitySolverOld`*<br>*`KRobinHoodChargeDensitySolver`*<br>*`KRobinHoodChargeDensitySolver`*<br>*`KElectrostaticBoundaryFieldTimer`*<br>*`KVTKViewerAsBoundaryFieldVisitor`*<br>*`KElectricZHFieldSolver`*|`directory`<br>`file`<br>`hash_masked_bits`<br>`hash_threshold`<br>`maximum_element_aspect_ratio`<br>`minimum_element_area`<br>`name`<br>`spaces`<br>`surfaces`<br>`symmetry`<br>`system`|*`string`*<br>*`string`*<br>*`unsigned int`*<br>*`double`*<br>*`double`*<br>*`double`*<br>*`string`*<br>*`string`*<br>*`string`*<br>*`string`*<br>*`string`*|

##### KBoundaryElementInfoDisplay

|element name|source files|child elements|child types|attributes|attribute types|
|-----|-----|-----|-----|-----|-----|
|<a name="kboundaryelementinfodisplay">`boundary_element_info`</a>|[*KBoundaryElementInfoDisplayBuilder.cc*](https://github.com/KATRIN-Experiment/Kassiopeia/tree/main/KEMField/Source/Bindings/Fields/Electric/src/KBoundaryElementInfoDisplayBuilder.cc)|—    |—    |—    |—    |

##### KCachedChargeDensitySolver

|element name|source files|child elements|child types|attributes|attribute types|
|-----|-----|-----|-----|-----|-----|
|<a name="kcachedchargedensitysolver">`cached_bem_solver`</a>|[*KCachedChargeDensitySolverBuilder.cc*](https://github.com/KATRIN-Experiment/Kassiopeia/tree/main/KEMField/Source/Bindings/ChargeDensitySolvers/Electric/src/KCachedChargeDensitySolverBuilder.cc)|—    |—    |`hash`<br>`name`|*`string`*<br>*`string`*|

##### KExplicitSuperpositionCachedChargeDensitySolver

|element name|source files|child elements|child types|attributes|attribute types|
|-----|-----|-----|-----|-----|-----|
|<a name="kexplicitsuperpositioncachedchargedensitysolver">`explicit_superposition_cached_bem_solver`</a>|[*KExplicitSuperpositionCachedChargeDensitySolverBuilder.cc*](https://github.com/KATRIN-Experiment/Kassiopeia/tree/main/KEMField/Source/Bindings/ChargeDensitySolvers/Electric/src/KExplicitSuperpositionCachedChargeDensitySolverBuilder.cc)|[`component`](#kexplicitsuperpositionsolutioncomponent)|*`KExplicitSuperpositionSolutionComponent`*|`name`|*`string`*|

###### KExplicitSuperpositionSolutionComponent

|element name|source files|child elements|child types|attributes|attribute types|
|-----|-----|-----|-----|-----|-----|
|<a name="kexplicitsuperpositionsolutioncomponent">`component`</a>|[*KExplicitSuperpositionCachedChargeDensitySolverBuilder.cc*](https://github.com/KATRIN-Experiment/Kassiopeia/tree/main/KEMField/Source/Bindings/ChargeDensitySolvers/Electric/src/KExplicitSuperpositionCachedChargeDensitySolverBuilder.cc)|—    |—    |`hash`<br>`name`<br>`scale`|*`string`*<br>*`string`*<br>*`double`*|

##### KElectricFastMultipoleFieldSolver

|element name|source files|child elements|child types|attributes|attribute types|
|-----|-----|-----|-----|-----|-----|
|<a name="kelectricfastmultipolefieldsolver">`fast_multipole_field_solver`</a>|[*KElectricFastMultipoleFieldSolverBuilder.cc*](https://github.com/KATRIN-Experiment/Kassiopeia/tree/main/KEMField/Source/Bindings/FieldSolvers/Electric/src/KElectricFastMultipoleFieldSolverBuilder.cc)|[`viewer`](#kfmvtkelectrostatictreeviewerdata)|*`KFMVTKElectrostaticTreeViewerData`*|`expansion_degree`<br>`insertion_ratio`<br>`maximum_tree_depth`<br>`neighbor_order`<br>`region_expansion_factor`<br>`split_mode`<br>`top_level_divisions`<br>`tree_level_divisions`<br>`use_caching`<br>`use_opencl`<br>`use_region_size_estimation`<br>`verbosity`<br>`world_cube_center_x`<br>`world_cube_center_y`<br>`world_cube_center_z`<br>`world_cube_length`|*`unsigned int`*<br>*`double`*<br>*`unsigned int`*<br>*`unsigned int`*<br>*`double`*<br>*`bool`*<br>*`unsigned int`*<br>*`unsigned int`*<br>*`bool`*<br>*`bool`*<br>*`bool`*<br>*`unsigned int`*<br>*`double`*<br>*`double`*<br>*`double`*<br>*`double`*|

###### KFMVTKElectrostaticTreeViewerData

|element name|source files|child elements|child types|attributes|attribute types|
|-----|-----|-----|-----|-----|-----|
|<a name="kfmvtkelectrostatictreeviewerdata">`viewer`</a>|[*KFMVTKElectrostaticTreeViewerBuilder.cc*](https://github.com/KATRIN-Experiment/Kassiopeia/tree/main/KEMField/Source/Bindings/FieldSolvers/Electric/src/KFMVTKElectrostaticTreeViewerBuilder.cc)|—    |—    |`file`<br>`save`<br>`view`|*`string`*<br>*`bool`*<br>*`bool `*|

##### KGaussSeidelChargeDensitySolver

|element name|source files|child elements|child types|attributes|attribute types|
|-----|-----|-----|-----|-----|-----|
|<a name="kgaussseidelchargedensitysolver">`gauss_seidel_bem_solver`</a>|[*KGaussSeidelChargeDensitySolverBuilder.cc*](https://github.com/KATRIN-Experiment/Kassiopeia/tree/main/KEMField/Source/Bindings/ChargeDensitySolvers/Electric/src/KGaussSeidelChargeDensitySolverBuilder.cc)|—    |—    |`integrator`<br>`use_opencl`|*`string`*<br>*`bool`*|

##### KGaussianEliminationChargeDensitySolver

|element name|source files|child elements|child types|attributes|attribute types|
|-----|-----|-----|-----|-----|-----|
|<a name="kgaussianeliminationchargedensitysolver">`gaussian_elimination_bem_solver`</a>|[*KGaussianEliminationChargeDensitySolverBuilder.cc*](https://github.com/KATRIN-Experiment/Kassiopeia/tree/main/KEMField/Source/Bindings/ChargeDensitySolvers/Electric/src/KGaussianEliminationChargeDensitySolverBuilder.cc)|—    |—    |`integrator`|*`string`*|

##### KIntegratingElectrostaticFieldSolver

|element name|source files|child elements|child types|attributes|attribute types|
|-----|-----|-----|-----|-----|-----|
|<a name="kintegratingelectrostaticfieldsolver">`integrating_field_solver`</a>|[*KIntegratingElectrostaticFieldSolverBuilder.cc*](https://github.com/KATRIN-Experiment/Kassiopeia/tree/main/KEMField/Source/Bindings/FieldSolvers/Electric/src/KIntegratingElectrostaticFieldSolverBuilder.cc)|—    |—    |`integrator`<br>`use_opencl`|*`string`*<br>*`bool`*|

##### KKrylovChargeDensitySolverOld

|element name|source files|child elements|child types|attributes|attribute types|
|-----|-----|-----|-----|-----|-----|
|<a name="kkrylovchargedensitysolverold">`krylov_bem_solver`</a>|[*KKrylovChargeDensitySolverOldBuilder.cc*](https://github.com/KATRIN-Experiment/Kassiopeia/tree/main/KEMField/Source/Bindings/ChargeDensitySolvers/Electric/src/KKrylovChargeDensitySolverOldBuilder.cc)|[`fftm_multiplication`](#kfmelectrostaticparameters)<br>[`preconditioner_electrostatic_parameters`](#kfmelectrostaticparameters)|*`KFMElectrostaticParameters`*<br>*`KFMElectrostaticParameters`*|`intermediate_save_interval`<br>`iterations_between_restarts`<br>`max_iterations`<br>`max_preconditioner_iterations`<br>`preconditioner`<br>`preconditioner_degree`<br>`preconditioner_tolerance`<br>`show_plot`<br>`solver_name`<br>`time_check_interval`<br>`time_limit_in_seconds`<br>`tolerance`<br>`use_display`<br>`use_timer`|*`unsigned int`*<br>*`unsigned int`*<br>*`unsigned int`*<br>*`unsigned int`*<br>*`string`*<br>*`unsigned int`*<br>*`double`*<br>*`bool`*<br>*`string`*<br>*`unsigned int`*<br>*`double`*<br>*`double`*<br>*`bool`*<br>*`bool`*|

###### KFMElectrostaticParameters

|element name|source files|child elements|child types|attributes|attribute types|
|-----|-----|-----|-----|-----|-----|
|<a name="kfmelectrostaticparameters">`fftm_multiplication`</a>|[*KKrylovChargeDensitySolverOldBuilder.cc*](https://github.com/KATRIN-Experiment/Kassiopeia/tree/main/KEMField/Source/Bindings/ChargeDensitySolvers/Electric/src/KKrylovChargeDensitySolverOldBuilder.cc)|—    |—    |`allowed_fraction`<br>`allowed_number`<br>`bias_degree`<br>`expansion_degree`<br>`insertion_ratio`<br>`maximum_tree_depth`<br>`neighbor_order`<br>`region_expansion_factor`<br>`strategy`<br>`top_level_divisions`<br>`tree_level_divisions`<br>`use_caching`<br>`use_region_size_estimation`<br>`verbosity`<br>`world_cube_center_x`<br>`world_cube_center_y`<br>`world_cube_center_z`<br>`world_cube_length`|*`unsigned int`*<br>*`unsigned int`*<br>*`double`*<br>*`unsigned int`*<br>*`double`*<br>*`unsigned int`*<br>*`unsigned int`*<br>*`double`*<br>*`string`*<br>*`unsigned int`*<br>*`unsigned int`*<br>*`bool`*<br>*`bool`*<br>*`unsigned int`*<br>*`double`*<br>*`double`*<br>*`double`*<br>*`double`*|

##### KKrylovChargeDensitySolver

|element name|source files|child elements|child types|attributes|attribute types|
|-----|-----|-----|-----|-----|-----|
|<a name="kkrylovchargedensitysolver">`krylov_bem_solver_new`</a>|[*KKrylovChargeDensitySolverBuilder.cc*](https://github.com/KATRIN-Experiment/Kassiopeia/tree/main/KEMField/Source/Bindings/ChargeDensitySolvers/Electric/src/KKrylovChargeDensitySolverBuilder.cc)|[`krylov_preconditioner`](#kkrylovpreconditionergenerator)|*`KKrylovPreconditionerGenerator`*|`intermediate_save_interval`<br>`iterations_between_restarts`<br>`max_iterations`<br>`show_plot`<br>`solver_name`<br>`time_check_interval`<br>`time_limit_in_seconds`<br>`tolerance`<br>`use_display`<br>`use_timer`|*`unsigned int`*<br>*`unsigned int`*<br>*`unsigned int`*<br>*`bool`*<br>*`string`*<br>*`unsigned int`*<br>*`double`*<br>*`double`*<br>*`bool`*<br>*`bool`*|

###### KKrylovPreconditionerGenerator

|element name|source files|child elements|child types|attributes|attribute types|
|-----|-----|-----|-----|-----|-----|
|<a name="kkrylovpreconditionergenerator">`krylov_preconditioner`</a>|[*KKrylovPreconditionerGeneratorBuilder.cc*](https://github.com/KATRIN-Experiment/Kassiopeia/tree/main/KEMField/Source/Bindings/ChargeDensitySolvers/Electric/src/KKrylovPreconditionerGeneratorBuilder.cc)|[`krylov_preconditioner`](#kkrylovpreconditionergenerator)|*`KKrylovPreconditionerGenerator`*|`intermediate_save_interval`<br>`iterations_between_restarts`<br>`max_iterations`<br>`show_plot`<br>`solver_name`<br>`time_check_interval`<br>`time_limit_in_seconds`<br>`tolerance`<br>`use_display`<br>`use_timer`|*`unsigned int`*<br>*`unsigned int`*<br>*`unsigned int`*<br>*`bool`*<br>*`string`*<br>*`unsigned int`*<br>*`double`*<br>*`double`*<br>*`bool`*<br>*`bool`*|

##### KRobinHoodChargeDensitySolver

|element name|source files|child elements|child types|attributes|attribute types|
|-----|-----|-----|-----|-----|-----|
|<a name="krobinhoodchargedensitysolver">`robin_hood_bem_solver`</a>|[*KRobinHoodChargeDensitySolverBuilder.cc*](https://github.com/KATRIN-Experiment/Kassiopeia/tree/main/KEMField/Source/Bindings/ChargeDensitySolvers/Electric/src/KRobinHoodChargeDensitySolverBuilder.cc)|—    |—    |`cache_matrix_elements`<br>`check_sub_interval`<br>`display_interval`<br>`integrator`<br>`plot_interval`<br>`split_mode`<br>`tolerance`<br>`use_opencl`<br>`use_vtk`<br>`write_interval`|*`bool`*<br>*`unsigned int`*<br>*`unsigned int`*<br>*`string`*<br>*`unsigned int`*<br>*`bool`*<br>*`double`*<br>*`bool`*<br>*`bool`*<br>*`unsigned int`*|

##### KElectrostaticBoundaryFieldTimer

|element name|source files|child elements|child types|attributes|attribute types|
|-----|-----|-----|-----|-----|-----|
|<a name="kelectrostaticboundaryfieldtimer">`timer`</a>|[*KElectrostaticBoundaryFieldTimerBuilder.cc*](https://github.com/KATRIN-Experiment/Kassiopeia/tree/main/KEMField/Source/Bindings/Fields/Electric/src/KElectrostaticBoundaryFieldTimerBuilder.cc)|—    |—    |—    |—    |

##### KVTKViewerAsBoundaryFieldVisitor

|element name|source files|child elements|child types|attributes|attribute types|
|-----|-----|-----|-----|-----|-----|
|<a name="kvtkviewerasboundaryfieldvisitor">`viewer`</a>|[*KVTKViewerVisitorBuilder.cc*](https://github.com/KATRIN-Experiment/Kassiopeia/tree/main/KEMField/Source/Bindings/Fields/Electric/src/KVTKViewerVisitorBuilder.cc)|—    |—    |`file`<br>`path`<br>`postprocessing`<br>`preprocessing`<br>`save`<br>`view`|*`string`*<br>*`string`*<br>*`bool`*<br>*`bool`*<br>*`bool`*<br>*`bool`*|

##### KElectricZHFieldSolver

|element name|source files|child elements|child types|attributes|attribute types|
|-----|-----|-----|-----|-----|-----|
|<a name="kelectriczhfieldsolver">`zonal_harmonic_field_solver`</a>|[*KElectricZHFieldSolverBuilder.cc*](https://github.com/KATRIN-Experiment/Kassiopeia/tree/main/KEMField/Source/Bindings/FieldSolvers/Electric/src/KElectricZHFieldSolverBuilder.cc)|—    |—    |`central_sourcepoint_end`<br>`central_sourcepoint_fractional_distance`<br>`central_sourcepoint_spacing`<br>`central_sourcepoint_start`<br>`coaxiality_tolerance`<br>`convergence_parameter`<br>`convergence_ratio`<br>`integrator`<br>`number_of_bifurcations`<br>`number_of_central_coefficients`<br>`number_of_remote_coefficients`<br>`number_of_remote_sourcepoints`<br>`proximity_to_sourcepoint`<br>`remote_sourcepoint_end`<br>`remote_sourcepoint_start`<br>`use_fractional_central_sourcepoint_spacing`|*`double`*<br>*`double`*<br>*`double`*<br>*`double`*<br>*`double`*<br>*`double`*<br>*`double`*<br>*`string`*<br>*`int`*<br>*`int`*<br>*`int`*<br>*`int`*<br>*`double`*<br>*`double`*<br>*`double`*<br>*`bool`*|

### KRampedElectricField

|element name|source files|child elements|child types|attributes|attribute types|
|-----|-----|-----|-----|-----|-----|
|<a name="krampedelectricfield">`ksfield_electric_ramped`</a>|[*KRampedElectricFieldBuilder.cc*](https://github.com/KATRIN-Experiment/Kassiopeia/tree/main/KEMField/Source/Bindings/Fields/Electric/src/KRampedElectricFieldBuilder.cc)<br>[*KSFieldKEMFieldObjectsBuilder.cxx*](https://github.com/KATRIN-Experiment/Kassiopeia/tree/main/Kassiopeia/Bindings/Fields/Source/KSFieldKEMFieldObjectsBuilder.cxx)|—    |—    |`name`<br>`num_cycles`<br>`ramp_down_delay`<br>`ramp_down_time`<br>`ramp_up_delay`<br>`ramp_up_time`<br>`ramping_type`<br>`root_field`<br>`time_constant`<br>`time_scaling`|*`string`*<br>*`int`*<br>*`double`*<br>*`double`*<br>*`double`*<br>*`double`*<br>*`string`*<br>*`string`*<br>*`double`*<br>*`double`*|

### KRampedElectric2Field

|element name|source files|child elements|child types|attributes|attribute types|
|-----|-----|-----|-----|-----|-----|
|<a name="krampedelectric2field">`ksfield_electric_ramped_2fields`</a>|[*KRampedElectric2FieldBuilder.cc*](https://github.com/KATRIN-Experiment/Kassiopeia/tree/main/KEMField/Source/Bindings/Fields/Electric/src/KRampedElectric2FieldBuilder.cc)<br>[*KSFieldKEMFieldObjectsBuilder.cxx*](https://github.com/KATRIN-Experiment/Kassiopeia/tree/main/Kassiopeia/Bindings/Fields/Source/KSFieldKEMFieldObjectsBuilder.cxx)|—    |—    |`focus_exponent`<br>`focus_time`<br>`name`<br>`num_cycles`<br>`potential_scaling`<br>`ramp_down_delay`<br>`ramp_down_time`<br>`ramp_up_delay`<br>`ramp_up_time`<br>`ramping_type`<br>`root_field_1`<br>`root_field_2`<br>`small_spectrometer`<br>`time_constant`<br>`time_scaling`|*`double`*<br>*`double`*<br>*`string`*<br>*`int`*<br>*`double`*<br>*`double`*<br>*`double`*<br>*`double`*<br>*`double`*<br>*`string`*<br>*`string`*<br>*`string`*<br>*`bool`*<br>*`double`*<br>*`double`*|

### KGStaticElectromagnetField

|element name|source files|child elements|child types|attributes|attribute types|
|-----|-----|-----|-----|-----|-----|
|<a name="kgstaticelectromagnetfield">`ksfield_electromagnet`</a>|[*KStaticElectromagnetFieldBuilder.cc*](https://github.com/KATRIN-Experiment/Kassiopeia/tree/main/KEMField/Source/Bindings/Fields/Magnetic/src/KStaticElectromagnetFieldBuilder.cc)<br>[*KSFieldKEMFieldObjectsBuilder.cxx*](https://github.com/KATRIN-Experiment/Kassiopeia/tree/main/Kassiopeia/Bindings/Fields/Source/KSFieldKEMFieldObjectsBuilder.cxx)|[`integrating_field_solver`](#kintegratingmagnetostaticfieldsolver)<br>[`zonal_harmonic_field_solver`](#kzonalharmonicmagnetostaticfieldsolver)|*`KIntegratingMagnetostaticFieldSolver`*<br>*`KZonalHarmonicMagnetostaticFieldSolver`*|`directory`<br>`directory_magfield3`<br>`file`<br>`name`<br>`save_magfield3`<br>`spaces`<br>`surfaces`<br>`system`|*`string`*<br>*`string`*<br>*`string`*<br>*`string`*<br>*`bool`*<br>*`string`*<br>*`string`*<br>*`string`*|

#### KIntegratingMagnetostaticFieldSolver

|element name|source files|child elements|child types|attributes|attribute types|
|-----|-----|-----|-----|-----|-----|
|<a name="kintegratingmagnetostaticfieldsolver">`integrating_field_solver`</a>|[*KIntegratingMagnetostaticFieldSolverBuilder.cc*](https://github.com/KATRIN-Experiment/Kassiopeia/tree/main/KEMField/Source/Bindings/FieldSolvers/Magnetic/src/KIntegratingMagnetostaticFieldSolverBuilder.cc)|—    |—    |—    |—    |

#### KZonalHarmonicMagnetostaticFieldSolver

|element name|source files|child elements|child types|attributes|attribute types|
|-----|-----|-----|-----|-----|-----|
|<a name="kzonalharmonicmagnetostaticfieldsolver">`zonal_harmonic_field_solver`</a>|[*KZonalHarmonicMagnetostaticFieldSolverBuilder.cc*](https://github.com/KATRIN-Experiment/Kassiopeia/tree/main/KEMField/Source/Bindings/FieldSolvers/Magnetic/src/KZonalHarmonicMagnetostaticFieldSolverBuilder.cc)|—    |—    |`central_sourcepoint_end`<br>`central_sourcepoint_fractional_distance`<br>`central_sourcepoint_spacing`<br>`central_sourcepoint_start`<br>`coaxiality_tolerance`<br>`convergence_parameter`<br>`convergence_ratio`<br>`number_of_bifurcations`<br>`number_of_central_coefficients`<br>`number_of_remote_coefficients`<br>`number_of_remote_sourcepoints`<br>`proximity_to_sourcepoint`<br>`remote_sourcepoint_end`<br>`remote_sourcepoint_start`<br>`use_fractional_central_sourcepoint_spacing`|*`double`*<br>*`double`*<br>*`double`*<br>*`double`*<br>*`double`*<br>*`double`*<br>*`double`*<br>*`int`*<br>*`int`*<br>*`int`*<br>*`int`*<br>*`double`*<br>*`double`*<br>*`double`*<br>*`bool`*|

### KMagnetostaticConstantField

|element name|source files|child elements|child types|attributes|attribute types|
|-----|-----|-----|-----|-----|-----|
|<a name="kmagnetostaticconstantfield">`ksfield_magnetic_constant`</a>|[*KMagnetostaticConstantFieldBuilder.cc*](https://github.com/KATRIN-Experiment/Kassiopeia/tree/main/KEMField/Source/Bindings/Fields/Magnetic/src/KMagnetostaticConstantFieldBuilder.cc)<br>[*KMagnetostaticFieldmapBuilder.cc*](https://github.com/KATRIN-Experiment/Kassiopeia/tree/main/KEMField/Source/Bindings/Fields/Magnetic/src/KMagnetostaticFieldmapBuilder.cc)<br>[*KSFieldKEMFieldObjectsBuilder.cxx*](https://github.com/KATRIN-Experiment/Kassiopeia/tree/main/Kassiopeia/Bindings/Fields/Source/KSFieldKEMFieldObjectsBuilder.cxx)|—    |—    |`field`<br>`location`<br>`name`|*`KEMStreamableThreeVector`*<br>*`KEMStreamableThreeVector`*<br>*`string`*|

### KMagneticDipoleField

|element name|source files|child elements|child types|attributes|attribute types|
|-----|-----|-----|-----|-----|-----|
|<a name="kmagneticdipolefield">`ksfield_magnetic_dipole`</a>|[*KMagneticDipoleFieldBuilder.cc*](https://github.com/KATRIN-Experiment/Kassiopeia/tree/main/KEMField/Source/Bindings/Fields/Magnetic/src/KMagneticDipoleFieldBuilder.cc)<br>[*KSFieldKEMFieldObjectsBuilder.cxx*](https://github.com/KATRIN-Experiment/Kassiopeia/tree/main/Kassiopeia/Bindings/Fields/Source/KSFieldKEMFieldObjectsBuilder.cxx)|—    |—    |`location`<br>`moment`<br>`name`|*`KEMStreamableThreeVector`*<br>*`KEMStreamableThreeVector`*<br>*`string`*|

### KMagnetostaticFieldmap

|element name|source files|child elements|child types|attributes|attribute types|
|-----|-----|-----|-----|-----|-----|
|<a name="kmagnetostaticfieldmap">`ksfield_magnetic_fieldmap`</a>|[*KMagnetostaticFieldmapBuilder.cc*](https://github.com/KATRIN-Experiment/Kassiopeia/tree/main/KEMField/Source/Bindings/Fields/Magnetic/src/KMagnetostaticFieldmapBuilder.cc)<br>[*KSFieldKEMFieldObjectsBuilder.cxx*](https://github.com/KATRIN-Experiment/Kassiopeia/tree/main/Kassiopeia/Bindings/Fields/Source/KSFieldKEMFieldObjectsBuilder.cxx)|—    |—    |`directory`<br>`file`<br>`interpolation`<br>`magnetic_gradient_numerical`<br>`name`|*`string`*<br>*`string`*<br>*`string`*<br>*`bool`*<br>*`string`*|

### KMagnetostaticFieldmapCalculator

|element name|source files|child elements|child types|attributes|attribute types|
|-----|-----|-----|-----|-----|-----|
|<a name="kmagnetostaticfieldmapcalculator">`ksfield_magnetic_fieldmap_calculator`</a>|[*KMagnetostaticFieldmapBuilder.cc*](https://github.com/KATRIN-Experiment/Kassiopeia/tree/main/KEMField/Source/Bindings/Fields/Magnetic/src/KMagnetostaticFieldmapBuilder.cc)<br>[*KSFieldKEMFieldObjectsBuilder.cxx*](https://github.com/KATRIN-Experiment/Kassiopeia/tree/main/Kassiopeia/Bindings/Fields/Source/KSFieldKEMFieldObjectsBuilder.cxx)|[`field_electromagnet`](#kstaticelectromagnetfield)<br>[`field_magnetic_constant`](#kmagnetostaticconstantfield)<br>[`field_magnetic_dipole`](#kmagneticdipolefieldbuilder)|*`KStaticElectromagnetField`*<br>*`KMagnetostaticConstantField`*<br>*`KMagneticDipoleFieldBuilder`*|`center`<br>`compute_gradient`<br>`directory`<br>`field`<br>`file`<br>`force_update`<br>`length`<br>`mirror_x`<br>`mirror_y`<br>`mirror_z`<br>`name`<br>`spaces`<br>`spacing`<br>`time`|*`KEMStreamableThreeVector`*<br>*`bool`*<br>*`string`*<br>*`string`*<br>*`string`*<br>*`bool`*<br>*`KEMStreamableThreeVector`*<br>*`bool`*<br>*`bool`*<br>*`bool`*<br>*`string`*<br>*`string`*<br>*`double`*<br>*`double`*|

#### KStaticElectromagnetField

|element name|source files|child elements|child types|attributes|attribute types|
|-----|-----|-----|-----|-----|-----|
|<a name="kstaticelectromagnetfield">`field_electromagnet`</a>|[*KMagnetostaticFieldmapBuilder.cc*](https://github.com/KATRIN-Experiment/Kassiopeia/tree/main/KEMField/Source/Bindings/Fields/Magnetic/src/KMagnetostaticFieldmapBuilder.cc)|—    |—    |—    |—    |

#### KMagneticDipoleFieldBuilder

|element name|source files|child elements|child types|attributes|attribute types|
|-----|-----|-----|-----|-----|-----|
|<a name="kmagneticdipolefieldbuilder">`field_magnetic_dipole`</a>|[*KMagnetostaticFieldmapBuilder.cc*](https://github.com/KATRIN-Experiment/Kassiopeia/tree/main/KEMField/Source/Bindings/Fields/Magnetic/src/KMagnetostaticFieldmapBuilder.cc)|—    |—    |—    |—    |

### KRampedMagneticField

|element name|source files|child elements|child types|attributes|attribute types|
|-----|-----|-----|-----|-----|-----|
|<a name="krampedmagneticfield">`ksfield_magnetic_ramped`</a>|[*KRampedMagneticFieldBuilder.cc*](https://github.com/KATRIN-Experiment/Kassiopeia/tree/main/KEMField/Source/Bindings/Fields/Magnetic/src/KRampedMagneticFieldBuilder.cc)<br>[*KSFieldKEMFieldObjectsBuilder.cxx*](https://github.com/KATRIN-Experiment/Kassiopeia/tree/main/Kassiopeia/Bindings/Fields/Source/KSFieldKEMFieldObjectsBuilder.cxx)|—    |—    |`name`<br>`num_cycles`<br>`ramp_down_delay`<br>`ramp_down_time`<br>`ramp_up_delay`<br>`ramp_up_time`<br>`ramping_type`<br>`root_field`<br>`time_constant`<br>`time_constant_2`<br>`time_scaling`|*`string`*<br>*`int`*<br>*`double`*<br>*`double`*<br>*`double`*<br>*`double`*<br>*`string`*<br>*`string`*<br>*`double`*<br>*`double`*<br>*`double`*|

### KMagneticSuperpositionField

|element name|source files|child elements|child types|attributes|attribute types|
|-----|-----|-----|-----|-----|-----|
|<a name="kmagneticsuperpositionfield">`ksfield_magnetic_super_position`</a>|[*KMagneticSuperpositionFieldBuilder.cc*](https://github.com/KATRIN-Experiment/Kassiopeia/tree/main/KEMField/Source/Bindings/Fields/Magnetic/src/KMagneticSuperpositionFieldBuilder.cc)<br>[*KSFieldKEMFieldObjectsBuilder.cxx*](https://github.com/KATRIN-Experiment/Kassiopeia/tree/main/Kassiopeia/Bindings/Fields/Source/KSFieldKEMFieldObjectsBuilder.cxx)|[`add_field`](#kmagneticsuperpositionfieldentry)|*`KMagneticSuperpositionFieldEntry`*|`name`<br>`require`<br>`use_caching`|*`string`*<br>*`string`*<br>*`bool`*|

#### KMagneticSuperpositionFieldEntry

|element name|source files|child elements|child types|attributes|attribute types|
|-----|-----|-----|-----|-----|-----|
|<a name="kmagneticsuperpositionfieldentry">`add_field`</a>|[*KMagneticSuperpositionFieldBuilder.cc*](https://github.com/KATRIN-Experiment/Kassiopeia/tree/main/KEMField/Source/Bindings/Fields/Magnetic/src/KMagneticSuperpositionFieldBuilder.cc)|—    |—    |`enhancement`<br>`name`|*`double`*<br>*`string`*|

### KSGenDirectionSphericalComposite

|element name|source files|child elements|child types|attributes|attribute types|
|-----|-----|-----|-----|-----|-----|
|<a name="ksgendirectionsphericalcomposite">`ksgen_direction_spherical_composite`</a>|[*KSGenDirectionSphericalCompositeBuilder.cxx*](https://github.com/KATRIN-Experiment/Kassiopeia/tree/main/Kassiopeia/Bindings/Generators/Source/KSGenDirectionSphericalCompositeBuilder.cxx)<br>[*KSGenGeneratorCompositeBuilder.cxx*](https://github.com/KATRIN-Experiment/Kassiopeia/tree/main/Kassiopeia/Bindings/Generators/Source/KSGenGeneratorCompositeBuilder.cxx)|[`phi_fix`](#ksgenvaluefix)<br>[`phi_formula`](#ksgenvalueformula)<br>[`phi_gauss`](#ksgenvaluegauss)<br>[`phi_generalized_gauss`](#ksgenvaluegeneralizedgauss)<br>[`phi_histogram`](#ksgenvaluehistogram)<br>[`phi_list`](#ksgenvaluelist)<br>[`phi_set`](#ksgenvalueset)<br>[`phi_uniform`](#ksgenvalueuniform)<br>[`theta_cosine`](#ksgenvalueanglecosine)<br>[`theta_fix`](#ksgenvaluefix)<br>[`theta_formula`](#ksgenvalueformula)<br>[`theta_gauss`](#ksgenvaluegauss)<br>[`theta_generalized_gauss`](#ksgenvaluegeneralizedgauss)<br>[`theta_histogram`](#ksgenvaluehistogram)<br>[`theta_list`](#ksgenvaluelist)<br>[`theta_set`](#ksgenvalueset)<br>[`theta_spherical`](#ksgenvalueanglespherical)<br>[`theta_uniform`](#ksgenvalueuniform)|*`KSGenValueFix`*<br>*`KSGenValueFormula`*<br>*`KSGenValueGauss`*<br>*`KSGenValueGeneralizedGauss`*<br>*`KSGenValueHistogram`*<br>*`KSGenValueList`*<br>*`KSGenValueSet`*<br>*`KSGenValueUniform`*<br>*`KSGenValueAngleCosine`*<br>*`KSGenValueFix`*<br>*`KSGenValueFormula`*<br>*`KSGenValueGauss`*<br>*`KSGenValueGeneralizedGauss`*<br>*`KSGenValueHistogram`*<br>*`KSGenValueList`*<br>*`KSGenValueSet`*<br>*`KSGenValueAngleSpherical`*<br>*`KSGenValueUniform`*|`name`<br>`phi`<br>`space`<br>`surface`<br>`theta`|*`string`*<br>*`string`*<br>*`string`*<br>*`string`*<br>*`string`*|

#### KSGenValueFix

|element name|source files|child elements|child types|attributes|attribute types|
|-----|-----|-----|-----|-----|-----|
|<a name="ksgenvaluefix">`phi_fix`</a>|[*KSGenDirectionSphericalCompositeBuilder.cxx*](https://github.com/KATRIN-Experiment/Kassiopeia/tree/main/Kassiopeia/Bindings/Generators/Source/KSGenDirectionSphericalCompositeBuilder.cxx)<br>[*KSGenDirectionSphericalMagneticFieldBuilder.cxx*](https://github.com/KATRIN-Experiment/Kassiopeia/tree/main/Kassiopeia/Bindings/Generators/Source/KSGenDirectionSphericalMagneticFieldBuilder.cxx)<br>[*KSGenDirectionSurfaceCompositeBuilder.cxx*](https://github.com/KATRIN-Experiment/Kassiopeia/tree/main/Kassiopeia/Bindings/Generators/Source/KSGenDirectionSurfaceCompositeBuilder.cxx)<br>[*KSGenEnergyCompositeBuilder.cxx*](https://github.com/KATRIN-Experiment/Kassiopeia/tree/main/Kassiopeia/Bindings/Generators/Source/KSGenEnergyCompositeBuilder.cxx)<br>[*KSGenGeneratorCompositeBuilder.cxx*](https://github.com/KATRIN-Experiment/Kassiopeia/tree/main/Kassiopeia/Bindings/Generators/Source/KSGenGeneratorCompositeBuilder.cxx)<br>[*KSGenLCompositeBuilder.cxx*](https://github.com/KATRIN-Experiment/Kassiopeia/tree/main/Kassiopeia/Bindings/Generators/Source/KSGenLCompositeBuilder.cxx)<br>[*KSGenMomentumRectangularCompositeBuilder.cxx*](https://github.com/KATRIN-Experiment/Kassiopeia/tree/main/Kassiopeia/Bindings/Generators/Source/KSGenMomentumRectangularCompositeBuilder.cxx)<br>[*KSGenNCompositeBuilder.cxx*](https://github.com/KATRIN-Experiment/Kassiopeia/tree/main/Kassiopeia/Bindings/Generators/Source/KSGenNCompositeBuilder.cxx)<br>[*KSGenPositionCylindricalCompositeBuilder.cxx*](https://github.com/KATRIN-Experiment/Kassiopeia/tree/main/Kassiopeia/Bindings/Generators/Source/KSGenPositionCylindricalCompositeBuilder.cxx)<br>[*KSGenPositionFluxTubeBuilder.cxx*](https://github.com/KATRIN-Experiment/Kassiopeia/tree/main/Kassiopeia/Bindings/Generators/Source/KSGenPositionFluxTubeBuilder.cxx)<br>[*KSGenPositionFrustrumCompositeBuilder.cxx*](https://github.com/KATRIN-Experiment/Kassiopeia/tree/main/Kassiopeia/Bindings/Generators/Source/KSGenPositionFrustrumCompositeBuilder.cxx)<br>[*KSGenPositionRectangularCompositeBuilder.cxx*](https://github.com/KATRIN-Experiment/Kassiopeia/tree/main/Kassiopeia/Bindings/Generators/Source/KSGenPositionRectangularCompositeBuilder.cxx)<br>[*KSGenPositionSphericalCompositeBuilder.cxx*](https://github.com/KATRIN-Experiment/Kassiopeia/tree/main/Kassiopeia/Bindings/Generators/Source/KSGenPositionSphericalCompositeBuilder.cxx)<br>[*KSGenSpinCompositeBuilder.cxx*](https://github.com/KATRIN-Experiment/Kassiopeia/tree/main/Kassiopeia/Bindings/Generators/Source/KSGenSpinCompositeBuilder.cxx)<br>[*KSGenSpinRelativeCompositeBuilder.cxx*](https://github.com/KATRIN-Experiment/Kassiopeia/tree/main/Kassiopeia/Bindings/Generators/Source/KSGenSpinRelativeCompositeBuilder.cxx)<br>[*KSGenTimeCompositeBuilder.cxx*](https://github.com/KATRIN-Experiment/Kassiopeia/tree/main/Kassiopeia/Bindings/Generators/Source/KSGenTimeCompositeBuilder.cxx)<br>[*KSGenValueFixBuilder.cxx*](https://github.com/KATRIN-Experiment/Kassiopeia/tree/main/Kassiopeia/Bindings/Generators/Source/KSGenValueFixBuilder.cxx)|—    |—    |`name`<br>`value`|*`string`*<br>*`double`*|

#### KSGenValueFormula

|element name|source files|child elements|child types|attributes|attribute types|
|-----|-----|-----|-----|-----|-----|
|<a name="ksgenvalueformula">`phi_formula`</a>|[*KSGenDirectionSphericalCompositeBuilder.cxx*](https://github.com/KATRIN-Experiment/Kassiopeia/tree/main/Kassiopeia/Bindings/Generators/Source/KSGenDirectionSphericalCompositeBuilder.cxx)<br>[*KSGenDirectionSphericalMagneticFieldBuilder.cxx*](https://github.com/KATRIN-Experiment/Kassiopeia/tree/main/Kassiopeia/Bindings/Generators/Source/KSGenDirectionSphericalMagneticFieldBuilder.cxx)<br>[*KSGenDirectionSurfaceCompositeBuilder.cxx*](https://github.com/KATRIN-Experiment/Kassiopeia/tree/main/Kassiopeia/Bindings/Generators/Source/KSGenDirectionSurfaceCompositeBuilder.cxx)<br>[*KSGenEnergyCompositeBuilder.cxx*](https://github.com/KATRIN-Experiment/Kassiopeia/tree/main/Kassiopeia/Bindings/Generators/Source/KSGenEnergyCompositeBuilder.cxx)<br>[*KSGenGeneratorCompositeBuilder.cxx*](https://github.com/KATRIN-Experiment/Kassiopeia/tree/main/Kassiopeia/Bindings/Generators/Source/KSGenGeneratorCompositeBuilder.cxx)<br>[*KSGenLCompositeBuilder.cxx*](https://github.com/KATRIN-Experiment/Kassiopeia/tree/main/Kassiopeia/Bindings/Generators/Source/KSGenLCompositeBuilder.cxx)<br>[*KSGenMomentumRectangularCompositeBuilder.cxx*](https://github.com/KATRIN-Experiment/Kassiopeia/tree/main/Kassiopeia/Bindings/Generators/Source/KSGenMomentumRectangularCompositeBuilder.cxx)<br>[*KSGenNCompositeBuilder.cxx*](https://github.com/KATRIN-Experiment/Kassiopeia/tree/main/Kassiopeia/Bindings/Generators/Source/KSGenNCompositeBuilder.cxx)<br>[*KSGenPositionCylindricalCompositeBuilder.cxx*](https://github.com/KATRIN-Experiment/Kassiopeia/tree/main/Kassiopeia/Bindings/Generators/Source/KSGenPositionCylindricalCompositeBuilder.cxx)<br>[*KSGenPositionFluxTubeBuilder.cxx*](https://github.com/KATRIN-Experiment/Kassiopeia/tree/main/Kassiopeia/Bindings/Generators/Source/KSGenPositionFluxTubeBuilder.cxx)<br>[*KSGenPositionFrustrumCompositeBuilder.cxx*](https://github.com/KATRIN-Experiment/Kassiopeia/tree/main/Kassiopeia/Bindings/Generators/Source/KSGenPositionFrustrumCompositeBuilder.cxx)<br>[*KSGenPositionRectangularCompositeBuilder.cxx*](https://github.com/KATRIN-Experiment/Kassiopeia/tree/main/Kassiopeia/Bindings/Generators/Source/KSGenPositionRectangularCompositeBuilder.cxx)<br>[*KSGenPositionSphericalCompositeBuilder.cxx*](https://github.com/KATRIN-Experiment/Kassiopeia/tree/main/Kassiopeia/Bindings/Generators/Source/KSGenPositionSphericalCompositeBuilder.cxx)<br>[*KSGenSpinCompositeBuilder.cxx*](https://github.com/KATRIN-Experiment/Kassiopeia/tree/main/Kassiopeia/Bindings/Generators/Source/KSGenSpinCompositeBuilder.cxx)<br>[*KSGenSpinRelativeCompositeBuilder.cxx*](https://github.com/KATRIN-Experiment/Kassiopeia/tree/main/Kassiopeia/Bindings/Generators/Source/KSGenSpinRelativeCompositeBuilder.cxx)<br>[*KSGenTimeCompositeBuilder.cxx*](https://github.com/KATRIN-Experiment/Kassiopeia/tree/main/Kassiopeia/Bindings/Generators/Source/KSGenTimeCompositeBuilder.cxx)<br>[*KSGenValueFormulaBuilder.cxx*](https://github.com/KATRIN-Experiment/Kassiopeia/tree/main/Kassiopeia/Bindings/Generators/Source/KSGenValueFormulaBuilder.cxx)|—    |—    |`name`<br>`value_formula`<br>`value_max`<br>`value_min`|*`string`*<br>*`string`*<br>*`double`*<br>*`double`*|

#### KSGenValueGauss

|element name|source files|child elements|child types|attributes|attribute types|
|-----|-----|-----|-----|-----|-----|
|<a name="ksgenvaluegauss">`phi_gauss`</a>|[*KSGenDirectionSphericalCompositeBuilder.cxx*](https://github.com/KATRIN-Experiment/Kassiopeia/tree/main/Kassiopeia/Bindings/Generators/Source/KSGenDirectionSphericalCompositeBuilder.cxx)<br>[*KSGenDirectionSphericalMagneticFieldBuilder.cxx*](https://github.com/KATRIN-Experiment/Kassiopeia/tree/main/Kassiopeia/Bindings/Generators/Source/KSGenDirectionSphericalMagneticFieldBuilder.cxx)<br>[*KSGenDirectionSurfaceCompositeBuilder.cxx*](https://github.com/KATRIN-Experiment/Kassiopeia/tree/main/Kassiopeia/Bindings/Generators/Source/KSGenDirectionSurfaceCompositeBuilder.cxx)<br>[*KSGenEnergyCompositeBuilder.cxx*](https://github.com/KATRIN-Experiment/Kassiopeia/tree/main/Kassiopeia/Bindings/Generators/Source/KSGenEnergyCompositeBuilder.cxx)<br>[*KSGenGeneratorCompositeBuilder.cxx*](https://github.com/KATRIN-Experiment/Kassiopeia/tree/main/Kassiopeia/Bindings/Generators/Source/KSGenGeneratorCompositeBuilder.cxx)<br>[*KSGenLCompositeBuilder.cxx*](https://github.com/KATRIN-Experiment/Kassiopeia/tree/main/Kassiopeia/Bindings/Generators/Source/KSGenLCompositeBuilder.cxx)<br>[*KSGenMomentumRectangularCompositeBuilder.cxx*](https://github.com/KATRIN-Experiment/Kassiopeia/tree/main/Kassiopeia/Bindings/Generators/Source/KSGenMomentumRectangularCompositeBuilder.cxx)<br>[*KSGenNCompositeBuilder.cxx*](https://github.com/KATRIN-Experiment/Kassiopeia/tree/main/Kassiopeia/Bindings/Generators/Source/KSGenNCompositeBuilder.cxx)<br>[*KSGenPositionCylindricalCompositeBuilder.cxx*](https://github.com/KATRIN-Experiment/Kassiopeia/tree/main/Kassiopeia/Bindings/Generators/Source/KSGenPositionCylindricalCompositeBuilder.cxx)<br>[*KSGenPositionFluxTubeBuilder.cxx*](https://github.com/KATRIN-Experiment/Kassiopeia/tree/main/Kassiopeia/Bindings/Generators/Source/KSGenPositionFluxTubeBuilder.cxx)<br>[*KSGenPositionFrustrumCompositeBuilder.cxx*](https://github.com/KATRIN-Experiment/Kassiopeia/tree/main/Kassiopeia/Bindings/Generators/Source/KSGenPositionFrustrumCompositeBuilder.cxx)<br>[*KSGenPositionRectangularCompositeBuilder.cxx*](https://github.com/KATRIN-Experiment/Kassiopeia/tree/main/Kassiopeia/Bindings/Generators/Source/KSGenPositionRectangularCompositeBuilder.cxx)<br>[*KSGenPositionSphericalCompositeBuilder.cxx*](https://github.com/KATRIN-Experiment/Kassiopeia/tree/main/Kassiopeia/Bindings/Generators/Source/KSGenPositionSphericalCompositeBuilder.cxx)<br>[*KSGenSpinCompositeBuilder.cxx*](https://github.com/KATRIN-Experiment/Kassiopeia/tree/main/Kassiopeia/Bindings/Generators/Source/KSGenSpinCompositeBuilder.cxx)<br>[*KSGenSpinRelativeCompositeBuilder.cxx*](https://github.com/KATRIN-Experiment/Kassiopeia/tree/main/Kassiopeia/Bindings/Generators/Source/KSGenSpinRelativeCompositeBuilder.cxx)<br>[*KSGenTimeCompositeBuilder.cxx*](https://github.com/KATRIN-Experiment/Kassiopeia/tree/main/Kassiopeia/Bindings/Generators/Source/KSGenTimeCompositeBuilder.cxx)<br>[*KSGenValueGaussBuilder.cxx*](https://github.com/KATRIN-Experiment/Kassiopeia/tree/main/Kassiopeia/Bindings/Generators/Source/KSGenValueGaussBuilder.cxx)|—    |—    |`name`<br>`value_max`<br>`value_mean`<br>`value_min`<br>`value_sigma`|*`string`*<br>*`double`*<br>*`double`*<br>*`double`*<br>*`double`*|

#### KSGenValueGeneralizedGauss

|element name|source files|child elements|child types|attributes|attribute types|
|-----|-----|-----|-----|-----|-----|
|<a name="ksgenvaluegeneralizedgauss">`phi_generalized_gauss`</a>|[*KSGenDirectionSphericalCompositeBuilder.cxx*](https://github.com/KATRIN-Experiment/Kassiopeia/tree/main/Kassiopeia/Bindings/Generators/Source/KSGenDirectionSphericalCompositeBuilder.cxx)<br>[*KSGenDirectionSphericalMagneticFieldBuilder.cxx*](https://github.com/KATRIN-Experiment/Kassiopeia/tree/main/Kassiopeia/Bindings/Generators/Source/KSGenDirectionSphericalMagneticFieldBuilder.cxx)<br>[*KSGenDirectionSurfaceCompositeBuilder.cxx*](https://github.com/KATRIN-Experiment/Kassiopeia/tree/main/Kassiopeia/Bindings/Generators/Source/KSGenDirectionSurfaceCompositeBuilder.cxx)<br>[*KSGenEnergyCompositeBuilder.cxx*](https://github.com/KATRIN-Experiment/Kassiopeia/tree/main/Kassiopeia/Bindings/Generators/Source/KSGenEnergyCompositeBuilder.cxx)<br>[*KSGenSpinCompositeBuilder.cxx*](https://github.com/KATRIN-Experiment/Kassiopeia/tree/main/Kassiopeia/Bindings/Generators/Source/KSGenSpinCompositeBuilder.cxx)<br>[*KSGenSpinRelativeCompositeBuilder.cxx*](https://github.com/KATRIN-Experiment/Kassiopeia/tree/main/Kassiopeia/Bindings/Generators/Source/KSGenSpinRelativeCompositeBuilder.cxx)<br>[*KSGenTimeCompositeBuilder.cxx*](https://github.com/KATRIN-Experiment/Kassiopeia/tree/main/Kassiopeia/Bindings/Generators/Source/KSGenTimeCompositeBuilder.cxx)<br>[*KSGenValueGeneralizedGaussBuilder.cxx*](https://github.com/KATRIN-Experiment/Kassiopeia/tree/main/Kassiopeia/Bindings/Generators/Source/KSGenValueGeneralizedGaussBuilder.cxx)|—    |—    |`name`<br>`value_max`<br>`value_mean`<br>`value_min`<br>`value_sigma`<br>`value_skew`|*`string`*<br>*`double`*<br>*`double`*<br>*`double`*<br>*`double`*<br>*`double`*|

#### KSGenValueHistogram

|element name|source files|child elements|child types|attributes|attribute types|
|-----|-----|-----|-----|-----|-----|
|<a name="ksgenvaluehistogram">`phi_histogram`</a>|[*KSGenDirectionSphericalCompositeBuilder.cxx*](https://github.com/KATRIN-Experiment/Kassiopeia/tree/main/Kassiopeia/Bindings/Generators/Source/KSGenDirectionSphericalCompositeBuilder.cxx)<br>[*KSGenDirectionSphericalMagneticFieldBuilder.cxx*](https://github.com/KATRIN-Experiment/Kassiopeia/tree/main/Kassiopeia/Bindings/Generators/Source/KSGenDirectionSphericalMagneticFieldBuilder.cxx)<br>[*KSGenDirectionSurfaceCompositeBuilder.cxx*](https://github.com/KATRIN-Experiment/Kassiopeia/tree/main/Kassiopeia/Bindings/Generators/Source/KSGenDirectionSurfaceCompositeBuilder.cxx)<br>[*KSGenEnergyCompositeBuilder.cxx*](https://github.com/KATRIN-Experiment/Kassiopeia/tree/main/Kassiopeia/Bindings/Generators/Source/KSGenEnergyCompositeBuilder.cxx)<br>[*KSGenGeneratorCompositeBuilder.cxx*](https://github.com/KATRIN-Experiment/Kassiopeia/tree/main/Kassiopeia/Bindings/Generators/Source/KSGenGeneratorCompositeBuilder.cxx)<br>[*KSGenMomentumRectangularCompositeBuilder.cxx*](https://github.com/KATRIN-Experiment/Kassiopeia/tree/main/Kassiopeia/Bindings/Generators/Source/KSGenMomentumRectangularCompositeBuilder.cxx)<br>[*KSGenPositionCylindricalCompositeBuilder.cxx*](https://github.com/KATRIN-Experiment/Kassiopeia/tree/main/Kassiopeia/Bindings/Generators/Source/KSGenPositionCylindricalCompositeBuilder.cxx)<br>[*KSGenPositionFrustrumCompositeBuilder.cxx*](https://github.com/KATRIN-Experiment/Kassiopeia/tree/main/Kassiopeia/Bindings/Generators/Source/KSGenPositionFrustrumCompositeBuilder.cxx)<br>[*KSGenPositionRectangularCompositeBuilder.cxx*](https://github.com/KATRIN-Experiment/Kassiopeia/tree/main/Kassiopeia/Bindings/Generators/Source/KSGenPositionRectangularCompositeBuilder.cxx)<br>[*KSGenPositionSphericalCompositeBuilder.cxx*](https://github.com/KATRIN-Experiment/Kassiopeia/tree/main/Kassiopeia/Bindings/Generators/Source/KSGenPositionSphericalCompositeBuilder.cxx)<br>[*KSGenSpinCompositeBuilder.cxx*](https://github.com/KATRIN-Experiment/Kassiopeia/tree/main/Kassiopeia/Bindings/Generators/Source/KSGenSpinCompositeBuilder.cxx)<br>[*KSGenSpinRelativeCompositeBuilder.cxx*](https://github.com/KATRIN-Experiment/Kassiopeia/tree/main/Kassiopeia/Bindings/Generators/Source/KSGenSpinRelativeCompositeBuilder.cxx)<br>[*KSGenTimeCompositeBuilder.cxx*](https://github.com/KATRIN-Experiment/Kassiopeia/tree/main/Kassiopeia/Bindings/Generators/Source/KSGenTimeCompositeBuilder.cxx)<br>[*KSGenValueHistogramBuilder.cxx*](https://github.com/KATRIN-Experiment/Kassiopeia/tree/main/Kassiopeia/Bindings/Generators/Source/KSGenValueHistogramBuilder.cxx)|—    |—    |`base`<br>`formula`<br>`histogram`<br>`name`<br>`path`|*`string`*<br>*`string`*<br>*`string`*<br>*`string`*<br>*`string`*|

#### KSGenValueList

|element name|source files|child elements|child types|attributes|attribute types|
|-----|-----|-----|-----|-----|-----|
|<a name="ksgenvaluelist">`phi_list`</a>|[*KSGenDirectionSphericalCompositeBuilder.cxx*](https://github.com/KATRIN-Experiment/Kassiopeia/tree/main/Kassiopeia/Bindings/Generators/Source/KSGenDirectionSphericalCompositeBuilder.cxx)<br>[*KSGenDirectionSphericalMagneticFieldBuilder.cxx*](https://github.com/KATRIN-Experiment/Kassiopeia/tree/main/Kassiopeia/Bindings/Generators/Source/KSGenDirectionSphericalMagneticFieldBuilder.cxx)<br>[*KSGenDirectionSurfaceCompositeBuilder.cxx*](https://github.com/KATRIN-Experiment/Kassiopeia/tree/main/Kassiopeia/Bindings/Generators/Source/KSGenDirectionSurfaceCompositeBuilder.cxx)<br>[*KSGenEnergyCompositeBuilder.cxx*](https://github.com/KATRIN-Experiment/Kassiopeia/tree/main/Kassiopeia/Bindings/Generators/Source/KSGenEnergyCompositeBuilder.cxx)<br>[*KSGenGeneratorCompositeBuilder.cxx*](https://github.com/KATRIN-Experiment/Kassiopeia/tree/main/Kassiopeia/Bindings/Generators/Source/KSGenGeneratorCompositeBuilder.cxx)<br>[*KSGenLCompositeBuilder.cxx*](https://github.com/KATRIN-Experiment/Kassiopeia/tree/main/Kassiopeia/Bindings/Generators/Source/KSGenLCompositeBuilder.cxx)<br>[*KSGenMomentumRectangularCompositeBuilder.cxx*](https://github.com/KATRIN-Experiment/Kassiopeia/tree/main/Kassiopeia/Bindings/Generators/Source/KSGenMomentumRectangularCompositeBuilder.cxx)<br>[*KSGenNCompositeBuilder.cxx*](https://github.com/KATRIN-Experiment/Kassiopeia/tree/main/Kassiopeia/Bindings/Generators/Source/KSGenNCompositeBuilder.cxx)<br>[*KSGenPositionCylindricalCompositeBuilder.cxx*](https://github.com/KATRIN-Experiment/Kassiopeia/tree/main/Kassiopeia/Bindings/Generators/Source/KSGenPositionCylindricalCompositeBuilder.cxx)<br>[*KSGenPositionFrustrumCompositeBuilder.cxx*](https://github.com/KATRIN-Experiment/Kassiopeia/tree/main/Kassiopeia/Bindings/Generators/Source/KSGenPositionFrustrumCompositeBuilder.cxx)<br>[*KSGenPositionRectangularCompositeBuilder.cxx*](https://github.com/KATRIN-Experiment/Kassiopeia/tree/main/Kassiopeia/Bindings/Generators/Source/KSGenPositionRectangularCompositeBuilder.cxx)<br>[*KSGenPositionSphericalCompositeBuilder.cxx*](https://github.com/KATRIN-Experiment/Kassiopeia/tree/main/Kassiopeia/Bindings/Generators/Source/KSGenPositionSphericalCompositeBuilder.cxx)<br>[*KSGenSpinCompositeBuilder.cxx*](https://github.com/KATRIN-Experiment/Kassiopeia/tree/main/Kassiopeia/Bindings/Generators/Source/KSGenSpinCompositeBuilder.cxx)<br>[*KSGenSpinRelativeCompositeBuilder.cxx*](https://github.com/KATRIN-Experiment/Kassiopeia/tree/main/Kassiopeia/Bindings/Generators/Source/KSGenSpinRelativeCompositeBuilder.cxx)<br>[*KSGenTimeCompositeBuilder.cxx*](https://github.com/KATRIN-Experiment/Kassiopeia/tree/main/Kassiopeia/Bindings/Generators/Source/KSGenTimeCompositeBuilder.cxx)<br>[*KSGenValueListBuilder.cxx*](https://github.com/KATRIN-Experiment/Kassiopeia/tree/main/Kassiopeia/Bindings/Generators/Source/KSGenValueListBuilder.cxx)|—    |—    |`add_value`<br>`name`<br>`randomize`|*`double`*<br>*`string`*<br>*`bool`*|

#### KSGenValueSet

|element name|source files|child elements|child types|attributes|attribute types|
|-----|-----|-----|-----|-----|-----|
|<a name="ksgenvalueset">`phi_set`</a>|[*KSGenDirectionSphericalCompositeBuilder.cxx*](https://github.com/KATRIN-Experiment/Kassiopeia/tree/main/Kassiopeia/Bindings/Generators/Source/KSGenDirectionSphericalCompositeBuilder.cxx)<br>[*KSGenDirectionSphericalMagneticFieldBuilder.cxx*](https://github.com/KATRIN-Experiment/Kassiopeia/tree/main/Kassiopeia/Bindings/Generators/Source/KSGenDirectionSphericalMagneticFieldBuilder.cxx)<br>[*KSGenDirectionSurfaceCompositeBuilder.cxx*](https://github.com/KATRIN-Experiment/Kassiopeia/tree/main/Kassiopeia/Bindings/Generators/Source/KSGenDirectionSurfaceCompositeBuilder.cxx)<br>[*KSGenEnergyCompositeBuilder.cxx*](https://github.com/KATRIN-Experiment/Kassiopeia/tree/main/Kassiopeia/Bindings/Generators/Source/KSGenEnergyCompositeBuilder.cxx)<br>[*KSGenGeneratorCompositeBuilder.cxx*](https://github.com/KATRIN-Experiment/Kassiopeia/tree/main/Kassiopeia/Bindings/Generators/Source/KSGenGeneratorCompositeBuilder.cxx)<br>[*KSGenLCompositeBuilder.cxx*](https://github.com/KATRIN-Experiment/Kassiopeia/tree/main/Kassiopeia/Bindings/Generators/Source/KSGenLCompositeBuilder.cxx)<br>[*KSGenMomentumRectangularCompositeBuilder.cxx*](https://github.com/KATRIN-Experiment/Kassiopeia/tree/main/Kassiopeia/Bindings/Generators/Source/KSGenMomentumRectangularCompositeBuilder.cxx)<br>[*KSGenNCompositeBuilder.cxx*](https://github.com/KATRIN-Experiment/Kassiopeia/tree/main/Kassiopeia/Bindings/Generators/Source/KSGenNCompositeBuilder.cxx)<br>[*KSGenPositionCylindricalCompositeBuilder.cxx*](https://github.com/KATRIN-Experiment/Kassiopeia/tree/main/Kassiopeia/Bindings/Generators/Source/KSGenPositionCylindricalCompositeBuilder.cxx)<br>[*KSGenPositionFluxTubeBuilder.cxx*](https://github.com/KATRIN-Experiment/Kassiopeia/tree/main/Kassiopeia/Bindings/Generators/Source/KSGenPositionFluxTubeBuilder.cxx)<br>[*KSGenPositionFrustrumCompositeBuilder.cxx*](https://github.com/KATRIN-Experiment/Kassiopeia/tree/main/Kassiopeia/Bindings/Generators/Source/KSGenPositionFrustrumCompositeBuilder.cxx)<br>[*KSGenPositionRectangularCompositeBuilder.cxx*](https://github.com/KATRIN-Experiment/Kassiopeia/tree/main/Kassiopeia/Bindings/Generators/Source/KSGenPositionRectangularCompositeBuilder.cxx)<br>[*KSGenPositionSphericalCompositeBuilder.cxx*](https://github.com/KATRIN-Experiment/Kassiopeia/tree/main/Kassiopeia/Bindings/Generators/Source/KSGenPositionSphericalCompositeBuilder.cxx)<br>[*KSGenSpinCompositeBuilder.cxx*](https://github.com/KATRIN-Experiment/Kassiopeia/tree/main/Kassiopeia/Bindings/Generators/Source/KSGenSpinCompositeBuilder.cxx)<br>[*KSGenSpinRelativeCompositeBuilder.cxx*](https://github.com/KATRIN-Experiment/Kassiopeia/tree/main/Kassiopeia/Bindings/Generators/Source/KSGenSpinRelativeCompositeBuilder.cxx)<br>[*KSGenTimeCompositeBuilder.cxx*](https://github.com/KATRIN-Experiment/Kassiopeia/tree/main/Kassiopeia/Bindings/Generators/Source/KSGenTimeCompositeBuilder.cxx)<br>[*KSGenValueSetBuilder.cxx*](https://github.com/KATRIN-Experiment/Kassiopeia/tree/main/Kassiopeia/Bindings/Generators/Source/KSGenValueSetBuilder.cxx)|—    |—    |`name`<br>`value_count`<br>`value_increment`<br>`value_start`<br>`value_stop`|*`string`*<br>*`unsigned int`*<br>*`double`*<br>*`double`*<br>*`double`*|

#### KSGenValueUniform

|element name|source files|child elements|child types|attributes|attribute types|
|-----|-----|-----|-----|-----|-----|
|<a name="ksgenvalueuniform">`phi_uniform`</a>|[*KSGenDirectionSphericalCompositeBuilder.cxx*](https://github.com/KATRIN-Experiment/Kassiopeia/tree/main/Kassiopeia/Bindings/Generators/Source/KSGenDirectionSphericalCompositeBuilder.cxx)<br>[*KSGenDirectionSphericalMagneticFieldBuilder.cxx*](https://github.com/KATRIN-Experiment/Kassiopeia/tree/main/Kassiopeia/Bindings/Generators/Source/KSGenDirectionSphericalMagneticFieldBuilder.cxx)<br>[*KSGenDirectionSurfaceCompositeBuilder.cxx*](https://github.com/KATRIN-Experiment/Kassiopeia/tree/main/Kassiopeia/Bindings/Generators/Source/KSGenDirectionSurfaceCompositeBuilder.cxx)<br>[*KSGenEnergyCompositeBuilder.cxx*](https://github.com/KATRIN-Experiment/Kassiopeia/tree/main/Kassiopeia/Bindings/Generators/Source/KSGenEnergyCompositeBuilder.cxx)<br>[*KSGenGeneratorCompositeBuilder.cxx*](https://github.com/KATRIN-Experiment/Kassiopeia/tree/main/Kassiopeia/Bindings/Generators/Source/KSGenGeneratorCompositeBuilder.cxx)<br>[*KSGenLCompositeBuilder.cxx*](https://github.com/KATRIN-Experiment/Kassiopeia/tree/main/Kassiopeia/Bindings/Generators/Source/KSGenLCompositeBuilder.cxx)<br>[*KSGenMomentumRectangularCompositeBuilder.cxx*](https://github.com/KATRIN-Experiment/Kassiopeia/tree/main/Kassiopeia/Bindings/Generators/Source/KSGenMomentumRectangularCompositeBuilder.cxx)<br>[*KSGenNCompositeBuilder.cxx*](https://github.com/KATRIN-Experiment/Kassiopeia/tree/main/Kassiopeia/Bindings/Generators/Source/KSGenNCompositeBuilder.cxx)<br>[*KSGenPositionCylindricalCompositeBuilder.cxx*](https://github.com/KATRIN-Experiment/Kassiopeia/tree/main/Kassiopeia/Bindings/Generators/Source/KSGenPositionCylindricalCompositeBuilder.cxx)<br>[*KSGenPositionFluxTubeBuilder.cxx*](https://github.com/KATRIN-Experiment/Kassiopeia/tree/main/Kassiopeia/Bindings/Generators/Source/KSGenPositionFluxTubeBuilder.cxx)<br>[*KSGenPositionFrustrumCompositeBuilder.cxx*](https://github.com/KATRIN-Experiment/Kassiopeia/tree/main/Kassiopeia/Bindings/Generators/Source/KSGenPositionFrustrumCompositeBuilder.cxx)<br>[*KSGenPositionRectangularCompositeBuilder.cxx*](https://github.com/KATRIN-Experiment/Kassiopeia/tree/main/Kassiopeia/Bindings/Generators/Source/KSGenPositionRectangularCompositeBuilder.cxx)<br>[*KSGenPositionSphericalCompositeBuilder.cxx*](https://github.com/KATRIN-Experiment/Kassiopeia/tree/main/Kassiopeia/Bindings/Generators/Source/KSGenPositionSphericalCompositeBuilder.cxx)<br>[*KSGenSpinCompositeBuilder.cxx*](https://github.com/KATRIN-Experiment/Kassiopeia/tree/main/Kassiopeia/Bindings/Generators/Source/KSGenSpinCompositeBuilder.cxx)<br>[*KSGenSpinRelativeCompositeBuilder.cxx*](https://github.com/KATRIN-Experiment/Kassiopeia/tree/main/Kassiopeia/Bindings/Generators/Source/KSGenSpinRelativeCompositeBuilder.cxx)<br>[*KSGenTimeCompositeBuilder.cxx*](https://github.com/KATRIN-Experiment/Kassiopeia/tree/main/Kassiopeia/Bindings/Generators/Source/KSGenTimeCompositeBuilder.cxx)<br>[*KSGenValueUniformBuilder.cxx*](https://github.com/KATRIN-Experiment/Kassiopeia/tree/main/Kassiopeia/Bindings/Generators/Source/KSGenValueUniformBuilder.cxx)|—    |—    |`name`<br>`value_max`<br>`value_min`|*`string`*<br>*`double`*<br>*`double`*|

#### KSGenValueAngleCosine

|element name|source files|child elements|child types|attributes|attribute types|
|-----|-----|-----|-----|-----|-----|
|<a name="ksgenvalueanglecosine">`theta_cosine`</a>|[*KSGenDirectionSphericalCompositeBuilder.cxx*](https://github.com/KATRIN-Experiment/Kassiopeia/tree/main/Kassiopeia/Bindings/Generators/Source/KSGenDirectionSphericalCompositeBuilder.cxx)<br>[*KSGenDirectionSphericalMagneticFieldBuilder.cxx*](https://github.com/KATRIN-Experiment/Kassiopeia/tree/main/Kassiopeia/Bindings/Generators/Source/KSGenDirectionSphericalMagneticFieldBuilder.cxx)<br>[*KSGenDirectionSurfaceCompositeBuilder.cxx*](https://github.com/KATRIN-Experiment/Kassiopeia/tree/main/Kassiopeia/Bindings/Generators/Source/KSGenDirectionSurfaceCompositeBuilder.cxx)<br>[*KSGenSpinCompositeBuilder.cxx*](https://github.com/KATRIN-Experiment/Kassiopeia/tree/main/Kassiopeia/Bindings/Generators/Source/KSGenSpinCompositeBuilder.cxx)<br>[*KSGenSpinRelativeCompositeBuilder.cxx*](https://github.com/KATRIN-Experiment/Kassiopeia/tree/main/Kassiopeia/Bindings/Generators/Source/KSGenSpinRelativeCompositeBuilder.cxx)<br>[*KSGenValueAngleCosineBuilder.cxx*](https://github.com/KATRIN-Experiment/Kassiopeia/tree/main/Kassiopeia/Bindings/Generators/Source/KSGenValueAngleCosineBuilder.cxx)|—    |—    |`angle_max`<br>`angle_min`<br>`mode`<br>`name`|*`double`*<br>*`double`*<br>*`string`*<br>*`string`*|

#### KSGenValueAngleSpherical

|element name|source files|child elements|child types|attributes|attribute types|
|-----|-----|-----|-----|-----|-----|
|<a name="ksgenvalueanglespherical">`theta_spherical`</a>|[*KSGenDirectionSphericalCompositeBuilder.cxx*](https://github.com/KATRIN-Experiment/Kassiopeia/tree/main/Kassiopeia/Bindings/Generators/Source/KSGenDirectionSphericalCompositeBuilder.cxx)<br>[*KSGenDirectionSphericalMagneticFieldBuilder.cxx*](https://github.com/KATRIN-Experiment/Kassiopeia/tree/main/Kassiopeia/Bindings/Generators/Source/KSGenDirectionSphericalMagneticFieldBuilder.cxx)<br>[*KSGenDirectionSurfaceCompositeBuilder.cxx*](https://github.com/KATRIN-Experiment/Kassiopeia/tree/main/Kassiopeia/Bindings/Generators/Source/KSGenDirectionSurfaceCompositeBuilder.cxx)<br>[*KSGenPositionSphericalCompositeBuilder.cxx*](https://github.com/KATRIN-Experiment/Kassiopeia/tree/main/Kassiopeia/Bindings/Generators/Source/KSGenPositionSphericalCompositeBuilder.cxx)<br>[*KSGenSpinCompositeBuilder.cxx*](https://github.com/KATRIN-Experiment/Kassiopeia/tree/main/Kassiopeia/Bindings/Generators/Source/KSGenSpinCompositeBuilder.cxx)<br>[*KSGenSpinRelativeCompositeBuilder.cxx*](https://github.com/KATRIN-Experiment/Kassiopeia/tree/main/Kassiopeia/Bindings/Generators/Source/KSGenSpinRelativeCompositeBuilder.cxx)<br>[*KSGenValueAngleSphericalBuilder.cxx*](https://github.com/KATRIN-Experiment/Kassiopeia/tree/main/Kassiopeia/Bindings/Generators/Source/KSGenValueAngleSphericalBuilder.cxx)|—    |—    |`angle_max`<br>`angle_min`<br>`name`|*`double`*<br>*`double`*<br>*`string`*|

### KSGenDirectionSphericalMagneticField

|element name|source files|child elements|child types|attributes|attribute types|
|-----|-----|-----|-----|-----|-----|
|<a name="ksgendirectionsphericalmagneticfield">`ksgen_direction_spherical_magnetic_field`</a>|[*KSGenDirectionSphericalMagneticFieldBuilder.cxx*](https://github.com/KATRIN-Experiment/Kassiopeia/tree/main/Kassiopeia/Bindings/Generators/Source/KSGenDirectionSphericalMagneticFieldBuilder.cxx)<br>[*KSGenGeneratorCompositeBuilder.cxx*](https://github.com/KATRIN-Experiment/Kassiopeia/tree/main/Kassiopeia/Bindings/Generators/Source/KSGenGeneratorCompositeBuilder.cxx)|[`phi_fix`](#ksgenvaluefix)<br>[`phi_formula`](#ksgenvalueformula)<br>[`phi_gauss`](#ksgenvaluegauss)<br>[`phi_generalized_gauss`](#ksgenvaluegeneralizedgauss)<br>[`phi_histogram`](#ksgenvaluehistogram)<br>[`phi_list`](#ksgenvaluelist)<br>[`phi_set`](#ksgenvalueset)<br>[`phi_uniform`](#ksgenvalueuniform)<br>[`theta_cosine`](#ksgenvalueanglecosine)<br>[`theta_fix`](#ksgenvaluefix)<br>[`theta_formula`](#ksgenvalueformula)<br>[`theta_gauss`](#ksgenvaluegauss)<br>[`theta_generalized_gauss`](#ksgenvaluegeneralizedgauss)<br>[`theta_histogram`](#ksgenvaluehistogram)<br>[`theta_list`](#ksgenvaluelist)<br>[`theta_set`](#ksgenvalueset)<br>[`theta_spherical`](#ksgenvalueanglespherical)<br>[`theta_uniform`](#ksgenvalueuniform)|*`KSGenValueFix`*<br>*`KSGenValueFormula`*<br>*`KSGenValueGauss`*<br>*`KSGenValueGeneralizedGauss`*<br>*`KSGenValueHistogram`*<br>*`KSGenValueList`*<br>*`KSGenValueSet`*<br>*`KSGenValueUniform`*<br>*`KSGenValueAngleCosine`*<br>*`KSGenValueFix`*<br>*`KSGenValueFormula`*<br>*`KSGenValueGauss`*<br>*`KSGenValueGeneralizedGauss`*<br>*`KSGenValueHistogram`*<br>*`KSGenValueList`*<br>*`KSGenValueSet`*<br>*`KSGenValueAngleSpherical`*<br>*`KSGenValueUniform`*|`magnetic_field_name`<br>`name`<br>`phi`<br>`theta`|*`string`*<br>*`string`*<br>*`string`*<br>*`string`*|

### KSGenDirectionSurfaceComposite

|element name|source files|child elements|child types|attributes|attribute types|
|-----|-----|-----|-----|-----|-----|
|<a name="ksgendirectionsurfacecomposite">`ksgen_direction_surface_composite`</a>|[*KSGenDirectionSurfaceCompositeBuilder.cxx*](https://github.com/KATRIN-Experiment/Kassiopeia/tree/main/Kassiopeia/Bindings/Generators/Source/KSGenDirectionSurfaceCompositeBuilder.cxx)<br>[*KSGenGeneratorCompositeBuilder.cxx*](https://github.com/KATRIN-Experiment/Kassiopeia/tree/main/Kassiopeia/Bindings/Generators/Source/KSGenGeneratorCompositeBuilder.cxx)|[`phi_fix`](#ksgenvaluefix)<br>[`phi_formula`](#ksgenvalueformula)<br>[`phi_gauss`](#ksgenvaluegauss)<br>[`phi_generalized_gauss`](#ksgenvaluegeneralizedgauss)<br>[`phi_histogram`](#ksgenvaluehistogram)<br>[`phi_list`](#ksgenvaluelist)<br>[`phi_set`](#ksgenvalueset)<br>[`phi_uniform`](#ksgenvalueuniform)<br>[`theta_cosine`](#ksgenvalueanglecosine)<br>[`theta_fix`](#ksgenvaluefix)<br>[`theta_formula`](#ksgenvalueformula)<br>[`theta_gauss`](#ksgenvaluegauss)<br>[`theta_generalized_gauss`](#ksgenvaluegeneralizedgauss)<br>[`theta_histogram`](#ksgenvaluehistogram)<br>[`theta_list`](#ksgenvaluelist)<br>[`theta_set`](#ksgenvalueset)<br>[`theta_spherical`](#ksgenvalueanglespherical)<br>[`theta_uniform`](#ksgenvalueuniform)|*`KSGenValueFix`*<br>*`KSGenValueFormula`*<br>*`KSGenValueGauss`*<br>*`KSGenValueGeneralizedGauss`*<br>*`KSGenValueHistogram`*<br>*`KSGenValueList`*<br>*`KSGenValueSet`*<br>*`KSGenValueUniform`*<br>*`KSGenValueAngleCosine`*<br>*`KSGenValueFix`*<br>*`KSGenValueFormula`*<br>*`KSGenValueGauss`*<br>*`KSGenValueGeneralizedGauss`*<br>*`KSGenValueHistogram`*<br>*`KSGenValueList`*<br>*`KSGenValueSet`*<br>*`KSGenValueAngleSpherical`*<br>*`KSGenValueUniform`*|`name`<br>`outside`<br>`phi`<br>`surfaces`<br>`theta`|*`string`*<br>*`bool`*<br>*`string`*<br>*`string`*<br>*`string`*|

### KSGenEnergyBetaDecay

|element name|source files|child elements|child types|attributes|attribute types|
|-----|-----|-----|-----|-----|-----|
|<a name="ksgenenergybetadecay">`ksgen_energy_beta_decay`</a>|[*KSGenEnergyBetaDecayBuilder.cxx*](https://github.com/KATRIN-Experiment/Kassiopeia/tree/main/Kassiopeia/Bindings/Generators/Source/KSGenEnergyBetaDecayBuilder.cxx)<br>[*KSGenGeneratorCompositeBuilder.cxx*](https://github.com/KATRIN-Experiment/Kassiopeia/tree/main/Kassiopeia/Bindings/Generators/Source/KSGenGeneratorCompositeBuilder.cxx)|—    |—    |`daughter_z`<br>`endpoint_ev`<br>`max_energy`<br>`min_energy`<br>`mnu_ev`<br>`name`<br>`nmax`|*`int`*<br>*`double`*<br>*`double`*<br>*`double`*<br>*`double`*<br>*`string`*<br>*`int`*|

### KSGenEnergyBetaRecoil

|element name|source files|child elements|child types|attributes|attribute types|
|-----|-----|-----|-----|-----|-----|
|<a name="ksgenenergybetarecoil">`ksgen_energy_beta_recoil`</a>|[*KSGenEnergyBetaRecoilBuilder.cxx*](https://github.com/KATRIN-Experiment/Kassiopeia/tree/main/Kassiopeia/Bindings/Generators/Source/KSGenEnergyBetaRecoilBuilder.cxx)<br>[*KSGenGeneratorCompositeBuilder.cxx*](https://github.com/KATRIN-Experiment/Kassiopeia/tree/main/Kassiopeia/Bindings/Generators/Source/KSGenGeneratorCompositeBuilder.cxx)|—    |—    |`max_energy`<br>`min_energy`<br>`name`|*`double`*<br>*`double`*<br>*`string`*|

### KSGenEnergyComposite

|element name|source files|child elements|child types|attributes|attribute types|
|-----|-----|-----|-----|-----|-----|
|<a name="ksgenenergycomposite">`ksgen_energy_composite`</a>|[*KSGenEnergyCompositeBuilder.cxx*](https://github.com/KATRIN-Experiment/Kassiopeia/tree/main/Kassiopeia/Bindings/Generators/Source/KSGenEnergyCompositeBuilder.cxx)<br>[*KSGenGeneratorCompositeBuilder.cxx*](https://github.com/KATRIN-Experiment/Kassiopeia/tree/main/Kassiopeia/Bindings/Generators/Source/KSGenGeneratorCompositeBuilder.cxx)|[`energy_boltzmann`](#ksgenvalueboltzmann)<br>[`energy_fix`](#ksgenvaluefix)<br>[`energy_formula`](#ksgenvalueformula)<br>[`energy_gauss`](#ksgenvaluegauss)<br>[`energy_generalized_gauss`](#ksgenvaluegeneralizedgauss)<br>[`energy_histogram`](#ksgenvaluehistogram)<br>[`energy_list`](#ksgenvaluelist)<br>[`energy_set`](#ksgenvalueset)<br>[`energy_uniform`](#ksgenvalueuniform)|*`KSGenValueBoltzmann`*<br>*`KSGenValueFix`*<br>*`KSGenValueFormula`*<br>*`KSGenValueGauss`*<br>*`KSGenValueGeneralizedGauss`*<br>*`KSGenValueHistogram`*<br>*`KSGenValueList`*<br>*`KSGenValueSet`*<br>*`KSGenValueUniform`*|`energy`<br>`name`|*`string`*<br>*`string`*|

#### KSGenValueBoltzmann

|element name|source files|child elements|child types|attributes|attribute types|
|-----|-----|-----|-----|-----|-----|
|<a name="ksgenvalueboltzmann">`energy_boltzmann`</a>|[*KSGenEnergyCompositeBuilder.cxx*](https://github.com/KATRIN-Experiment/Kassiopeia/tree/main/Kassiopeia/Bindings/Generators/Source/KSGenEnergyCompositeBuilder.cxx)<br>[*KSGenValueBoltzmannBuilder.cxx*](https://github.com/KATRIN-Experiment/Kassiopeia/tree/main/Kassiopeia/Bindings/Generators/Source/KSGenValueBoltzmannBuilder.cxx)|—    |—    |`name`<br>`unit_eV`<br>`value_kT`<br>`value_mass`|*`string`*<br>*`bool`*<br>*`double`*<br>*`double`*|

### KSGenEnergyKryptonEvent

|element name|source files|child elements|child types|attributes|attribute types|
|-----|-----|-----|-----|-----|-----|
|<a name="ksgenenergykryptonevent">`ksgen_energy_krypton_event`</a>|[*KSGenEnergyKryptonEventBuilder.cxx*](https://github.com/KATRIN-Experiment/Kassiopeia/tree/main/Kassiopeia/Bindings/Generators/Source/KSGenEnergyKryptonEventBuilder.cxx)<br>[*KSGenGeneratorCompositeBuilder.cxx*](https://github.com/KATRIN-Experiment/Kassiopeia/tree/main/Kassiopeia/Bindings/Generators/Source/KSGenGeneratorCompositeBuilder.cxx)|—    |—    |`do_auger`<br>`do_conversion`<br>`force_conversion`<br>`name`|*`bool`*<br>*`bool`*<br>*`bool`*<br>*`string`*|

### KSGenEnergyLeadEvent

|element name|source files|child elements|child types|attributes|attribute types|
|-----|-----|-----|-----|-----|-----|
|<a name="ksgenenergyleadevent">`ksgen_energy_lead_event`</a>|[*KSGenEnergyLeadEventBuilder.cxx*](https://github.com/KATRIN-Experiment/Kassiopeia/tree/main/Kassiopeia/Bindings/Generators/Source/KSGenEnergyLeadEventBuilder.cxx)<br>[*KSGenGeneratorCompositeBuilder.cxx*](https://github.com/KATRIN-Experiment/Kassiopeia/tree/main/Kassiopeia/Bindings/Generators/Source/KSGenGeneratorCompositeBuilder.cxx)|—    |—    |`do_auger`<br>`do_conversion`<br>`force_conversion`<br>`name`|*`bool`*<br>*`bool`*<br>*`bool`*<br>*`string`*|

### KSGenEnergyRadonEvent

|element name|source files|child elements|child types|attributes|attribute types|
|-----|-----|-----|-----|-----|-----|
|<a name="ksgenenergyradonevent">`ksgen_energy_radon_event`</a>|[*KSGenEnergyRadonEventBuilder.cxx*](https://github.com/KATRIN-Experiment/Kassiopeia/tree/main/Kassiopeia/Bindings/Generators/Source/KSGenEnergyRadonEventBuilder.cxx)<br>[*KSGenGeneratorCompositeBuilder.cxx*](https://github.com/KATRIN-Experiment/Kassiopeia/tree/main/Kassiopeia/Bindings/Generators/Source/KSGenGeneratorCompositeBuilder.cxx)|—    |—    |`do_auger`<br>`do_conversion`<br>`do_shake_off`<br>`force_conversion`<br>`force_shake_off`<br>`isotope_number`<br>`name`|*`bool`*<br>*`bool`*<br>*`bool`*<br>*`bool`*<br>*`bool`*<br>*`int`*<br>*`string`*|

### KSGenEnergyRydberg

|element name|source files|child elements|child types|attributes|attribute types|
|-----|-----|-----|-----|-----|-----|
|<a name="ksgenenergyrydberg">`ksgen_energy_rydberg`</a>|[*KSGenEnergyRydbergBuilder.cxx*](https://github.com/KATRIN-Experiment/Kassiopeia/tree/main/Kassiopeia/Bindings/Generators/Source/KSGenEnergyRydbergBuilder.cxx)<br>[*KSGenGeneratorCompositeBuilder.cxx*](https://github.com/KATRIN-Experiment/Kassiopeia/tree/main/Kassiopeia/Bindings/Generators/Source/KSGenGeneratorCompositeBuilder.cxx)|—    |—    |`deposited_energy`<br>`ionization_energy`<br>`name`|*`double`*<br>*`double`*<br>*`string`*|

### KSGenGeneratorComposite

|element name|source files|child elements|child types|attributes|attribute types|
|-----|-----|-----|-----|-----|-----|
|<a name="ksgengeneratorcomposite">`ksgen_generator_composite`</a>|[*KSGenGeneratorCompositeBuilder.cxx*](https://github.com/KATRIN-Experiment/Kassiopeia/tree/main/Kassiopeia/Bindings/Generators/Source/KSGenGeneratorCompositeBuilder.cxx)<br>[*KSIntDecayCalculatorDeathConstRateBuilder.cxx*](https://github.com/KATRIN-Experiment/Kassiopeia/tree/main/Kassiopeia/Bindings/Interactions/Source/KSIntDecayCalculatorDeathConstRateBuilder.cxx)<br>[*KSIntDecayCalculatorFerencBBRTransitionBuilder.cxx*](https://github.com/KATRIN-Experiment/Kassiopeia/tree/main/Kassiopeia/Bindings/Interactions/Source/KSIntDecayCalculatorFerencBBRTransitionBuilder.cxx)<br>[*KSIntDecayCalculatorFerencIonisationBuilder.cxx*](https://github.com/KATRIN-Experiment/Kassiopeia/tree/main/Kassiopeia/Bindings/Interactions/Source/KSIntDecayCalculatorFerencIonisationBuilder.cxx)<br>[*KSIntDecayCalculatorFerencSpontaneousBuilder.cxx*](https://github.com/KATRIN-Experiment/Kassiopeia/tree/main/Kassiopeia/Bindings/Interactions/Source/KSIntDecayCalculatorFerencSpontaneousBuilder.cxx)<br>[*KSIntDecayCalculatorGlukhovDeExcitationBuilder.cxx*](https://github.com/KATRIN-Experiment/Kassiopeia/tree/main/Kassiopeia/Bindings/Interactions/Source/KSIntDecayCalculatorGlukhovDeExcitationBuilder.cxx)<br>[*KSIntDecayCalculatorGlukhovExcitationBuilder.cxx*](https://github.com/KATRIN-Experiment/Kassiopeia/tree/main/Kassiopeia/Bindings/Interactions/Source/KSIntDecayCalculatorGlukhovExcitationBuilder.cxx)<br>[*KSIntDecayCalculatorGlukhovIonisationBuilder.cxx*](https://github.com/KATRIN-Experiment/Kassiopeia/tree/main/Kassiopeia/Bindings/Interactions/Source/KSIntDecayCalculatorGlukhovIonisationBuilder.cxx)<br>[*KSIntDecayCalculatorGlukhovSpontaneousBuilder.cxx*](https://github.com/KATRIN-Experiment/Kassiopeia/tree/main/Kassiopeia/Bindings/Interactions/Source/KSIntDecayCalculatorGlukhovSpontaneousBuilder.cxx)|[`direction_spherical_composite`](#ksgendirectionsphericalcomposite)<br>[`direction_spherical_magnetic_field`](#ksgendirectionsphericalmagneticfield)<br>[`direction_surface_composite`](#ksgendirectionsurfacecomposite)<br>[`energy_beta_decay`](#ksgenenergybetadecay)<br>[`energy_beta_recoil`](#ksgenenergybetarecoil)<br>[`energy_composite`](#ksgenenergycomposite)<br>[`energy_krypton_event`](#ksgenenergykryptonevent)<br>[`energy_lead_event`](#ksgenenergyleadevent)<br>[`energy_radon_event`](#ksgenenergyradonevent)<br>[`energy_rydberg`](#ksgenenergyrydberg)<br>[`l_composite`](#ksgenlcomposite)<br>[`l_statistical`](#ksgenlstatistical)<br>[`l_uniform_max_n`](#ksgenluniformmaxn)<br>[`momentum_rectangular_composite`](#ksgenmomentumrectangularcomposite)<br>[`n_composite`](#ksgenncomposite)<br>[`pid_fix`](#ksgenvaluefix)<br>[`pid_formula`](#ksgenvalueformula)<br>[`pid_gauss`](#ksgenvaluegauss)<br>[`pid_histogram`](#ksgenvaluehistogram)<br>[`pid_list`](#ksgenvaluelist)<br>[`pid_pareto`](#ksgenvaluepareto)<br>[`pid_set`](#ksgenvalueset)<br>[`pid_uniform`](#ksgenvalueuniform)<br>[`position_cylindrical_composite`](#ksgenpositioncylindricalcomposite)<br>[`position_flux_tube`](#ksgenpositionfluxtube)<br>[`position_frustrum_composite`](#ksgenpositionfrustrumcomposite)<br>[`position_homogeneous_flux_tube`](#ksgenpositionhomogeneousfluxtube)<br>[`position_mask`](#ksgenpositionmask)<br>[`position_mesh_surface_random`](#ksgenpositionmeshsurfacerandom)<br>[`position_rectangular_composite`](#ksgenpositionrectangularcomposite)<br>[`position_space_random`](#ksgenpositionspacerandom)<br>[`position_spherical_composite`](#ksgenpositionsphericalcomposite)<br>[`position_surface_adjustment_step`](#ksgenpositionsurfaceadjustmentstep)<br>[`position_surface_random`](#ksgenpositionsurfacerandom)<br>[`spin_composite`](#ksgenspincomposite)<br>[`spin_relative_composite`](#ksgenspinrelativecomposite)<br>[`time_composite`](#ksgentimecomposite)|*`KSGenDirectionSphericalComposite`*<br>*`KSGenDirectionSphericalMagneticField`*<br>*`KSGenDirectionSurfaceComposite`*<br>*`KSGenEnergyBetaDecay`*<br>*`KSGenEnergyBetaRecoil`*<br>*`KSGenEnergyComposite`*<br>*`KSGenEnergyKryptonEvent`*<br>*`KSGenEnergyLeadEvent`*<br>*`KSGenEnergyRadonEvent`*<br>*`KSGenEnergyRydberg`*<br>*`KSGenLComposite`*<br>*`KSGenLStatistical`*<br>*`KSGenLUniformMaxN`*<br>*`KSGenMomentumRectangularComposite`*<br>*`KSGenNComposite`*<br>*`KSGenValueFix`*<br>*`KSGenValueFormula`*<br>*`KSGenValueGauss`*<br>*`KSGenValueHistogram`*<br>*`KSGenValueList`*<br>*`KSGenValuePareto`*<br>*`KSGenValueSet`*<br>*`KSGenValueUniform`*<br>*`KSGenPositionCylindricalComposite`*<br>*`KSGenPositionFluxTube`*<br>*`KSGenPositionFrustrumComposite`*<br>*`KSGenPositionHomogeneousFluxTube`*<br>*`KSGenPositionMask`*<br>*`KSGenPositionMeshSurfaceRandom`*<br>*`KSGenPositionRectangularComposite`*<br>*`KSGenPositionSpaceRandom`*<br>*`KSGenPositionSphericalComposite`*<br>*`KSGenPositionSurfaceAdjustmentStep`*<br>*`KSGenPositionSurfaceRandom`*<br>*`KSGenSpinComposite`*<br>*`KSGenSpinRelativeComposite`*<br>*`KSGenTimeComposite`*|`creator`<br>`name`<br>`pid`<br>`special`<br>`string_id`|*`string`*<br>*`string`*<br>*`double`*<br>*`string`*<br>*`string`*|

#### KSGenLComposite

|element name|source files|child elements|child types|attributes|attribute types|
|-----|-----|-----|-----|-----|-----|
|<a name="ksgenlcomposite">`l_composite`</a>|[*KSGenGeneratorCompositeBuilder.cxx*](https://github.com/KATRIN-Experiment/Kassiopeia/tree/main/Kassiopeia/Bindings/Generators/Source/KSGenGeneratorCompositeBuilder.cxx)<br>[*KSGenLCompositeBuilder.cxx*](https://github.com/KATRIN-Experiment/Kassiopeia/tree/main/Kassiopeia/Bindings/Generators/Source/KSGenLCompositeBuilder.cxx)|[`l_fix`](#ksgenvaluefix)<br>[`l_formula`](#ksgenvalueformula)<br>[`l_gauss`](#ksgenvaluegauss)<br>[`l_list`](#ksgenvaluelist)<br>[`l_set`](#ksgenvalueset)<br>[`l_uniform`](#ksgenvalueuniform)|*`KSGenValueFix`*<br>*`KSGenValueFormula`*<br>*`KSGenValueGauss`*<br>*`KSGenValueList`*<br>*`KSGenValueSet`*<br>*`KSGenValueUniform`*|`l_value`<br>`name`|*`string`*<br>*`string`*|

#### KSGenLStatistical

|element name|source files|child elements|child types|attributes|attribute types|
|-----|-----|-----|-----|-----|-----|
|<a name="ksgenlstatistical">`l_statistical`</a>|[*KSGenGeneratorCompositeBuilder.cxx*](https://github.com/KATRIN-Experiment/Kassiopeia/tree/main/Kassiopeia/Bindings/Generators/Source/KSGenGeneratorCompositeBuilder.cxx)<br>[*KSGenLStatisticalBuilder.cxx*](https://github.com/KATRIN-Experiment/Kassiopeia/tree/main/Kassiopeia/Bindings/Generators/Source/KSGenLStatisticalBuilder.cxx)|—    |—    |`name`|*`string`*|

#### KSGenLUniformMaxN

|element name|source files|child elements|child types|attributes|attribute types|
|-----|-----|-----|-----|-----|-----|
|<a name="ksgenluniformmaxn">`l_uniform_max_n`</a>|[*KSGenGeneratorCompositeBuilder.cxx*](https://github.com/KATRIN-Experiment/Kassiopeia/tree/main/Kassiopeia/Bindings/Generators/Source/KSGenGeneratorCompositeBuilder.cxx)<br>[*KSGenLUniformMaxNBuilder.cxx*](https://github.com/KATRIN-Experiment/Kassiopeia/tree/main/Kassiopeia/Bindings/Generators/Source/KSGenLUniformMaxNBuilder.cxx)|—    |—    |`name`|*`string`*|

#### KSGenMomentumRectangularComposite

|element name|source files|child elements|child types|attributes|attribute types|
|-----|-----|-----|-----|-----|-----|
|<a name="ksgenmomentumrectangularcomposite">`momentum_rectangular_composite`</a>|[*KSGenGeneratorCompositeBuilder.cxx*](https://github.com/KATRIN-Experiment/Kassiopeia/tree/main/Kassiopeia/Bindings/Generators/Source/KSGenGeneratorCompositeBuilder.cxx)<br>[*KSGenMomentumRectangularCompositeBuilder.cxx*](https://github.com/KATRIN-Experiment/Kassiopeia/tree/main/Kassiopeia/Bindings/Generators/Source/KSGenMomentumRectangularCompositeBuilder.cxx)|[`x_fix`](#ksgenvaluefix)<br>[`x_formula`](#ksgenvalueformula)<br>[`x_gauss`](#ksgenvaluegauss)<br>[`x_histogram`](#ksgenvaluehistogram)<br>[`x_list`](#ksgenvaluelist)<br>[`x_set`](#ksgenvalueset)<br>[`x_uniform`](#ksgenvalueuniform)<br>[`y_fix`](#ksgenvaluefix)<br>[`y_formula`](#ksgenvalueformula)<br>[`y_gauss`](#ksgenvaluegauss)<br>[`y_histogram`](#ksgenvaluehistogram)<br>[`y_list`](#ksgenvaluelist)<br>[`y_set`](#ksgenvalueset)<br>[`y_uniform`](#ksgenvalueuniform)<br>[`z_fix`](#ksgenvaluefix)<br>[`z_formula`](#ksgenvalueformula)<br>[`z_gauss`](#ksgenvaluegauss)<br>[`z_histogram`](#ksgenvaluehistogram)<br>[`z_list`](#ksgenvaluelist)<br>[`z_set`](#ksgenvalueset)<br>[`z_uniform`](#ksgenvalueuniform)|*`KSGenValueFix`*<br>*`KSGenValueFormula`*<br>*`KSGenValueGauss`*<br>*`KSGenValueHistogram`*<br>*`KSGenValueList`*<br>*`KSGenValueSet`*<br>*`KSGenValueUniform`*<br>*`KSGenValueFix`*<br>*`KSGenValueFormula`*<br>*`KSGenValueGauss`*<br>*`KSGenValueHistogram`*<br>*`KSGenValueList`*<br>*`KSGenValueSet`*<br>*`KSGenValueUniform`*<br>*`KSGenValueFix`*<br>*`KSGenValueFormula`*<br>*`KSGenValueGauss`*<br>*`KSGenValueHistogram`*<br>*`KSGenValueList`*<br>*`KSGenValueSet`*<br>*`KSGenValueUniform`*|`name`<br>`space`<br>`surface`<br>`x`<br>`y`<br>`z`|*`string`*<br>*`string`*<br>*`string`*<br>*`string`*<br>*`string`*<br>*`string`*|

#### KSGenNComposite

|element name|source files|child elements|child types|attributes|attribute types|
|-----|-----|-----|-----|-----|-----|
|<a name="ksgenncomposite">`n_composite`</a>|[*KSGenGeneratorCompositeBuilder.cxx*](https://github.com/KATRIN-Experiment/Kassiopeia/tree/main/Kassiopeia/Bindings/Generators/Source/KSGenGeneratorCompositeBuilder.cxx)<br>[*KSGenNCompositeBuilder.cxx*](https://github.com/KATRIN-Experiment/Kassiopeia/tree/main/Kassiopeia/Bindings/Generators/Source/KSGenNCompositeBuilder.cxx)|[`n_fix`](#ksgenvaluefix)<br>[`n_formula`](#ksgenvalueformula)<br>[`n_gauss`](#ksgenvaluegauss)<br>[`n_list`](#ksgenvaluelist)<br>[`n_pareto`](#ksgenvaluepareto)<br>[`n_set`](#ksgenvalueset)<br>[`n_uniform`](#ksgenvalueuniform)|*`KSGenValueFix`*<br>*`KSGenValueFormula`*<br>*`KSGenValueGauss`*<br>*`KSGenValueList`*<br>*`KSGenValuePareto`*<br>*`KSGenValueSet`*<br>*`KSGenValueUniform`*|`n_value`<br>`name`|*`string`*<br>*`string`*|

##### KSGenValuePareto

|element name|source files|child elements|child types|attributes|attribute types|
|-----|-----|-----|-----|-----|-----|
|<a name="ksgenvaluepareto">`n_pareto`</a>|[*KSGenGeneratorCompositeBuilder.cxx*](https://github.com/KATRIN-Experiment/Kassiopeia/tree/main/Kassiopeia/Bindings/Generators/Source/KSGenGeneratorCompositeBuilder.cxx)<br>[*KSGenNCompositeBuilder.cxx*](https://github.com/KATRIN-Experiment/Kassiopeia/tree/main/Kassiopeia/Bindings/Generators/Source/KSGenNCompositeBuilder.cxx)<br>[*KSGenValueParetoBuilder.cxx*](https://github.com/KATRIN-Experiment/Kassiopeia/tree/main/Kassiopeia/Bindings/Generators/Source/KSGenValueParetoBuilder.cxx)|—    |—    |`cutoff`<br>`name`<br>`offset`<br>`slope`<br>`value_max`<br>`value_min`|*`double`*<br>*`string`*<br>*`double`*<br>*`double`*<br>*`double`*<br>*`double`*|

#### KSGenPositionCylindricalComposite

|element name|source files|child elements|child types|attributes|attribute types|
|-----|-----|-----|-----|-----|-----|
|<a name="ksgenpositioncylindricalcomposite">`position_cylindrical_composite`</a>|[*KSGenGeneratorCompositeBuilder.cxx*](https://github.com/KATRIN-Experiment/Kassiopeia/tree/main/Kassiopeia/Bindings/Generators/Source/KSGenGeneratorCompositeBuilder.cxx)<br>[*KSGenPositionCylindricalCompositeBuilder.cxx*](https://github.com/KATRIN-Experiment/Kassiopeia/tree/main/Kassiopeia/Bindings/Generators/Source/KSGenPositionCylindricalCompositeBuilder.cxx)<br>[*KSGenPositionMaskBuilder.cxx*](https://github.com/KATRIN-Experiment/Kassiopeia/tree/main/Kassiopeia/Bindings/Generators/Source/KSGenPositionMaskBuilder.cxx)|[`phi_fix`](#ksgenvaluefix)<br>[`phi_formula`](#ksgenvalueformula)<br>[`phi_gauss`](#ksgenvaluegauss)<br>[`phi_histogram`](#ksgenvaluehistogram)<br>[`phi_list`](#ksgenvaluelist)<br>[`phi_set`](#ksgenvalueset)<br>[`phi_uniform`](#ksgenvalueuniform)<br>[`r_cylindrical`](#ksgenvalueradiuscylindrical)<br>[`r_fix`](#ksgenvaluefix)<br>[`r_formula`](#ksgenvalueformula)<br>[`r_fraction`](#ksgenvalueradiusfraction)<br>[`r_gauss`](#ksgenvaluegauss)<br>[`r_histogram`](#ksgenvaluehistogram)<br>[`r_list`](#ksgenvaluelist)<br>[`r_set`](#ksgenvalueset)<br>[`r_uniform`](#ksgenvalueuniform)<br>[`z_fix`](#ksgenvaluefix)<br>[`z_formula`](#ksgenvalueformula)<br>[`z_gauss`](#ksgenvaluegauss)<br>[`z_histogram`](#ksgenvaluehistogram)<br>[`z_list`](#ksgenvaluelist)<br>[`z_set`](#ksgenvalueset)<br>[`z_uniform`](#ksgenvalueuniform)|*`KSGenValueFix`*<br>*`KSGenValueFormula`*<br>*`KSGenValueGauss`*<br>*`KSGenValueHistogram`*<br>*`KSGenValueList`*<br>*`KSGenValueSet`*<br>*`KSGenValueUniform`*<br>*`KSGenValueRadiusCylindrical`*<br>*`KSGenValueFix`*<br>*`KSGenValueFormula`*<br>*`KSGenValueRadiusFraction`*<br>*`KSGenValueGauss`*<br>*`KSGenValueHistogram`*<br>*`KSGenValueList`*<br>*`KSGenValueSet`*<br>*`KSGenValueUniform`*<br>*`KSGenValueFix`*<br>*`KSGenValueFormula`*<br>*`KSGenValueGauss`*<br>*`KSGenValueHistogram`*<br>*`KSGenValueList`*<br>*`KSGenValueSet`*<br>*`KSGenValueUniform`*|`name`<br>`phi`<br>`r`<br>`space`<br>`surface`<br>`z`|*`string`*<br>*`string`*<br>*`string`*<br>*`string`*<br>*`string`*<br>*`string`*|

##### KSGenValueRadiusCylindrical

|element name|source files|child elements|child types|attributes|attribute types|
|-----|-----|-----|-----|-----|-----|
|<a name="ksgenvalueradiuscylindrical">`r_cylindrical`</a>|[*KSGenPositionCylindricalCompositeBuilder.cxx*](https://github.com/KATRIN-Experiment/Kassiopeia/tree/main/Kassiopeia/Bindings/Generators/Source/KSGenPositionCylindricalCompositeBuilder.cxx)<br>[*KSGenPositionFrustrumCompositeBuilder.cxx*](https://github.com/KATRIN-Experiment/Kassiopeia/tree/main/Kassiopeia/Bindings/Generators/Source/KSGenPositionFrustrumCompositeBuilder.cxx)<br>[*KSGenValueRadiusCylindricalBuilder.cxx*](https://github.com/KATRIN-Experiment/Kassiopeia/tree/main/Kassiopeia/Bindings/Generators/Source/KSGenValueRadiusCylindricalBuilder.cxx)|—    |—    |`name`<br>`radius_max`<br>`radius_min`|*`string`*<br>*`double`*<br>*`double`*|

##### KSGenValueRadiusFraction

|element name|source files|child elements|child types|attributes|attribute types|
|-----|-----|-----|-----|-----|-----|
|<a name="ksgenvalueradiusfraction">`r_fraction`</a>|[*KSGenPositionCylindricalCompositeBuilder.cxx*](https://github.com/KATRIN-Experiment/Kassiopeia/tree/main/Kassiopeia/Bindings/Generators/Source/KSGenPositionCylindricalCompositeBuilder.cxx)<br>[*KSGenPositionFrustrumCompositeBuilder.cxx*](https://github.com/KATRIN-Experiment/Kassiopeia/tree/main/Kassiopeia/Bindings/Generators/Source/KSGenPositionFrustrumCompositeBuilder.cxx)<br>[*KSGenValueRadiusFractionBuilder.cxx*](https://github.com/KATRIN-Experiment/Kassiopeia/tree/main/Kassiopeia/Bindings/Generators/Source/KSGenValueRadiusFractionBuilder.cxx)|—    |—    |`name`|*`string`*|

#### KSGenPositionFluxTube

|element name|source files|child elements|child types|attributes|attribute types|
|-----|-----|-----|-----|-----|-----|
|<a name="ksgenpositionfluxtube">`position_flux_tube`</a>|[*KSGenPositionFluxTubeBuilder.cxx*](https://github.com/KATRIN-Experiment/Kassiopeia/tree/main/Kassiopeia/Bindings/Generators/Source/KSGenPositionFluxTubeBuilder.cxx)|[`phi_fix`](#ksgenvaluefix)<br>[`phi_gauss`](#ksgenvaluegauss)<br>[`phi_set`](#ksgenvalueset)<br>[`phi_uniform`](#ksgenvalueuniform)<br>[`r_formula`](#ksgenvalueformula)<br>[`z_fix`](#ksgenvaluefix)<br>[`z_formula`](#ksgenvalueformula)<br>[`z_gauss`](#ksgenvaluegauss)<br>[`z_set`](#ksgenvalueset)<br>[`z_uniform`](#ksgenvalueuniform)|*`KSGenValueFix`*<br>*`KSGenValueGauss`*<br>*`KSGenValueSet`*<br>*`KSGenValueUniform`*<br>*`KSGenValueFormula`*<br>*`KSGenValueFix`*<br>*`KSGenValueFormula`*<br>*`KSGenValueGauss`*<br>*`KSGenValueSet`*<br>*`KSGenValueUniform`*|`flux`<br>`magnetic_field_name`<br>`n_integration_step`<br>`name`<br>`only_surface`<br>`phi`<br>`space`<br>`surface`<br>`z`|*`double`*<br>*`string`*<br>*`int`*<br>*`string`*<br>*`bool`*<br>*`string`*<br>*`string`*<br>*`string`*<br>*`string`*|

#### KSGenPositionFrustrumComposite

|element name|source files|child elements|child types|attributes|attribute types|
|-----|-----|-----|-----|-----|-----|
|<a name="ksgenpositionfrustrumcomposite">`position_frustrum_composite`</a>|[*KSGenGeneratorCompositeBuilder.cxx*](https://github.com/KATRIN-Experiment/Kassiopeia/tree/main/Kassiopeia/Bindings/Generators/Source/KSGenGeneratorCompositeBuilder.cxx)<br>[*KSGenPositionFrustrumCompositeBuilder.cxx*](https://github.com/KATRIN-Experiment/Kassiopeia/tree/main/Kassiopeia/Bindings/Generators/Source/KSGenPositionFrustrumCompositeBuilder.cxx)|[`phi_fix`](#ksgenvaluefix)<br>[`phi_formula`](#ksgenvalueformula)<br>[`phi_gauss`](#ksgenvaluegauss)<br>[`phi_histogram`](#ksgenvaluehistogram)<br>[`phi_list`](#ksgenvaluelist)<br>[`phi_set`](#ksgenvalueset)<br>[`phi_uniform`](#ksgenvalueuniform)<br>[`r1_fix`](#ksgenvaluefix)<br>[`r2_fix`](#ksgenvaluefix)<br>[`r_cylindrical`](#ksgenvalueradiuscylindrical)<br>[`r_fix`](#ksgenvaluefix)<br>[`r_formula`](#ksgenvalueformula)<br>[`r_fraction`](#ksgenvalueradiusfraction)<br>[`r_gauss`](#ksgenvaluegauss)<br>[`r_histogram`](#ksgenvaluehistogram)<br>[`r_list`](#ksgenvaluelist)<br>[`r_set`](#ksgenvalueset)<br>[`r_uniform`](#ksgenvalueuniform)<br>[`z1_fix`](#ksgenvaluefix)<br>[`z2_fix`](#ksgenvaluefix)<br>[`z_fix`](#ksgenvaluefix)<br>[`z_formula`](#ksgenvalueformula)<br>[`z_frustrum`](#ksgenvaluezfrustrum)<br>[`z_gauss`](#ksgenvaluegauss)<br>[`z_histogram`](#ksgenvaluehistogram)<br>[`z_list`](#ksgenvaluelist)<br>[`z_set`](#ksgenvalueset)<br>[`z_uniform`](#ksgenvalueuniform)|*`KSGenValueFix`*<br>*`KSGenValueFormula`*<br>*`KSGenValueGauss`*<br>*`KSGenValueHistogram`*<br>*`KSGenValueList`*<br>*`KSGenValueSet`*<br>*`KSGenValueUniform`*<br>*`KSGenValueFix`*<br>*`KSGenValueFix`*<br>*`KSGenValueRadiusCylindrical`*<br>*`KSGenValueFix`*<br>*`KSGenValueFormula`*<br>*`KSGenValueRadiusFraction`*<br>*`KSGenValueGauss`*<br>*`KSGenValueHistogram`*<br>*`KSGenValueList`*<br>*`KSGenValueSet`*<br>*`KSGenValueUniform`*<br>*`KSGenValueFix`*<br>*`KSGenValueFix`*<br>*`KSGenValueFix`*<br>*`KSGenValueFormula`*<br>*`KSGenValueZFrustrum`*<br>*`KSGenValueGauss`*<br>*`KSGenValueHistogram`*<br>*`KSGenValueList`*<br>*`KSGenValueSet`*<br>*`KSGenValueUniform`*|`name`<br>`phi`<br>`r`<br>`r1`<br>`r2`<br>`space`<br>`surface`<br>`z`<br>`z1`<br>`z2`|*`string`*<br>*`string`*<br>*`string`*<br>*`string`*<br>*`string`*<br>*`string`*<br>*`string`*<br>*`string`*<br>*`string`*<br>*`string`*|

##### KSGenValueZFrustrum

|element name|source files|child elements|child types|attributes|attribute types|
|-----|-----|-----|-----|-----|-----|
|<a name="ksgenvaluezfrustrum">`z_frustrum`</a>|[*KSGenPositionFrustrumCompositeBuilder.cxx*](https://github.com/KATRIN-Experiment/Kassiopeia/tree/main/Kassiopeia/Bindings/Generators/Source/KSGenPositionFrustrumCompositeBuilder.cxx)<br>[*KSGenValueZFrustrumBuilder.cxx*](https://github.com/KATRIN-Experiment/Kassiopeia/tree/main/Kassiopeia/Bindings/Generators/Source/KSGenValueZFrustrumBuilder.cxx)|—    |—    |`name`<br>`r1`<br>`r2`<br>`z1`<br>`z2`|*`string`*<br>*`double`*<br>*`double`*<br>*`double`*<br>*`double`*|

#### KSGenPositionHomogeneousFluxTube

|element name|source files|child elements|child types|attributes|attribute types|
|-----|-----|-----|-----|-----|-----|
|<a name="ksgenpositionhomogeneousfluxtube">`position_homogeneous_flux_tube`</a>|[*KSGenPositionHomogeneousFluxTubeBuilder.cxx*](https://github.com/KATRIN-Experiment/Kassiopeia/tree/main/Kassiopeia/Bindings/Generators/Source/KSGenPositionHomogeneousFluxTubeBuilder.cxx)|—    |—    |`flux`<br>`magnetic_field_name`<br>`n_integration_step`<br>`name`<br>`phi_max`<br>`phi_min`<br>`r_max`<br>`z_max`<br>`z_min`|*`double`*<br>*`string`*<br>*`int`*<br>*`string`*<br>*`double`*<br>*`double`*<br>*`double`*<br>*`double`*<br>*`double`*|

#### KSGenPositionMask

|element name|source files|child elements|child types|attributes|attribute types|
|-----|-----|-----|-----|-----|-----|
|<a name="ksgenpositionmask">`position_mask`</a>|[*KSGenGeneratorCompositeBuilder.cxx*](https://github.com/KATRIN-Experiment/Kassiopeia/tree/main/Kassiopeia/Bindings/Generators/Source/KSGenGeneratorCompositeBuilder.cxx)<br>[*KSGenPositionMaskBuilder.cxx*](https://github.com/KATRIN-Experiment/Kassiopeia/tree/main/Kassiopeia/Bindings/Generators/Source/KSGenPositionMaskBuilder.cxx)|[`position_cylindrical_composite`](#ksgenpositioncylindricalcomposite)<br>[`position_mesh_surface_random`](#ksgenpositionmeshsurfacerandom)<br>[`position_rectangular_composite`](#ksgenpositionrectangularcomposite)<br>[`position_space_random`](#ksgenpositionspacerandom)<br>[`position_spherical_composite`](#ksgenpositionsphericalcomposite)<br>[`position_surface_random`](#ksgenpositionsurfacerandom )|*`KSGenPositionCylindricalComposite`*<br>*`KSGenPositionMeshSurfaceRandom`*<br>*`KSGenPositionRectangularComposite`*<br>*`KSGenPositionSpaceRandom`*<br>*`KSGenPositionSphericalComposite`*<br>*`KSGenPositionSurfaceRandom `*|`max_retries`<br>`name`<br>`spaces_allowed`<br>`spaces_forbidden`|*`unsigned int`*<br>*`string`*<br>*`string`*<br>*`string`*|

##### KSGenPositionMeshSurfaceRandom

|element name|source files|child elements|child types|attributes|attribute types|
|-----|-----|-----|-----|-----|-----|
|<a name="ksgenpositionmeshsurfacerandom">`position_mesh_surface_random`</a>|[*KSGenGeneratorCompositeBuilder.cxx*](https://github.com/KATRIN-Experiment/Kassiopeia/tree/main/Kassiopeia/Bindings/Generators/Source/KSGenGeneratorCompositeBuilder.cxx)<br>[*KSGenPositionMaskBuilder.cxx*](https://github.com/KATRIN-Experiment/Kassiopeia/tree/main/Kassiopeia/Bindings/Generators/Source/KSGenPositionMaskBuilder.cxx)<br>[*KSGenPositionMeshSurfaceRandomBuilder.cxx*](https://github.com/KATRIN-Experiment/Kassiopeia/tree/main/Kassiopeia/Bindings/Generators/Source/KSGenPositionMeshSurfaceRandomBuilder.cxx)|—    |—    |`name`<br>`surfaces`|*`string`*<br>*`string`*|

##### KSGenPositionRectangularComposite

|element name|source files|child elements|child types|attributes|attribute types|
|-----|-----|-----|-----|-----|-----|
|<a name="ksgenpositionrectangularcomposite">`position_rectangular_composite`</a>|[*KSGenGeneratorCompositeBuilder.cxx*](https://github.com/KATRIN-Experiment/Kassiopeia/tree/main/Kassiopeia/Bindings/Generators/Source/KSGenGeneratorCompositeBuilder.cxx)<br>[*KSGenPositionMaskBuilder.cxx*](https://github.com/KATRIN-Experiment/Kassiopeia/tree/main/Kassiopeia/Bindings/Generators/Source/KSGenPositionMaskBuilder.cxx)<br>[*KSGenPositionRectangularCompositeBuilder.cxx*](https://github.com/KATRIN-Experiment/Kassiopeia/tree/main/Kassiopeia/Bindings/Generators/Source/KSGenPositionRectangularCompositeBuilder.cxx)|[`x_fix`](#ksgenvaluefix)<br>[`x_formula`](#ksgenvalueformula)<br>[`x_gauss`](#ksgenvaluegauss)<br>[`x_histogram`](#ksgenvaluehistogram)<br>[`x_list`](#ksgenvaluelist)<br>[`x_set`](#ksgenvalueset)<br>[`x_uniform`](#ksgenvalueuniform)<br>[`y_fix`](#ksgenvaluefix)<br>[`y_formula`](#ksgenvalueformula)<br>[`y_gauss`](#ksgenvaluegauss)<br>[`y_histogram`](#ksgenvaluehistogram)<br>[`y_list`](#ksgenvaluelist)<br>[`y_set`](#ksgenvalueset)<br>[`y_uniform`](#ksgenvalueuniform)<br>[`z_fix`](#ksgenvaluefix)<br>[`z_formula`](#ksgenvalueformula)<br>[`z_gauss`](#ksgenvaluegauss)<br>[`z_histogram`](#ksgenvaluehistogram)<br>[`z_list`](#ksgenvaluelist)<br>[`z_set`](#ksgenvalueset)<br>[`z_uniform`](#ksgenvalueuniform)|*`KSGenValueFix`*<br>*`KSGenValueFormula`*<br>*`KSGenValueGauss`*<br>*`KSGenValueHistogram`*<br>*`KSGenValueList`*<br>*`KSGenValueSet`*<br>*`KSGenValueUniform`*<br>*`KSGenValueFix`*<br>*`KSGenValueFormula`*<br>*`KSGenValueGauss`*<br>*`KSGenValueHistogram`*<br>*`KSGenValueList`*<br>*`KSGenValueSet`*<br>*`KSGenValueUniform`*<br>*`KSGenValueFix`*<br>*`KSGenValueFormula`*<br>*`KSGenValueGauss`*<br>*`KSGenValueHistogram`*<br>*`KSGenValueList`*<br>*`KSGenValueSet`*<br>*`KSGenValueUniform`*|`name`<br>`space`<br>`surface`<br>`x`<br>`y`<br>`z`|*`string`*<br>*`string`*<br>*`string`*<br>*`string`*<br>*`string`*<br>*`string`*|

##### KSGenPositionSpaceRandom

|element name|source files|child elements|child types|attributes|attribute types|
|-----|-----|-----|-----|-----|-----|
|<a name="ksgenpositionspacerandom">`position_space_random`</a>|[*KSGenGeneratorCompositeBuilder.cxx*](https://github.com/KATRIN-Experiment/Kassiopeia/tree/main/Kassiopeia/Bindings/Generators/Source/KSGenGeneratorCompositeBuilder.cxx)<br>[*KSGenPositionMaskBuilder.cxx*](https://github.com/KATRIN-Experiment/Kassiopeia/tree/main/Kassiopeia/Bindings/Generators/Source/KSGenPositionMaskBuilder.cxx)<br>[*KSGenPositionSpaceRandomBuilder.cxx*](https://github.com/KATRIN-Experiment/Kassiopeia/tree/main/Kassiopeia/Bindings/Generators/Source/KSGenPositionSpaceRandomBuilder.cxx)|—    |—    |`name`<br>`spaces`|*`string`*<br>*`string`*|

##### KSGenPositionSphericalComposite

|element name|source files|child elements|child types|attributes|attribute types|
|-----|-----|-----|-----|-----|-----|
|<a name="ksgenpositionsphericalcomposite">`position_spherical_composite`</a>|[*KSGenGeneratorCompositeBuilder.cxx*](https://github.com/KATRIN-Experiment/Kassiopeia/tree/main/Kassiopeia/Bindings/Generators/Source/KSGenGeneratorCompositeBuilder.cxx)<br>[*KSGenPositionMaskBuilder.cxx*](https://github.com/KATRIN-Experiment/Kassiopeia/tree/main/Kassiopeia/Bindings/Generators/Source/KSGenPositionMaskBuilder.cxx)<br>[*KSGenPositionSphericalCompositeBuilder.cxx*](https://github.com/KATRIN-Experiment/Kassiopeia/tree/main/Kassiopeia/Bindings/Generators/Source/KSGenPositionSphericalCompositeBuilder.cxx)|[`phi_fix`](#ksgenvaluefix)<br>[`phi_formula`](#ksgenvalueformula)<br>[`phi_gauss`](#ksgenvaluegauss)<br>[`phi_histogram`](#ksgenvaluehistogram)<br>[`phi_list`](#ksgenvaluelist)<br>[`phi_set`](#ksgenvalueset)<br>[`phi_uniform`](#ksgenvalueuniform)<br>[`r_fix`](#ksgenvaluefix)<br>[`r_formula`](#ksgenvalueformula)<br>[`r_gauss`](#ksgenvaluegauss)<br>[`r_histogram`](#ksgenvaluehistogram)<br>[`r_list`](#ksgenvaluelist)<br>[`r_set`](#ksgenvalueset)<br>[`r_spherical`](#ksgenvalueradiusspherical)<br>[`r_uniform`](#ksgenvalueuniform)<br>[`theta_fix`](#ksgenvaluefix)<br>[`theta_formula`](#ksgenvalueformula)<br>[`theta_gauss`](#ksgenvaluegauss)<br>[`theta_histogram`](#ksgenvaluehistogram)<br>[`theta_list`](#ksgenvaluelist)<br>[`theta_set`](#ksgenvalueset)<br>[`theta_spherical`](#ksgenvalueanglespherical)<br>[`theta_uniform`](#ksgenvalueuniform)|*`KSGenValueFix`*<br>*`KSGenValueFormula`*<br>*`KSGenValueGauss`*<br>*`KSGenValueHistogram`*<br>*`KSGenValueList`*<br>*`KSGenValueSet`*<br>*`KSGenValueUniform`*<br>*`KSGenValueFix`*<br>*`KSGenValueFormula`*<br>*`KSGenValueGauss`*<br>*`KSGenValueHistogram`*<br>*`KSGenValueList`*<br>*`KSGenValueSet`*<br>*`KSGenValueRadiusSpherical`*<br>*`KSGenValueUniform`*<br>*`KSGenValueFix`*<br>*`KSGenValueFormula`*<br>*`KSGenValueGauss`*<br>*`KSGenValueHistogram`*<br>*`KSGenValueList`*<br>*`KSGenValueSet`*<br>*`KSGenValueAngleSpherical`*<br>*`KSGenValueUniform`*|`name`<br>`phi`<br>`r`<br>`space`<br>`surface`<br>`theta`|*`string`*<br>*`string`*<br>*`string`*<br>*`string`*<br>*`string`*<br>*`string`*|

###### KSGenValueRadiusSpherical

|element name|source files|child elements|child types|attributes|attribute types|
|-----|-----|-----|-----|-----|-----|
|<a name="ksgenvalueradiusspherical">`r_spherical`</a>|[*KSGenPositionSphericalCompositeBuilder.cxx*](https://github.com/KATRIN-Experiment/Kassiopeia/tree/main/Kassiopeia/Bindings/Generators/Source/KSGenPositionSphericalCompositeBuilder.cxx)<br>[*KSGenValueRadiusSphericalBuilder.cxx*](https://github.com/KATRIN-Experiment/Kassiopeia/tree/main/Kassiopeia/Bindings/Generators/Source/KSGenValueRadiusSphericalBuilder.cxx)|—    |—    |`name`<br>`radius_max`<br>`radius_min`|*`string`*<br>*`double`*<br>*`double`*|

##### KSGenPositionSurfaceRandom 

|element name|source files|child elements|child types|attributes|attribute types|
|-----|-----|-----|-----|-----|-----|
|<a name="ksgenpositionsurfacerandom ">`position_surface_random`</a>|[*KSGenPositionMaskBuilder.cxx*](https://github.com/KATRIN-Experiment/Kassiopeia/tree/main/Kassiopeia/Bindings/Generators/Source/KSGenPositionMaskBuilder.cxx)|—    |—    |—    |—    |

#### KSGenPositionSurfaceAdjustmentStep

|element name|source files|child elements|child types|attributes|attribute types|
|-----|-----|-----|-----|-----|-----|
|<a name="ksgenpositionsurfaceadjustmentstep">`position_surface_adjustment_step`</a>|[*KSGenGeneratorCompositeBuilder.cxx*](https://github.com/KATRIN-Experiment/Kassiopeia/tree/main/Kassiopeia/Bindings/Generators/Source/KSGenGeneratorCompositeBuilder.cxx)<br>[*KSGenPositionSurfaceAdjustmentStepBuilder.cxx*](https://github.com/KATRIN-Experiment/Kassiopeia/tree/main/Kassiopeia/Bindings/Generators/Source/KSGenPositionSurfaceAdjustmentStepBuilder.cxx)|—    |—    |`length`<br>`name`|*`double`*<br>*`string`*|

#### KSGenPositionSurfaceRandom

|element name|source files|child elements|child types|attributes|attribute types|
|-----|-----|-----|-----|-----|-----|
|<a name="ksgenpositionsurfacerandom">`position_surface_random`</a>|[*KSGenGeneratorCompositeBuilder.cxx*](https://github.com/KATRIN-Experiment/Kassiopeia/tree/main/Kassiopeia/Bindings/Generators/Source/KSGenGeneratorCompositeBuilder.cxx)<br>[*KSGenPositionSurfaceRandomBuilder.cxx*](https://github.com/KATRIN-Experiment/Kassiopeia/tree/main/Kassiopeia/Bindings/Generators/Source/KSGenPositionSurfaceRandomBuilder.cxx)|—    |—    |`name`<br>`surfaces`|*`string`*<br>*`string`*|

#### KSGenSpinComposite

|element name|source files|child elements|child types|attributes|attribute types|
|-----|-----|-----|-----|-----|-----|
|<a name="ksgenspincomposite">`spin_composite`</a>|[*KSGenGeneratorCompositeBuilder.cxx*](https://github.com/KATRIN-Experiment/Kassiopeia/tree/main/Kassiopeia/Bindings/Generators/Source/KSGenGeneratorCompositeBuilder.cxx)<br>[*KSGenSpinCompositeBuilder.cxx*](https://github.com/KATRIN-Experiment/Kassiopeia/tree/main/Kassiopeia/Bindings/Generators/Source/KSGenSpinCompositeBuilder.cxx)|[`phi_fix`](#ksgenvaluefix)<br>[`phi_formula`](#ksgenvalueformula)<br>[`phi_gauss`](#ksgenvaluegauss)<br>[`phi_generalized_gauss`](#ksgenvaluegeneralizedgauss)<br>[`phi_histogram`](#ksgenvaluehistogram)<br>[`phi_list`](#ksgenvaluelist)<br>[`phi_set`](#ksgenvalueset)<br>[`phi_uniform`](#ksgenvalueuniform)<br>[`theta_cosine`](#ksgenvalueanglecosine)<br>[`theta_fix`](#ksgenvaluefix)<br>[`theta_formula`](#ksgenvalueformula)<br>[`theta_gauss`](#ksgenvaluegauss)<br>[`theta_generalized_gauss`](#ksgenvaluegeneralizedgauss)<br>[`theta_histogram`](#ksgenvaluehistogram)<br>[`theta_list`](#ksgenvaluelist)<br>[`theta_set`](#ksgenvalueset)<br>[`theta_spherical`](#ksgenvalueanglespherical)<br>[`theta_uniform`](#ksgenvalueuniform)|*`KSGenValueFix`*<br>*`KSGenValueFormula`*<br>*`KSGenValueGauss`*<br>*`KSGenValueGeneralizedGauss`*<br>*`KSGenValueHistogram`*<br>*`KSGenValueList`*<br>*`KSGenValueSet`*<br>*`KSGenValueUniform`*<br>*`KSGenValueAngleCosine`*<br>*`KSGenValueFix`*<br>*`KSGenValueFormula`*<br>*`KSGenValueGauss`*<br>*`KSGenValueGeneralizedGauss`*<br>*`KSGenValueHistogram`*<br>*`KSGenValueList`*<br>*`KSGenValueSet`*<br>*`KSGenValueAngleSpherical`*<br>*`KSGenValueUniform`*|`name`<br>`phi`<br>`space`<br>`surface`<br>`theta`|*`string`*<br>*`string`*<br>*`string`*<br>*`string`*<br>*`string`*|

#### KSGenSpinRelativeComposite

|element name|source files|child elements|child types|attributes|attribute types|
|-----|-----|-----|-----|-----|-----|
|<a name="ksgenspinrelativecomposite">`spin_relative_composite`</a>|[*KSGenGeneratorCompositeBuilder.cxx*](https://github.com/KATRIN-Experiment/Kassiopeia/tree/main/Kassiopeia/Bindings/Generators/Source/KSGenGeneratorCompositeBuilder.cxx)<br>[*KSGenSpinRelativeCompositeBuilder.cxx*](https://github.com/KATRIN-Experiment/Kassiopeia/tree/main/Kassiopeia/Bindings/Generators/Source/KSGenSpinRelativeCompositeBuilder.cxx)|[`phi_fix`](#ksgenvaluefix)<br>[`phi_formula`](#ksgenvalueformula)<br>[`phi_gauss`](#ksgenvaluegauss)<br>[`phi_generalized_gauss`](#ksgenvaluegeneralizedgauss)<br>[`phi_histogram`](#ksgenvaluehistogram)<br>[`phi_list`](#ksgenvaluelist)<br>[`phi_set`](#ksgenvalueset)<br>[`phi_uniform`](#ksgenvalueuniform)<br>[`theta_cosine`](#ksgenvalueanglecosine)<br>[`theta_fix`](#ksgenvaluefix)<br>[`theta_formula`](#ksgenvalueformula)<br>[`theta_gauss`](#ksgenvaluegauss)<br>[`theta_generalized_gauss`](#ksgenvaluegeneralizedgauss)<br>[`theta_histogram`](#ksgenvaluehistogram)<br>[`theta_list`](#ksgenvaluelist)<br>[`theta_set`](#ksgenvalueset)<br>[`theta_spherical`](#ksgenvalueanglespherical)<br>[`theta_uniform`](#ksgenvalueuniform)|*`KSGenValueFix`*<br>*`KSGenValueFormula`*<br>*`KSGenValueGauss`*<br>*`KSGenValueGeneralizedGauss`*<br>*`KSGenValueHistogram`*<br>*`KSGenValueList`*<br>*`KSGenValueSet`*<br>*`KSGenValueUniform`*<br>*`KSGenValueAngleCosine`*<br>*`KSGenValueFix`*<br>*`KSGenValueFormula`*<br>*`KSGenValueGauss`*<br>*`KSGenValueGeneralizedGauss`*<br>*`KSGenValueHistogram`*<br>*`KSGenValueList`*<br>*`KSGenValueSet`*<br>*`KSGenValueAngleSpherical`*<br>*`KSGenValueUniform`*|`name`<br>`phi`<br>`space`<br>`surface`<br>`theta`|*`string`*<br>*`string`*<br>*`string`*<br>*`string`*<br>*`string`*|

#### KSGenTimeComposite

|element name|source files|child elements|child types|attributes|attribute types|
|-----|-----|-----|-----|-----|-----|
|<a name="ksgentimecomposite">`time_composite`</a>|[*KSGenGeneratorCompositeBuilder.cxx*](https://github.com/KATRIN-Experiment/Kassiopeia/tree/main/Kassiopeia/Bindings/Generators/Source/KSGenGeneratorCompositeBuilder.cxx)<br>[*KSGenTimeCompositeBuilder.cxx*](https://github.com/KATRIN-Experiment/Kassiopeia/tree/main/Kassiopeia/Bindings/Generators/Source/KSGenTimeCompositeBuilder.cxx)|[`time_fix`](#ksgenvaluefix)<br>[`time_formula`](#ksgenvalueformula)<br>[`time_gauss`](#ksgenvaluegauss)<br>[`time_generalized_gauss`](#ksgenvaluegeneralizedgauss)<br>[`time_histogram`](#ksgenvaluehistogram)<br>[`time_list`](#ksgenvaluelist)<br>[`time_set`](#ksgenvalueset)<br>[`time_uniform`](#ksgenvalueuniform)|*`KSGenValueFix`*<br>*`KSGenValueFormula`*<br>*`KSGenValueGauss`*<br>*`KSGenValueGeneralizedGauss`*<br>*`KSGenValueHistogram`*<br>*`KSGenValueList`*<br>*`KSGenValueSet`*<br>*`KSGenValueUniform`*|`name`<br>`time_value`|*`string`*<br>*`string`*|

### KSGenGeneratorTextFile

|element name|source files|child elements|child types|attributes|attribute types|
|-----|-----|-----|-----|-----|-----|
|<a name="ksgengeneratortextfile">`ksgen_generator_file`</a>|[*KSGenGeneratorTextFileBuilder.cxx*](https://github.com/KATRIN-Experiment/Kassiopeia/tree/main/Kassiopeia/Bindings/Generators/Source/KSGenGeneratorTextFileBuilder.cxx)|—    |—    |`base`<br>`name`<br>`path`|*`string`*<br>*`string`*<br>*`string`*|

### KSGenGeneratorSimulation

|element name|source files|child elements|child types|attributes|attribute types|
|-----|-----|-----|-----|-----|-----|
|<a name="ksgengeneratorsimulation">`ksgen_generator_simulation`</a>|[*KSGenGeneratorSimulationBuilder.cxx*](https://github.com/KATRIN-Experiment/Kassiopeia/tree/main/Kassiopeia/Bindings/Generators/Source/KSGenGeneratorSimulationBuilder.cxx)|—    |—    |`base`<br>`direction_x`<br>`direction_y`<br>`direction_z`<br>`energy`<br>`generator`<br>`kinetic_energy_field`<br>`momentum_field`<br>`name`<br>`path`<br>`pid_field`<br>`position_field`<br>`position_x`<br>`position_y`<br>`position_z`<br>`terminator`<br>`time`<br>`time_field`<br>`track_group`|*`string`*<br>*`string`*<br>*`string`*<br>*`string`*<br>*`string`*<br>*`string`*<br>*`string`*<br>*`string`*<br>*`string`*<br>*`string`*<br>*`string`*<br>*`string`*<br>*`string`*<br>*`string`*<br>*`string`*<br>*`string`*<br>*`string`*<br>*`string`*<br>*`string`*|

### KSGeoSide

|element name|source files|child elements|child types|attributes|attribute types|
|-----|-----|-----|-----|-----|-----|
|<a name="ksgeoside">`ksgeo_side`</a>|[*KSGeoSideBuilder.cxx*](https://github.com/KATRIN-Experiment/Kassiopeia/tree/main/Kassiopeia/Bindings/Geometry/Source/KSGeoSideBuilder.cxx)<br>[*KSGeoSpaceBuilder.cxx*](https://github.com/KATRIN-Experiment/Kassiopeia/tree/main/Kassiopeia/Bindings/Geometry/Source/KSGeoSpaceBuilder.cxx)|[`add_electric_field`](#kscommandmemberaddelectricfielddata)<br>[`add_magnetic_field`](#kscommandmemberaddmagneticfielddata)<br>[`add_step_modifier`](#kscommandmemberaddstepmodifierdata)<br>[`add_step_output`](#kscommandmemberaddstepoutputdata)<br>[`add_terminator`](#kscommandmemberaddterminatordata)<br>[`add_track_output`](#kscommandmemberaddtrackoutputdata)<br>[`clear_surface_interaction`](#kscommandmemberclearsurfaceinteractiondata)<br>[`command`](#kscommandmemberdata)<br>[`remove_magnetic_field`](#kscommandmemberremoveelectricfielddata)<br>[`remove_step_modifier`](#kscommandmemberremovestepmodifierdata)<br>[`remove_step_output`](#kscommandmemberremovestepoutputdata)<br>[`remove_terminator`](#kscommandmemberremoveterminatordata)<br>[`remove_track_output`](#kscommandmemberremovetrackoutputdata)<br>[`set_surface_interaction`](#kscommandmembersetsurfaceinteractiondata)|*`KSCommandMemberAddElectricFieldData`*<br>*`KSCommandMemberAddMagneticFieldData`*<br>*`KSCommandMemberAddStepModifierData`*<br>*`KSCommandMemberAddStepOutputData`*<br>*`KSCommandMemberAddTerminatorData`*<br>*`KSCommandMemberAddTrackOutputData`*<br>*`KSCommandMemberClearSurfaceInteractionData`*<br>*`KSCommandMemberData`*<br>*`KSCommandMemberRemoveElectricFieldData`*<br>*`KSCommandMemberRemoveStepModifierData`*<br>*`KSCommandMemberRemoveStepOutputData`*<br>*`KSCommandMemberRemoveTerminatorData`*<br>*`KSCommandMemberRemoveTrackOutputData`*<br>*`KSCommandMemberSetSurfaceInteractionData`*|`name`<br>`spaces`<br>`surfaces`|*`string`*<br>*`string`*<br>*`string`*|

#### KSCommandMemberAddElectricFieldData

|element name|source files|child elements|child types|attributes|attribute types|
|-----|-----|-----|-----|-----|-----|
|<a name="kscommandmemberaddelectricfielddata">`add_electric_field`</a>|[*KSGeoSideBuilder.cxx*](https://github.com/KATRIN-Experiment/Kassiopeia/tree/main/Kassiopeia/Bindings/Geometry/Source/KSGeoSideBuilder.cxx)<br>[*KSGeoSpaceBuilder.cxx*](https://github.com/KATRIN-Experiment/Kassiopeia/tree/main/Kassiopeia/Bindings/Geometry/Source/KSGeoSpaceBuilder.cxx)<br>[*KSGeoSurfaceBuilder.cxx*](https://github.com/KATRIN-Experiment/Kassiopeia/tree/main/Kassiopeia/Bindings/Geometry/Source/KSGeoSurfaceBuilder.cxx)|—    |—    |—    |—    |

#### KSCommandMemberAddMagneticFieldData

|element name|source files|child elements|child types|attributes|attribute types|
|-----|-----|-----|-----|-----|-----|
|<a name="kscommandmemberaddmagneticfielddata">`add_magnetic_field`</a>|[*KSGeoSideBuilder.cxx*](https://github.com/KATRIN-Experiment/Kassiopeia/tree/main/Kassiopeia/Bindings/Geometry/Source/KSGeoSideBuilder.cxx)<br>[*KSGeoSpaceBuilder.cxx*](https://github.com/KATRIN-Experiment/Kassiopeia/tree/main/Kassiopeia/Bindings/Geometry/Source/KSGeoSpaceBuilder.cxx)<br>[*KSGeoSurfaceBuilder.cxx*](https://github.com/KATRIN-Experiment/Kassiopeia/tree/main/Kassiopeia/Bindings/Geometry/Source/KSGeoSurfaceBuilder.cxx)|—    |—    |—    |—    |

#### KSCommandMemberAddStepModifierData

|element name|source files|child elements|child types|attributes|attribute types|
|-----|-----|-----|-----|-----|-----|
|<a name="kscommandmemberaddstepmodifierdata">`add_step_modifier`</a>|[*KSGeoSideBuilder.cxx*](https://github.com/KATRIN-Experiment/Kassiopeia/tree/main/Kassiopeia/Bindings/Geometry/Source/KSGeoSideBuilder.cxx)<br>[*KSGeoSpaceBuilder.cxx*](https://github.com/KATRIN-Experiment/Kassiopeia/tree/main/Kassiopeia/Bindings/Geometry/Source/KSGeoSpaceBuilder.cxx)<br>[*KSGeoSurfaceBuilder.cxx*](https://github.com/KATRIN-Experiment/Kassiopeia/tree/main/Kassiopeia/Bindings/Geometry/Source/KSGeoSurfaceBuilder.cxx)|—    |—    |—    |—    |

#### KSCommandMemberAddStepOutputData

|element name|source files|child elements|child types|attributes|attribute types|
|-----|-----|-----|-----|-----|-----|
|<a name="kscommandmemberaddstepoutputdata">`add_step_output`</a>|[*KSGeoSideBuilder.cxx*](https://github.com/KATRIN-Experiment/Kassiopeia/tree/main/Kassiopeia/Bindings/Geometry/Source/KSGeoSideBuilder.cxx)<br>[*KSGeoSpaceBuilder.cxx*](https://github.com/KATRIN-Experiment/Kassiopeia/tree/main/Kassiopeia/Bindings/Geometry/Source/KSGeoSpaceBuilder.cxx)<br>[*KSGeoSurfaceBuilder.cxx*](https://github.com/KATRIN-Experiment/Kassiopeia/tree/main/Kassiopeia/Bindings/Geometry/Source/KSGeoSurfaceBuilder.cxx)|—    |—    |—    |—    |

#### KSCommandMemberAddTerminatorData

|element name|source files|child elements|child types|attributes|attribute types|
|-----|-----|-----|-----|-----|-----|
|<a name="kscommandmemberaddterminatordata">`add_terminator`</a>|[*KSGeoSideBuilder.cxx*](https://github.com/KATRIN-Experiment/Kassiopeia/tree/main/Kassiopeia/Bindings/Geometry/Source/KSGeoSideBuilder.cxx)<br>[*KSGeoSpaceBuilder.cxx*](https://github.com/KATRIN-Experiment/Kassiopeia/tree/main/Kassiopeia/Bindings/Geometry/Source/KSGeoSpaceBuilder.cxx)<br>[*KSGeoSurfaceBuilder.cxx*](https://github.com/KATRIN-Experiment/Kassiopeia/tree/main/Kassiopeia/Bindings/Geometry/Source/KSGeoSurfaceBuilder.cxx)|—    |—    |—    |—    |

#### KSCommandMemberAddTrackOutputData

|element name|source files|child elements|child types|attributes|attribute types|
|-----|-----|-----|-----|-----|-----|
|<a name="kscommandmemberaddtrackoutputdata">`add_track_output`</a>|[*KSGeoSideBuilder.cxx*](https://github.com/KATRIN-Experiment/Kassiopeia/tree/main/Kassiopeia/Bindings/Geometry/Source/KSGeoSideBuilder.cxx)<br>[*KSGeoSpaceBuilder.cxx*](https://github.com/KATRIN-Experiment/Kassiopeia/tree/main/Kassiopeia/Bindings/Geometry/Source/KSGeoSpaceBuilder.cxx)<br>[*KSGeoSurfaceBuilder.cxx*](https://github.com/KATRIN-Experiment/Kassiopeia/tree/main/Kassiopeia/Bindings/Geometry/Source/KSGeoSurfaceBuilder.cxx)|—    |—    |—    |—    |

#### KSCommandMemberClearSurfaceInteractionData

|element name|source files|child elements|child types|attributes|attribute types|
|-----|-----|-----|-----|-----|-----|
|<a name="kscommandmemberclearsurfaceinteractiondata">`clear_surface_interaction`</a>|[*KSGeoSideBuilder.cxx*](https://github.com/KATRIN-Experiment/Kassiopeia/tree/main/Kassiopeia/Bindings/Geometry/Source/KSGeoSideBuilder.cxx)<br>[*KSGeoSurfaceBuilder.cxx*](https://github.com/KATRIN-Experiment/Kassiopeia/tree/main/Kassiopeia/Bindings/Geometry/Source/KSGeoSurfaceBuilder.cxx)|—    |—    |—    |—    |

#### KSCommandMemberRemoveElectricFieldData

|element name|source files|child elements|child types|attributes|attribute types|
|-----|-----|-----|-----|-----|-----|
|<a name="kscommandmemberremoveelectricfielddata">`remove_magnetic_field`</a>|[*KSGeoSideBuilder.cxx*](https://github.com/KATRIN-Experiment/Kassiopeia/tree/main/Kassiopeia/Bindings/Geometry/Source/KSGeoSideBuilder.cxx)<br>[*KSGeoSpaceBuilder.cxx*](https://github.com/KATRIN-Experiment/Kassiopeia/tree/main/Kassiopeia/Bindings/Geometry/Source/KSGeoSpaceBuilder.cxx)<br>[*KSGeoSurfaceBuilder.cxx*](https://github.com/KATRIN-Experiment/Kassiopeia/tree/main/Kassiopeia/Bindings/Geometry/Source/KSGeoSurfaceBuilder.cxx)|—    |—    |—    |—    |

#### KSCommandMemberRemoveStepModifierData

|element name|source files|child elements|child types|attributes|attribute types|
|-----|-----|-----|-----|-----|-----|
|<a name="kscommandmemberremovestepmodifierdata">`remove_step_modifier`</a>|[*KSGeoSideBuilder.cxx*](https://github.com/KATRIN-Experiment/Kassiopeia/tree/main/Kassiopeia/Bindings/Geometry/Source/KSGeoSideBuilder.cxx)<br>[*KSGeoSpaceBuilder.cxx*](https://github.com/KATRIN-Experiment/Kassiopeia/tree/main/Kassiopeia/Bindings/Geometry/Source/KSGeoSpaceBuilder.cxx)<br>[*KSGeoSurfaceBuilder.cxx*](https://github.com/KATRIN-Experiment/Kassiopeia/tree/main/Kassiopeia/Bindings/Geometry/Source/KSGeoSurfaceBuilder.cxx)|—    |—    |—    |—    |

#### KSCommandMemberRemoveStepOutputData

|element name|source files|child elements|child types|attributes|attribute types|
|-----|-----|-----|-----|-----|-----|
|<a name="kscommandmemberremovestepoutputdata">`remove_step_output`</a>|[*KSGeoSideBuilder.cxx*](https://github.com/KATRIN-Experiment/Kassiopeia/tree/main/Kassiopeia/Bindings/Geometry/Source/KSGeoSideBuilder.cxx)<br>[*KSGeoSpaceBuilder.cxx*](https://github.com/KATRIN-Experiment/Kassiopeia/tree/main/Kassiopeia/Bindings/Geometry/Source/KSGeoSpaceBuilder.cxx)<br>[*KSGeoSurfaceBuilder.cxx*](https://github.com/KATRIN-Experiment/Kassiopeia/tree/main/Kassiopeia/Bindings/Geometry/Source/KSGeoSurfaceBuilder.cxx)|—    |—    |—    |—    |

#### KSCommandMemberRemoveTerminatorData

|element name|source files|child elements|child types|attributes|attribute types|
|-----|-----|-----|-----|-----|-----|
|<a name="kscommandmemberremoveterminatordata">`remove_terminator`</a>|[*KSGeoSideBuilder.cxx*](https://github.com/KATRIN-Experiment/Kassiopeia/tree/main/Kassiopeia/Bindings/Geometry/Source/KSGeoSideBuilder.cxx)<br>[*KSGeoSpaceBuilder.cxx*](https://github.com/KATRIN-Experiment/Kassiopeia/tree/main/Kassiopeia/Bindings/Geometry/Source/KSGeoSpaceBuilder.cxx)<br>[*KSGeoSurfaceBuilder.cxx*](https://github.com/KATRIN-Experiment/Kassiopeia/tree/main/Kassiopeia/Bindings/Geometry/Source/KSGeoSurfaceBuilder.cxx)|—    |—    |—    |—    |

#### KSCommandMemberRemoveTrackOutputData

|element name|source files|child elements|child types|attributes|attribute types|
|-----|-----|-----|-----|-----|-----|
|<a name="kscommandmemberremovetrackoutputdata">`remove_track_output`</a>|[*KSGeoSideBuilder.cxx*](https://github.com/KATRIN-Experiment/Kassiopeia/tree/main/Kassiopeia/Bindings/Geometry/Source/KSGeoSideBuilder.cxx)<br>[*KSGeoSpaceBuilder.cxx*](https://github.com/KATRIN-Experiment/Kassiopeia/tree/main/Kassiopeia/Bindings/Geometry/Source/KSGeoSpaceBuilder.cxx)<br>[*KSGeoSurfaceBuilder.cxx*](https://github.com/KATRIN-Experiment/Kassiopeia/tree/main/Kassiopeia/Bindings/Geometry/Source/KSGeoSurfaceBuilder.cxx)|—    |—    |—    |—    |

#### KSCommandMemberSetSurfaceInteractionData

|element name|source files|child elements|child types|attributes|attribute types|
|-----|-----|-----|-----|-----|-----|
|<a name="kscommandmembersetsurfaceinteractiondata">`set_surface_interaction`</a>|[*KSGeoSideBuilder.cxx*](https://github.com/KATRIN-Experiment/Kassiopeia/tree/main/Kassiopeia/Bindings/Geometry/Source/KSGeoSideBuilder.cxx)<br>[*KSGeoSurfaceBuilder.cxx*](https://github.com/KATRIN-Experiment/Kassiopeia/tree/main/Kassiopeia/Bindings/Geometry/Source/KSGeoSurfaceBuilder.cxx)|—    |—    |—    |—    |

### KSGeoSpace

|element name|source files|child elements|child types|attributes|attribute types|
|-----|-----|-----|-----|-----|-----|
|<a name="ksgeospace">`ksgeo_space`</a>|[*KSGeoSpaceBuilder.cxx*](https://github.com/KATRIN-Experiment/Kassiopeia/tree/main/Kassiopeia/Bindings/Geometry/Source/KSGeoSpaceBuilder.cxx)|[`add_control`](#kscommandmemberaddcontroldata)<br>[`add_electric_field`](#kscommandmemberaddelectricfielddata)<br>[`add_magnetic_field`](#kscommandmemberaddmagneticfielddata)<br>[`add_space_interaction`](#kscommandmemberaddspaceinteractiondata)<br>[`add_step_modifier`](#kscommandmemberaddstepmodifierdata)<br>[`add_step_output`](#kscommandmemberaddstepoutputdata)<br>[`add_term`](#kscommandmemberaddtermdata)<br>[`add_terminator`](#kscommandmemberaddterminatordata)<br>[`add_track_output`](#kscommandmemberaddtrackoutputdata)<br>[`clear_density`](#kscommandmembercleardensitydata)<br>[`clear_step_data`](#kscommandmemberclearstepdatadata)<br>[`clear_step_point`](#kscommandmemberclearsteppointdata)<br>[`clear_track_data`](#kscommandmembercleartrackdatadata)<br>[`clear_track_point`](#kscommandmembercleartrackpointdata)<br>[`clear_trajectory`](#kscommandmembercleartrajectorydata)<br>[`command`](#kscommandmemberdata)<br>[`geo_side`](#ksgeoside)<br>[`geo_space`](#ksgeospace)<br>[`geo_surface`](#ksgeosurface)<br>[`remove_control`](#kscommandmemberremovecontroldata)<br>[`remove_electric_field`](#kscommandmemberremoveelectricfielddata)<br>[`remove_magnetic_field`](#kscommandmemberremovemagneticfielddata)<br>[`remove_space_interaction`](#kscommandmemberremovespaceinteractiondata)<br>[`remove_step_modifier`](#kscommandmemberremovestepmodifierdata)<br>[`remove_step_output`](#kscommandmemberremovestepoutputdata)<br>[`remove_term`](#kscommandmemberremovetermdata)<br>[`remove_terminator`](#kscommandmemberremoveterminatordata)<br>[`remove_track_output`](#kscommandmemberremovetrackoutputdata)<br>[`set_density`](#kscommandmembersetdensitydata)<br>[`set_step_data`](#kscommandmembersetstepdatadata)<br>[`set_step_point`](#kscommandmembersetsteppointdata)<br>[`set_track_data`](#kscommandmembersettrackdatadata)<br>[`set_track_point`](#kscommandmembersettrackpointdata)<br>[`set_trajectory`](#kscommandmembersettrajectorydata)|*`KSCommandMemberAddControlData`*<br>*`KSCommandMemberAddElectricFieldData`*<br>*`KSCommandMemberAddMagneticFieldData`*<br>*`KSCommandMemberAddSpaceInteractionData`*<br>*`KSCommandMemberAddStepModifierData`*<br>*`KSCommandMemberAddStepOutputData`*<br>*`KSCommandMemberAddTermData`*<br>*`KSCommandMemberAddTerminatorData`*<br>*`KSCommandMemberAddTrackOutputData`*<br>*`KSCommandMemberClearDensityData`*<br>*`KSCommandMemberClearStepDataData`*<br>*`KSCommandMemberClearStepPointData`*<br>*`KSCommandMemberClearTrackDataData`*<br>*`KSCommandMemberClearTrackPointData`*<br>*`KSCommandMemberClearTrajectoryData`*<br>*`KSCommandMemberData`*<br>*`KSGeoSide`*<br>*`KSGeoSpace`*<br>*`KSGeoSurface`*<br>*`KSCommandMemberRemoveControlData`*<br>*`KSCommandMemberRemoveElectricFieldData`*<br>*`KSCommandMemberRemoveMagneticFieldData`*<br>*`KSCommandMemberRemoveSpaceInteractionData`*<br>*`KSCommandMemberRemoveStepModifierData`*<br>*`KSCommandMemberRemoveStepOutputData`*<br>*`KSCommandMemberRemoveTermData`*<br>*`KSCommandMemberRemoveTerminatorData`*<br>*`KSCommandMemberRemoveTrackOutputData`*<br>*`KSCommandMemberSetDensityData`*<br>*`KSCommandMemberSetStepDataData`*<br>*`KSCommandMemberSetStepPointData`*<br>*`KSCommandMemberSetTrackDataData`*<br>*`KSCommandMemberSetTrackPointData`*<br>*`KSCommandMemberSetTrajectoryData`*|`name`<br>`spaces`|*`string`*<br>*`string`*|

#### KSCommandMemberAddControlData

|element name|source files|child elements|child types|attributes|attribute types|
|-----|-----|-----|-----|-----|-----|
|<a name="kscommandmemberaddcontroldata">`add_control`</a>|[*KSGeoSpaceBuilder.cxx*](https://github.com/KATRIN-Experiment/Kassiopeia/tree/main/Kassiopeia/Bindings/Geometry/Source/KSGeoSpaceBuilder.cxx)|—    |—    |—    |—    |

#### KSCommandMemberAddSpaceInteractionData

|element name|source files|child elements|child types|attributes|attribute types|
|-----|-----|-----|-----|-----|-----|
|<a name="kscommandmemberaddspaceinteractiondata">`add_space_interaction`</a>|[*KSGeoSpaceBuilder.cxx*](https://github.com/KATRIN-Experiment/Kassiopeia/tree/main/Kassiopeia/Bindings/Geometry/Source/KSGeoSpaceBuilder.cxx)|—    |—    |—    |—    |

#### KSCommandMemberAddTermData

|element name|source files|child elements|child types|attributes|attribute types|
|-----|-----|-----|-----|-----|-----|
|<a name="kscommandmemberaddtermdata">`add_term`</a>|[*KSGeoSpaceBuilder.cxx*](https://github.com/KATRIN-Experiment/Kassiopeia/tree/main/Kassiopeia/Bindings/Geometry/Source/KSGeoSpaceBuilder.cxx)|—    |—    |—    |—    |

#### KSCommandMemberClearDensityData

|element name|source files|child elements|child types|attributes|attribute types|
|-----|-----|-----|-----|-----|-----|
|<a name="kscommandmembercleardensitydata">`clear_density`</a>|[*KSGeoSpaceBuilder.cxx*](https://github.com/KATRIN-Experiment/Kassiopeia/tree/main/Kassiopeia/Bindings/Geometry/Source/KSGeoSpaceBuilder.cxx)|—    |—    |—    |—    |

#### KSCommandMemberClearStepDataData

|element name|source files|child elements|child types|attributes|attribute types|
|-----|-----|-----|-----|-----|-----|
|<a name="kscommandmemberclearstepdatadata">`clear_step_data`</a>|[*KSGeoSpaceBuilder.cxx*](https://github.com/KATRIN-Experiment/Kassiopeia/tree/main/Kassiopeia/Bindings/Geometry/Source/KSGeoSpaceBuilder.cxx)|—    |—    |—    |—    |

#### KSCommandMemberClearStepPointData

|element name|source files|child elements|child types|attributes|attribute types|
|-----|-----|-----|-----|-----|-----|
|<a name="kscommandmemberclearsteppointdata">`clear_step_point`</a>|[*KSGeoSpaceBuilder.cxx*](https://github.com/KATRIN-Experiment/Kassiopeia/tree/main/Kassiopeia/Bindings/Geometry/Source/KSGeoSpaceBuilder.cxx)|—    |—    |—    |—    |

#### KSCommandMemberClearTrackDataData

|element name|source files|child elements|child types|attributes|attribute types|
|-----|-----|-----|-----|-----|-----|
|<a name="kscommandmembercleartrackdatadata">`clear_track_data`</a>|[*KSGeoSpaceBuilder.cxx*](https://github.com/KATRIN-Experiment/Kassiopeia/tree/main/Kassiopeia/Bindings/Geometry/Source/KSGeoSpaceBuilder.cxx)|—    |—    |—    |—    |

#### KSCommandMemberClearTrackPointData

|element name|source files|child elements|child types|attributes|attribute types|
|-----|-----|-----|-----|-----|-----|
|<a name="kscommandmembercleartrackpointdata">`clear_track_point`</a>|[*KSGeoSpaceBuilder.cxx*](https://github.com/KATRIN-Experiment/Kassiopeia/tree/main/Kassiopeia/Bindings/Geometry/Source/KSGeoSpaceBuilder.cxx)|—    |—    |—    |—    |

#### KSCommandMemberClearTrajectoryData

|element name|source files|child elements|child types|attributes|attribute types|
|-----|-----|-----|-----|-----|-----|
|<a name="kscommandmembercleartrajectorydata">`clear_trajectory`</a>|[*KSGeoSpaceBuilder.cxx*](https://github.com/KATRIN-Experiment/Kassiopeia/tree/main/Kassiopeia/Bindings/Geometry/Source/KSGeoSpaceBuilder.cxx)|—    |—    |—    |—    |

#### KSGeoSurface

|element name|source files|child elements|child types|attributes|attribute types|
|-----|-----|-----|-----|-----|-----|
|<a name="ksgeosurface">`geo_surface`</a>|[*KSGeoSpaceBuilder.cxx*](https://github.com/KATRIN-Experiment/Kassiopeia/tree/main/Kassiopeia/Bindings/Geometry/Source/KSGeoSpaceBuilder.cxx)<br>[*KSGeoSurfaceBuilder.cxx*](https://github.com/KATRIN-Experiment/Kassiopeia/tree/main/Kassiopeia/Bindings/Geometry/Source/KSGeoSurfaceBuilder.cxx)|[`add_electric_field`](#kscommandmemberaddelectricfielddata)<br>[`add_magnetic_field`](#kscommandmemberaddmagneticfielddata)<br>[`add_step_modifier`](#kscommandmemberaddstepmodifierdata)<br>[`add_step_output`](#kscommandmemberaddstepoutputdata)<br>[`add_terminator`](#kscommandmemberaddterminatordata)<br>[`add_track_output`](#kscommandmemberaddtrackoutputdata)<br>[`clear_surface_interaction`](#kscommandmemberclearsurfaceinteractiondata)<br>[`command`](#kscommandmemberdata)<br>[`remove_magnetic_field`](#kscommandmemberremoveelectricfielddata)<br>[`remove_step_modifier`](#kscommandmemberremovestepmodifierdata)<br>[`remove_step_output`](#kscommandmemberremovestepoutputdata)<br>[`remove_terminator`](#kscommandmemberremoveterminatordata)<br>[`remove_track_output`](#kscommandmemberremovetrackoutputdata)<br>[`set_surface_interaction`](#kscommandmembersetsurfaceinteractiondata)|*`KSCommandMemberAddElectricFieldData`*<br>*`KSCommandMemberAddMagneticFieldData`*<br>*`KSCommandMemberAddStepModifierData`*<br>*`KSCommandMemberAddStepOutputData`*<br>*`KSCommandMemberAddTerminatorData`*<br>*`KSCommandMemberAddTrackOutputData`*<br>*`KSCommandMemberClearSurfaceInteractionData`*<br>*`KSCommandMemberData`*<br>*`KSCommandMemberRemoveElectricFieldData`*<br>*`KSCommandMemberRemoveStepModifierData`*<br>*`KSCommandMemberRemoveStepOutputData`*<br>*`KSCommandMemberRemoveTerminatorData`*<br>*`KSCommandMemberRemoveTrackOutputData`*<br>*`KSCommandMemberSetSurfaceInteractionData`*|`name`<br>`spaces`<br>`surfaces`|*`string`*<br>*`string`*<br>*`string`*|

#### KSCommandMemberRemoveControlData

|element name|source files|child elements|child types|attributes|attribute types|
|-----|-----|-----|-----|-----|-----|
|<a name="kscommandmemberremovecontroldata">`remove_control`</a>|[*KSGeoSpaceBuilder.cxx*](https://github.com/KATRIN-Experiment/Kassiopeia/tree/main/Kassiopeia/Bindings/Geometry/Source/KSGeoSpaceBuilder.cxx)|—    |—    |—    |—    |

#### KSCommandMemberRemoveMagneticFieldData

|element name|source files|child elements|child types|attributes|attribute types|
|-----|-----|-----|-----|-----|-----|
|<a name="kscommandmemberremovemagneticfielddata">`remove_magnetic_field`</a>|[*KSGeoSideBuilder.cxx*](https://github.com/KATRIN-Experiment/Kassiopeia/tree/main/Kassiopeia/Bindings/Geometry/Source/KSGeoSideBuilder.cxx)<br>[*KSGeoSpaceBuilder.cxx*](https://github.com/KATRIN-Experiment/Kassiopeia/tree/main/Kassiopeia/Bindings/Geometry/Source/KSGeoSpaceBuilder.cxx)<br>[*KSGeoSurfaceBuilder.cxx*](https://github.com/KATRIN-Experiment/Kassiopeia/tree/main/Kassiopeia/Bindings/Geometry/Source/KSGeoSurfaceBuilder.cxx)|—    |—    |—    |—    |

#### KSCommandMemberRemoveSpaceInteractionData

|element name|source files|child elements|child types|attributes|attribute types|
|-----|-----|-----|-----|-----|-----|
|<a name="kscommandmemberremovespaceinteractiondata">`remove_space_interaction`</a>|[*KSGeoSpaceBuilder.cxx*](https://github.com/KATRIN-Experiment/Kassiopeia/tree/main/Kassiopeia/Bindings/Geometry/Source/KSGeoSpaceBuilder.cxx)|—    |—    |—    |—    |

#### KSCommandMemberRemoveTermData

|element name|source files|child elements|child types|attributes|attribute types|
|-----|-----|-----|-----|-----|-----|
|<a name="kscommandmemberremovetermdata">`remove_term`</a>|[*KSGeoSpaceBuilder.cxx*](https://github.com/KATRIN-Experiment/Kassiopeia/tree/main/Kassiopeia/Bindings/Geometry/Source/KSGeoSpaceBuilder.cxx)|—    |—    |—    |—    |

#### KSCommandMemberSetDensityData

|element name|source files|child elements|child types|attributes|attribute types|
|-----|-----|-----|-----|-----|-----|
|<a name="kscommandmembersetdensitydata">`set_density`</a>|[*KSGeoSpaceBuilder.cxx*](https://github.com/KATRIN-Experiment/Kassiopeia/tree/main/Kassiopeia/Bindings/Geometry/Source/KSGeoSpaceBuilder.cxx)|—    |—    |—    |—    |

#### KSCommandMemberSetStepDataData

|element name|source files|child elements|child types|attributes|attribute types|
|-----|-----|-----|-----|-----|-----|
|<a name="kscommandmembersetstepdatadata">`set_step_data`</a>|[*KSGeoSpaceBuilder.cxx*](https://github.com/KATRIN-Experiment/Kassiopeia/tree/main/Kassiopeia/Bindings/Geometry/Source/KSGeoSpaceBuilder.cxx)|—    |—    |—    |—    |

#### KSCommandMemberSetStepPointData

|element name|source files|child elements|child types|attributes|attribute types|
|-----|-----|-----|-----|-----|-----|
|<a name="kscommandmembersetsteppointdata">`set_step_point`</a>|[*KSGeoSpaceBuilder.cxx*](https://github.com/KATRIN-Experiment/Kassiopeia/tree/main/Kassiopeia/Bindings/Geometry/Source/KSGeoSpaceBuilder.cxx)|—    |—    |—    |—    |

#### KSCommandMemberSetTrackDataData

|element name|source files|child elements|child types|attributes|attribute types|
|-----|-----|-----|-----|-----|-----|
|<a name="kscommandmembersettrackdatadata">`set_track_data`</a>|[*KSGeoSpaceBuilder.cxx*](https://github.com/KATRIN-Experiment/Kassiopeia/tree/main/Kassiopeia/Bindings/Geometry/Source/KSGeoSpaceBuilder.cxx)|—    |—    |—    |—    |

#### KSCommandMemberSetTrackPointData

|element name|source files|child elements|child types|attributes|attribute types|
|-----|-----|-----|-----|-----|-----|
|<a name="kscommandmembersettrackpointdata">`set_track_point`</a>|[*KSGeoSpaceBuilder.cxx*](https://github.com/KATRIN-Experiment/Kassiopeia/tree/main/Kassiopeia/Bindings/Geometry/Source/KSGeoSpaceBuilder.cxx)|—    |—    |—    |—    |

#### KSCommandMemberSetTrajectoryData

|element name|source files|child elements|child types|attributes|attribute types|
|-----|-----|-----|-----|-----|-----|
|<a name="kscommandmembersettrajectorydata">`set_trajectory`</a>|[*KSGeoSpaceBuilder.cxx*](https://github.com/KATRIN-Experiment/Kassiopeia/tree/main/Kassiopeia/Bindings/Geometry/Source/KSGeoSpaceBuilder.cxx)|—    |—    |—    |—    |

### KSIntCalculatorConstant

|element name|source files|child elements|child types|attributes|attribute types|
|-----|-----|-----|-----|-----|-----|
|<a name="ksintcalculatorconstant">`ksint_calculator_constant`</a>|[*KSIntCalculatorConstantBuilder.cxx*](https://github.com/KATRIN-Experiment/Kassiopeia/tree/main/Kassiopeia/Bindings/Interactions/Source/KSIntCalculatorConstantBuilder.cxx)<br>[*KSIntScatteringBuilder.cxx*](https://github.com/KATRIN-Experiment/Kassiopeia/tree/main/Kassiopeia/Bindings/Interactions/Source/KSIntScatteringBuilder.cxx)|—    |—    |`cross_section`<br>`name`|*`double`*<br>*`string`*|

### KSIntCalculatorIon

|element name|source files|child elements|child types|attributes|attribute types|
|-----|-----|-----|-----|-----|-----|
|<a name="ksintcalculatorion">`ksint_calculator_ion`</a>|[*KSIntCalculatorIonBuilder.cxx*](https://github.com/KATRIN-Experiment/Kassiopeia/tree/main/Kassiopeia/Bindings/Interactions/Source/KSIntCalculatorIonBuilder.cxx)<br>[*KSIntScatteringBuilder.cxx*](https://github.com/KATRIN-Experiment/Kassiopeia/tree/main/Kassiopeia/Bindings/Interactions/Source/KSIntScatteringBuilder.cxx)|—    |—    |`gas`<br>`name`|*`string`*<br>*`string`*|

### KSIntDecay

|element name|source files|child elements|child types|attributes|attribute types|
|-----|-----|-----|-----|-----|-----|
|<a name="ksintdecay">`ksint_decay`</a>|[*KSIntDecayBuilder.cxx*](https://github.com/KATRIN-Experiment/Kassiopeia/tree/main/Kassiopeia/Bindings/Interactions/Source/KSIntDecayBuilder.cxx)|[`decay_death_const_rate`](#ksintdecaycalculatordeathconstrate)<br>[`decay_ferenc_bbr`](#ksintdecaycalculatorferencbbrtransition)<br>[`decay_ferenc_ionisation`](#ksintdecaycalculatorferencionisation)<br>[`decay_ferenc_spontaneous`](#ksintdecaycalculatorferencspontaneous)<br>[`decay_glukhov_deexcitation`](#ksintdecaycalculatorglukhovdeexcitation)<br>[`decay_glukhov_excitation`](#ksintdecaycalculatorglukhovexcitation)<br>[`decay_glukhov_ionisation`](#ksintdecaycalculatorglukhovionisation)<br>[`decay_glukhov_spontaneous`](#ksintdecaycalculatorglukhovspontaneous)|*`KSIntDecayCalculatorDeathConstRate`*<br>*`KSIntDecayCalculatorFerencBBRTransition`*<br>*`KSIntDecayCalculatorFerencIonisation`*<br>*`KSIntDecayCalculatorFerencSpontaneous`*<br>*`KSIntDecayCalculatorGlukhovDeExcitation`*<br>*`KSIntDecayCalculatorGlukhovExcitation`*<br>*`KSIntDecayCalculatorGlukhovIonisation`*<br>*`KSIntDecayCalculatorGlukhovSpontaneous`*|`calculator`<br>`calculators`<br>`enhancement`<br>`name`|*`string`*<br>*`string`*<br>*`double`*<br>*`string`*|

#### KSIntDecayCalculatorDeathConstRate

|element name|source files|child elements|child types|attributes|attribute types|
|-----|-----|-----|-----|-----|-----|
|<a name="ksintdecaycalculatordeathconstrate">`decay_death_const_rate`</a>|[*KSIntDecayBuilder.cxx*](https://github.com/KATRIN-Experiment/Kassiopeia/tree/main/Kassiopeia/Bindings/Interactions/Source/KSIntDecayBuilder.cxx)<br>[*KSIntDecayCalculatorDeathConstRateBuilder.cxx*](https://github.com/KATRIN-Experiment/Kassiopeia/tree/main/Kassiopeia/Bindings/Interactions/Source/KSIntDecayCalculatorDeathConstRateBuilder.cxx)|[`decay_product_generator`](#ksgengeneratorcomposite)|*`KSGenGeneratorComposite`*|`life_time`<br>`max_pid`<br>`min_pid`<br>`name`<br>`target_pid`|*`double`*<br>*`long long`*<br>*`long long`*<br>*`string`*<br>*`long long`*|

#### KSIntDecayCalculatorFerencBBRTransition

|element name|source files|child elements|child types|attributes|attribute types|
|-----|-----|-----|-----|-----|-----|
|<a name="ksintdecaycalculatorferencbbrtransition">`decay_ferenc_bbr`</a>|[*KSIntDecayBuilder.cxx*](https://github.com/KATRIN-Experiment/Kassiopeia/tree/main/Kassiopeia/Bindings/Interactions/Source/KSIntDecayBuilder.cxx)<br>[*KSIntDecayCalculatorFerencBBRTransitionBuilder.cxx*](https://github.com/KATRIN-Experiment/Kassiopeia/tree/main/Kassiopeia/Bindings/Interactions/Source/KSIntDecayCalculatorFerencBBRTransitionBuilder.cxx)|[`decay_product_generator`](#ksgengeneratorcomposite)|*`KSGenGeneratorComposite`*|`max_pid`<br>`min_pid`<br>`name`<br>`target_pid`<br>`temperature`|*`long long`*<br>*`long long`*<br>*`string`*<br>*`long long`*<br>*`double`*|

#### KSIntDecayCalculatorFerencIonisation

|element name|source files|child elements|child types|attributes|attribute types|
|-----|-----|-----|-----|-----|-----|
|<a name="ksintdecaycalculatorferencionisation">`decay_ferenc_ionisation`</a>|[*KSIntDecayBuilder.cxx*](https://github.com/KATRIN-Experiment/Kassiopeia/tree/main/Kassiopeia/Bindings/Interactions/Source/KSIntDecayBuilder.cxx)<br>[*KSIntDecayCalculatorFerencIonisationBuilder.cxx*](https://github.com/KATRIN-Experiment/Kassiopeia/tree/main/Kassiopeia/Bindings/Interactions/Source/KSIntDecayCalculatorFerencIonisationBuilder.cxx)|[`decay_product_generator`](#ksgengeneratorcomposite)|*`KSGenGeneratorComposite`*|`max_pid`<br>`min_pid`<br>`name`<br>`target_pid`<br>`temperature`|*`long long`*<br>*`long long`*<br>*`string`*<br>*`long long`*<br>*`double`*|

#### KSIntDecayCalculatorFerencSpontaneous

|element name|source files|child elements|child types|attributes|attribute types|
|-----|-----|-----|-----|-----|-----|
|<a name="ksintdecaycalculatorferencspontaneous">`decay_ferenc_spontaneous`</a>|[*KSIntDecayBuilder.cxx*](https://github.com/KATRIN-Experiment/Kassiopeia/tree/main/Kassiopeia/Bindings/Interactions/Source/KSIntDecayBuilder.cxx)<br>[*KSIntDecayCalculatorFerencSpontaneousBuilder.cxx*](https://github.com/KATRIN-Experiment/Kassiopeia/tree/main/Kassiopeia/Bindings/Interactions/Source/KSIntDecayCalculatorFerencSpontaneousBuilder.cxx)|[`decay_product_generator`](#ksgengeneratorcomposite)|*`KSGenGeneratorComposite`*|`max_pid`<br>`min_pid`<br>`name`<br>`target_pid`|*`long long`*<br>*`long long`*<br>*`string`*<br>*`long long`*|

#### KSIntDecayCalculatorGlukhovDeExcitation

|element name|source files|child elements|child types|attributes|attribute types|
|-----|-----|-----|-----|-----|-----|
|<a name="ksintdecaycalculatorglukhovdeexcitation">`decay_glukhov_deexcitation`</a>|[*KSIntDecayBuilder.cxx*](https://github.com/KATRIN-Experiment/Kassiopeia/tree/main/Kassiopeia/Bindings/Interactions/Source/KSIntDecayBuilder.cxx)<br>[*KSIntDecayCalculatorGlukhovDeExcitationBuilder.cxx*](https://github.com/KATRIN-Experiment/Kassiopeia/tree/main/Kassiopeia/Bindings/Interactions/Source/KSIntDecayCalculatorGlukhovDeExcitationBuilder.cxx)|[`decay_product_generator`](#ksgengeneratorcomposite)|*`KSGenGeneratorComposite`*|`max_pid`<br>`min_pid`<br>`name`<br>`target_pid`<br>`temperature`|*`long long`*<br>*`long long`*<br>*`string`*<br>*`long long`*<br>*`double`*|

#### KSIntDecayCalculatorGlukhovExcitation

|element name|source files|child elements|child types|attributes|attribute types|
|-----|-----|-----|-----|-----|-----|
|<a name="ksintdecaycalculatorglukhovexcitation">`decay_glukhov_excitation`</a>|[*KSIntDecayBuilder.cxx*](https://github.com/KATRIN-Experiment/Kassiopeia/tree/main/Kassiopeia/Bindings/Interactions/Source/KSIntDecayBuilder.cxx)<br>[*KSIntDecayCalculatorGlukhovExcitationBuilder.cxx*](https://github.com/KATRIN-Experiment/Kassiopeia/tree/main/Kassiopeia/Bindings/Interactions/Source/KSIntDecayCalculatorGlukhovExcitationBuilder.cxx)|[`decay_product_generator`](#ksgengeneratorcomposite)|*`KSGenGeneratorComposite`*|`max_pid`<br>`min_pid`<br>`name`<br>`target_pid`<br>`temperature`|*`long long`*<br>*`long long`*<br>*`string`*<br>*`long long`*<br>*`double`*|

#### KSIntDecayCalculatorGlukhovIonisation

|element name|source files|child elements|child types|attributes|attribute types|
|-----|-----|-----|-----|-----|-----|
|<a name="ksintdecaycalculatorglukhovionisation">`decay_glukhov_ionisation`</a>|[*KSIntDecayBuilder.cxx*](https://github.com/KATRIN-Experiment/Kassiopeia/tree/main/Kassiopeia/Bindings/Interactions/Source/KSIntDecayBuilder.cxx)<br>[*KSIntDecayCalculatorGlukhovIonisationBuilder.cxx*](https://github.com/KATRIN-Experiment/Kassiopeia/tree/main/Kassiopeia/Bindings/Interactions/Source/KSIntDecayCalculatorGlukhovIonisationBuilder.cxx)|[`decay_product_generator`](#ksgengeneratorcomposite)|*`KSGenGeneratorComposite`*|`max_pid`<br>`min_pid`<br>`name`<br>`target_pid`<br>`temperature`|*`long long`*<br>*`long long`*<br>*`string`*<br>*`long long`*<br>*`double`*|

#### KSIntDecayCalculatorGlukhovSpontaneous

|element name|source files|child elements|child types|attributes|attribute types|
|-----|-----|-----|-----|-----|-----|
|<a name="ksintdecaycalculatorglukhovspontaneous">`decay_glukhov_spontaneous`</a>|[*KSIntDecayBuilder.cxx*](https://github.com/KATRIN-Experiment/Kassiopeia/tree/main/Kassiopeia/Bindings/Interactions/Source/KSIntDecayBuilder.cxx)<br>[*KSIntDecayCalculatorGlukhovSpontaneousBuilder.cxx*](https://github.com/KATRIN-Experiment/Kassiopeia/tree/main/Kassiopeia/Bindings/Interactions/Source/KSIntDecayCalculatorGlukhovSpontaneousBuilder.cxx)|[`decay_product_generator`](#ksgengeneratorcomposite)|*`KSGenGeneratorComposite`*|`max_pid`<br>`min_pid`<br>`name`<br>`target_pid`|*`long long`*<br>*`long long`*<br>*`string`*<br>*`long long`*|

### KSIntDensityConstant

|element name|source files|child elements|child types|attributes|attribute types|
|-----|-----|-----|-----|-----|-----|
|<a name="ksintdensityconstant">`ksint_density_constant`</a>|[*KSIntDensityConstantBuilder.cxx*](https://github.com/KATRIN-Experiment/Kassiopeia/tree/main/Kassiopeia/Bindings/Interactions/Source/KSIntDensityConstantBuilder.cxx)<br>[*KSIntScatteringBuilder.cxx*](https://github.com/KATRIN-Experiment/Kassiopeia/tree/main/Kassiopeia/Bindings/Interactions/Source/KSIntScatteringBuilder.cxx)|—    |—    |`density`<br>`name`<br>`pressure`<br>`pressure_mbar`<br>`temperature`|*`double`*<br>*`string`*<br>*`double`*<br>*`double`*<br>*`double`*|

### KSIntScattering

|element name|source files|child elements|child types|attributes|attribute types|
|-----|-----|-----|-----|-----|-----|
|<a name="ksintscattering">`ksint_scattering`</a>|[*KSIntScatteringBuilder.cxx*](https://github.com/KATRIN-Experiment/Kassiopeia/tree/main/Kassiopeia/Bindings/Interactions/Source/KSIntScatteringBuilder.cxx)|[`calculator_argon`](#ksintcalculatorargonset)<br>[`calculator_constant`](#ksintcalculatorconstant)<br>[`calculator_hydrogen`](#ksintcalculatorhydrogenset)<br>[`calculator_ion`](#ksintcalculatorion)<br>[`calculator_kess`](#ksintcalculatorkessset)<br>[`density_constant`](#ksintdensityconstant)|*`KSIntCalculatorArgonSet`*<br>*`KSIntCalculatorConstant`*<br>*`KSIntCalculatorHydrogenSet`*<br>*`KSIntCalculatorIon`*<br>*`KSIntCalculatorKESSSet`*<br>*`KSIntDensityConstant`*|`calculator`<br>`calculators`<br>`density`<br>`enhancement`<br>`name`<br>`split`|*`string`*<br>*`string`*<br>*`string`*<br>*`double`*<br>*`string`*<br>*`bool`*|

#### KSIntCalculatorArgonSet

|element name|source files|child elements|child types|attributes|attribute types|
|-----|-----|-----|-----|-----|-----|
|<a name="ksintcalculatorargonset">`calculator_argon`</a>|[*KSIntScatteringBuilder.cxx*](https://github.com/KATRIN-Experiment/Kassiopeia/tree/main/Kassiopeia/Bindings/Interactions/Source/KSIntScatteringBuilder.cxx)|—    |—    |`double_ionisation`<br>`elastic`<br>`excitation`<br>`name`<br>`single_ionisation`|*`bool`*<br>*`bool`*<br>*`bool`*<br>*`string`*<br>*`bool`*|

#### KSIntCalculatorHydrogenSet

|element name|source files|child elements|child types|attributes|attribute types|
|-----|-----|-----|-----|-----|-----|
|<a name="ksintcalculatorhydrogenset">`calculator_hydrogen`</a>|[*KSIntScatteringBuilder.cxx*](https://github.com/KATRIN-Experiment/Kassiopeia/tree/main/Kassiopeia/Bindings/Interactions/Source/KSIntScatteringBuilder.cxx)|—    |—    |`elastic`<br>`excitation`<br>`ionisation`<br>`molecule`<br>`name`|*`bool`*<br>*`bool`*<br>*`bool`*<br>*`string`*<br>*`string`*|

#### KSIntCalculatorKESSSet

|element name|source files|child elements|child types|attributes|attribute types|
|-----|-----|-----|-----|-----|-----|
|<a name="ksintcalculatorkessset">`calculator_kess`</a>|[*KSIntScatteringBuilder.cxx*](https://github.com/KATRIN-Experiment/Kassiopeia/tree/main/Kassiopeia/Bindings/Interactions/Source/KSIntScatteringBuilder.cxx)|—    |—    |`auger_relaxation`<br>`elastic`<br>`inelastic`<br>`name`<br>`photo_absorbtion`|*`string`*<br>*`bool`*<br>*`string`*<br>*`string`*<br>*`bool`*|

### KSIntSpinFlip

|element name|source files|child elements|child types|attributes|attribute types|
|-----|-----|-----|-----|-----|-----|
|<a name="ksintspinflip">`ksint_spin_flip`</a>|[*KSIntSpinFlipBuilder.cxx*](https://github.com/KATRIN-Experiment/Kassiopeia/tree/main/Kassiopeia/Bindings/Interactions/Source/KSIntSpinFlipBuilder.cxx)|—    |—    |`name`|*`string`*|

### KSIntSpinFlipPulse

|element name|source files|child elements|child types|attributes|attribute types|
|-----|-----|-----|-----|-----|-----|
|<a name="ksintspinflippulse">`ksint_spin_flip_pulse`</a>|[*KSIntSpinFlipPulseBuilder.cxx*](https://github.com/KATRIN-Experiment/Kassiopeia/tree/main/Kassiopeia/Bindings/Interactions/Source/KSIntSpinFlipPulseBuilder.cxx)|—    |—    |`name`<br>`time`|*`string`*<br>*`double`*|

### KSIntSpinRotateYPulse

|element name|source files|child elements|child types|attributes|attribute types|
|-----|-----|-----|-----|-----|-----|
|<a name="ksintspinrotateypulse">`ksint_spin_rotate_y_pulse`</a>|[*KSIntSpinRotateYPulseBuilder.cxx*](https://github.com/KATRIN-Experiment/Kassiopeia/tree/main/Kassiopeia/Bindings/Interactions/Source/KSIntSpinRotateYPulseBuilder.cxx)|—    |—    |`angle`<br>`is_adiabatic`<br>`name`<br>`time`|*`double`*<br>*`bool`*<br>*`string`*<br>*`double`*|

### KSIntSurfaceUCN

|element name|source files|child elements|child types|attributes|attribute types|
|-----|-----|-----|-----|-----|-----|
|<a name="ksintsurfaceucn">`ksint_surface_UCN`</a>|[*KSIntSurfaceUCNBuilder.cxx*](https://github.com/KATRIN-Experiment/Kassiopeia/tree/main/Kassiopeia/Bindings/Interactions/Source/KSIntSurfaceUCNBuilder.cxx)|—    |—    |`alpha`<br>`correlation_length`<br>`eta`<br>`name`<br>`real_optical_potential`|*`double`*<br>*`double`*<br>*`double`*<br>*`string`*<br>*`double`*|

### KSIntSurfaceDiffuse

|element name|source files|child elements|child types|attributes|attribute types|
|-----|-----|-----|-----|-----|-----|
|<a name="ksintsurfacediffuse">`ksint_surface_diffuse`</a>|[*KSIntSurfaceDiffuseBuilder.cxx*](https://github.com/KATRIN-Experiment/Kassiopeia/tree/main/Kassiopeia/Bindings/Interactions/Source/KSIntSurfaceDiffuseBuilder.cxx)|—    |—    |`name`<br>`probability`<br>`reflection_loss`<br>`reflection_loss_fraction`<br>`transmission_loss`<br>`transmission_loss_fraction`|*`string`*<br>*`double`*<br>*`double`*<br>*`double`*<br>*`double`*<br>*`double`*|

### KSIntSurfaceMultiplication

|element name|source files|child elements|child types|attributes|attribute types|
|-----|-----|-----|-----|-----|-----|
|<a name="ksintsurfacemultiplication">`ksint_surface_multiplication`</a>|[*KSIntSurfaceMultiplicationBuilder.cxx*](https://github.com/KATRIN-Experiment/Kassiopeia/tree/main/Kassiopeia/Bindings/Interactions/Source/KSIntSurfaceMultiplicationBuilder.cxx)|—    |—    |`energy_loss_fraction`<br>`name`<br>`required_energy_per_particle_ev`<br>`side`|*`double`*<br>*`string`*<br>*`double`*<br>*`string`*|

### KSIntSurfaceSpecular

|element name|source files|child elements|child types|attributes|attribute types|
|-----|-----|-----|-----|-----|-----|
|<a name="ksintsurfacespecular">`ksint_surface_specular`</a>|[*KSIntSurfaceSpecularBuilder.cxx*](https://github.com/KATRIN-Experiment/Kassiopeia/tree/main/Kassiopeia/Bindings/Interactions/Source/KSIntSurfaceSpecularBuilder.cxx)|—    |—    |`name`<br>`probability`<br>`reflection_loss`<br>`reflection_loss_fraction`<br>`transmission_loss`<br>`transmission_loss_fraction`|*`string`*<br>*`double`*<br>*`double`*<br>*`double`*<br>*`double`*<br>*`double`*|

### KSIntSurfaceSpinFlip

|element name|source files|child elements|child types|attributes|attribute types|
|-----|-----|-----|-----|-----|-----|
|<a name="ksintsurfacespinflip">`ksint_surface_spin_flip`</a>|[*KSIntSurfaceSpinFlipBuilder.cxx*](https://github.com/KATRIN-Experiment/Kassiopeia/tree/main/Kassiopeia/Bindings/Interactions/Source/KSIntSurfaceSpinFlipBuilder.cxx)|—    |—    |`name`<br>`probability`|*`string`*<br>*`double`*|

### KSModDynamicEnhancement

|element name|source files|child elements|child types|attributes|attribute types|
|-----|-----|-----|-----|-----|-----|
|<a name="ksmoddynamicenhancement">`ksmod_dynamic_enhancement`</a>|[*KSModDynamicEnhancementBuilder.cxx*](https://github.com/KATRIN-Experiment/Kassiopeia/tree/main/Kassiopeia/Bindings/Modifiers/Source/KSModDynamicEnhancementBuilder.cxx)|—    |—    |`dynamic`<br>`name`<br>`reference_energy`<br>`scattering`<br>`static_enhancement`<br>`synchrotron`|*`bool`*<br>*`string`*<br>*`double`*<br>*`string`*<br>*`double`*<br>*`string`*|

### KSModEventReport

|element name|source files|child elements|child types|attributes|attribute types|
|-----|-----|-----|-----|-----|-----|
|<a name="ksmodeventreport">`ksmod_event_report`</a>|[*KSModEventReportBuilder.cxx*](https://github.com/KATRIN-Experiment/Kassiopeia/tree/main/Kassiopeia/Bindings/Modifiers/Source/KSModEventReportBuilder.cxx)|—    |—    |`name`|*`string`*|

### KSModSplitOnTurn

|element name|source files|child elements|child types|attributes|attribute types|
|-----|-----|-----|-----|-----|-----|
|<a name="ksmodsplitonturn">`ksmod_split_on_turn`</a>|[*KSModSplitOnTurnBuilder.cxx*](https://github.com/KATRIN-Experiment/Kassiopeia/tree/main/Kassiopeia/Bindings/Modifiers/Source/KSModSplitOnTurnBuilder.cxx)|—    |—    |`direction`<br>`name`|*`string`*<br>*`string`*|

### KSNavMeshedSpace

|element name|source files|child elements|child types|attributes|attribute types|
|-----|-----|-----|-----|-----|-----|
|<a name="ksnavmeshedspace">`ksnav_meshed_space`</a>|[*KSNavMeshedSpaceBuilder.cxx*](https://github.com/KATRIN-Experiment/Kassiopeia/tree/main/Kassiopeia/Bindings/Navigators/Source/KSNavMeshedSpaceBuilder.cxx)|—    |—    |`absolute_tolerance`<br>`enter_split`<br>`exit_split`<br>`fail_check`<br>`max_octree_depth`<br>`n_allowed_elements`<br>`name`<br>`octree_file`<br>`path`<br>`relative_tolerance`<br>`root_space`<br>`spatial_resolution`|*`double`*<br>*`bool`*<br>*`bool`*<br>*`bool`*<br>*`unsigned int`*<br>*`unsigned int`*<br>*`string`*<br>*`string`*<br>*`string`*<br>*`double`*<br>*`string`*<br>*`double`*|

### KSNavSpace

|element name|source files|child elements|child types|attributes|attribute types|
|-----|-----|-----|-----|-----|-----|
|<a name="ksnavspace">`ksnav_space`</a>|[*KSNavSpaceBuilder.cxx*](https://github.com/KATRIN-Experiment/Kassiopeia/tree/main/Kassiopeia/Bindings/Navigators/Source/KSNavSpaceBuilder.cxx)|—    |—    |`enter_split`<br>`exit_split`<br>`fail_check`<br>`name`<br>`tolerance`|*`bool`*<br>*`bool`*<br>*`bool`*<br>*`string`*<br>*`double`*|

### KSNavSurface

|element name|source files|child elements|child types|attributes|attribute types|
|-----|-----|-----|-----|-----|-----|
|<a name="ksnavsurface">`ksnav_surface`</a>|[*KSNavSurfaceBuilder.cxx*](https://github.com/KATRIN-Experiment/Kassiopeia/tree/main/Kassiopeia/Bindings/Navigators/Source/KSNavSurfaceBuilder.cxx)|—    |—    |`name`<br>`reflection_split`<br>`transmission_split`|*`string`*<br>*`bool`*<br>*`bool`*|

### KSTermDeath

|element name|source files|child elements|child types|attributes|attribute types|
|-----|-----|-----|-----|-----|-----|
|<a name="kstermdeath">`ksterm_death`</a>|[*KSTermDeathBuilder.cxx*](https://github.com/KATRIN-Experiment/Kassiopeia/tree/main/Kassiopeia/Bindings/Terminators/Source/KSTermDeathBuilder.cxx)|—    |—    |`name`|*`string`*|

### KSTermMagnetron

|element name|source files|child elements|child types|attributes|attribute types|
|-----|-----|-----|-----|-----|-----|
|<a name="kstermmagnetron">`ksterm_magnetron`</a>|[*KSTermMagnetronBuilder.cxx*](https://github.com/KATRIN-Experiment/Kassiopeia/tree/main/Kassiopeia/Bindings/Terminators/Source/KSTermMagnetronBuilder.cxx)|—    |—    |`max_phi`<br>`name`|*`double`*<br>*`string`*|

### KSTermMaxEnergy

|element name|source files|child elements|child types|attributes|attribute types|
|-----|-----|-----|-----|-----|-----|
|<a name="kstermmaxenergy">`ksterm_max_energy`</a>|[*KSTermMaxEnergyBuilder.cxx*](https://github.com/KATRIN-Experiment/Kassiopeia/tree/main/Kassiopeia/Bindings/Terminators/Source/KSTermMaxEnergyBuilder.cxx)|—    |—    |`energy`<br>`name`|*`double`*<br>*`string`*|

### KSTermMaxLength

|element name|source files|child elements|child types|attributes|attribute types|
|-----|-----|-----|-----|-----|-----|
|<a name="kstermmaxlength">`ksterm_max_length`</a>|[*KSTermMaxLengthBuilder.cxx*](https://github.com/KATRIN-Experiment/Kassiopeia/tree/main/Kassiopeia/Bindings/Terminators/Source/KSTermMaxLengthBuilder.cxx)|—    |—    |`length`<br>`name`|*`double`*<br>*`string`*|

### KSTermMaxLongEnergy

|element name|source files|child elements|child types|attributes|attribute types|
|-----|-----|-----|-----|-----|-----|
|<a name="kstermmaxlongenergy">`ksterm_max_long_energy`</a>|[*KSTermMaxLongEnergyBuilder.cxx*](https://github.com/KATRIN-Experiment/Kassiopeia/tree/main/Kassiopeia/Bindings/Terminators/Source/KSTermMaxLongEnergyBuilder.cxx)|—    |—    |`long_energy`<br>`name`|*`double`*<br>*`string`*|

### KSTermMaxR

|element name|source files|child elements|child types|attributes|attribute types|
|-----|-----|-----|-----|-----|-----|
|<a name="kstermmaxr">`ksterm_max_r`</a>|[*KSTermMaxRBuilder.cxx*](https://github.com/KATRIN-Experiment/Kassiopeia/tree/main/Kassiopeia/Bindings/Terminators/Source/KSTermMaxRBuilder.cxx)|—    |—    |`name`<br>`r`|*`string`*<br>*`double`*|

### KSTermMaxStepTime

|element name|source files|child elements|child types|attributes|attribute types|
|-----|-----|-----|-----|-----|-----|
|<a name="kstermmaxsteptime">`ksterm_max_step_time`</a>|[*KSTermMaxStepTimeBuilder.cxx*](https://github.com/KATRIN-Experiment/Kassiopeia/tree/main/Kassiopeia/Bindings/Terminators/Source/KSTermMaxStepTimeBuilder.cxx)|—    |—    |`name`<br>`time`|*`string`*<br>*`double`*|

### KSTermMaxSteps

|element name|source files|child elements|child types|attributes|attribute types|
|-----|-----|-----|-----|-----|-----|
|<a name="kstermmaxsteps">`ksterm_max_steps`</a>|[*KSTermMaxStepsBuilder.cxx*](https://github.com/KATRIN-Experiment/Kassiopeia/tree/main/Kassiopeia/Bindings/Terminators/Source/KSTermMaxStepsBuilder.cxx)|—    |—    |`name`<br>`steps`|*`string`*<br>*`unsigned int`*|

### KSTermMaxTime

|element name|source files|child elements|child types|attributes|attribute types|
|-----|-----|-----|-----|-----|-----|
|<a name="kstermmaxtime">`ksterm_max_time`</a>|[*KSTermMaxTimeBuilder.cxx*](https://github.com/KATRIN-Experiment/Kassiopeia/tree/main/Kassiopeia/Bindings/Terminators/Source/KSTermMaxTimeBuilder.cxx)|—    |—    |`name`<br>`time`|*`string`*<br>*`double`*|

### KSTermMaxTotalTime

|element name|source files|child elements|child types|attributes|attribute types|
|-----|-----|-----|-----|-----|-----|
|<a name="kstermmaxtotaltime">`ksterm_max_total_time`</a>|[*KSTermMaxTotalTimeBuilder.cxx*](https://github.com/KATRIN-Experiment/Kassiopeia/tree/main/Kassiopeia/Bindings/Terminators/Source/KSTermMaxTotalTimeBuilder.cxx)|—    |—    |`name`<br>`time`|*`string`*<br>*`double`*|

### KSTermMaxZ

|element name|source files|child elements|child types|attributes|attribute types|
|-----|-----|-----|-----|-----|-----|
|<a name="kstermmaxz">`ksterm_max_z`</a>|[*KSTermMaxZBuilder.cxx*](https://github.com/KATRIN-Experiment/Kassiopeia/tree/main/Kassiopeia/Bindings/Terminators/Source/KSTermMaxZBuilder.cxx)|—    |—    |`name`<br>`z`|*`string`*<br>*`double`*|

### KSTermMinDistance

|element name|source files|child elements|child types|attributes|attribute types|
|-----|-----|-----|-----|-----|-----|
|<a name="kstermmindistance">`ksterm_min_distance`</a>|[*KSTermMinDistanceBuilder.cxx*](https://github.com/KATRIN-Experiment/Kassiopeia/tree/main/Kassiopeia/Bindings/Terminators/Source/KSTermMinDistanceBuilder.cxx)|—    |—    |`min_distance`<br>`name`<br>`surfaces`|*`double`*<br>*`string`*<br>*`string`*|

### KSTermMinEnergy

|element name|source files|child elements|child types|attributes|attribute types|
|-----|-----|-----|-----|-----|-----|
|<a name="kstermminenergy">`ksterm_min_energy`</a>|[*KSTermMinEnergyBuilder.cxx*](https://github.com/KATRIN-Experiment/Kassiopeia/tree/main/Kassiopeia/Bindings/Terminators/Source/KSTermMinEnergyBuilder.cxx)|—    |—    |`energy`<br>`name`|*`double`*<br>*`string`*|

### KSTermMinLongEnergy

|element name|source files|child elements|child types|attributes|attribute types|
|-----|-----|-----|-----|-----|-----|
|<a name="kstermminlongenergy">`ksterm_min_long_energy`</a>|[*KSTermMinLongEnergyBuilder.cxx*](https://github.com/KATRIN-Experiment/Kassiopeia/tree/main/Kassiopeia/Bindings/Terminators/Source/KSTermMinLongEnergyBuilder.cxx)|—    |—    |`long_energy`<br>`name`|*`double`*<br>*`string`*|

### KSTermMinR

|element name|source files|child elements|child types|attributes|attribute types|
|-----|-----|-----|-----|-----|-----|
|<a name="kstermminr">`ksterm_min_r`</a>|[*KSTermMinRBuilder.cxx*](https://github.com/KATRIN-Experiment/Kassiopeia/tree/main/Kassiopeia/Bindings/Terminators/Source/KSTermMinRBuilder.cxx)|—    |—    |`name`<br>`r`|*`string`*<br>*`double`*|

### KSTermMinZ

|element name|source files|child elements|child types|attributes|attribute types|
|-----|-----|-----|-----|-----|-----|
|<a name="kstermminz">`ksterm_min_z`</a>|[*KSTermMinZBuilder.cxx*](https://github.com/KATRIN-Experiment/Kassiopeia/tree/main/Kassiopeia/Bindings/Terminators/Source/KSTermMinZBuilder.cxx)|—    |—    |`name`<br>`z`|*`string`*<br>*`double`*|

### KSTermOutputData

|element name|source files|child elements|child types|attributes|attribute types|
|-----|-----|-----|-----|-----|-----|
|<a name="kstermoutputdata">`ksterm_output`</a>|[*KSTermOutputBuilder.cxx*](https://github.com/KATRIN-Experiment/Kassiopeia/tree/main/Kassiopeia/Bindings/Terminators/Source/KSTermOutputBuilder.cxx)|—    |—    |`component`<br>`group`<br>`max_value`<br>`min_value`<br>`name`|*`string`*<br>*`string`*<br>*`double`*<br>*`double`*<br>*`string`*|

### KSTermSecondaries

|element name|source files|child elements|child types|attributes|attribute types|
|-----|-----|-----|-----|-----|-----|
|<a name="kstermsecondaries">`ksterm_secondaries`</a>|[*KSTermSecondariesBuilder.cxx*](https://github.com/KATRIN-Experiment/Kassiopeia/tree/main/Kassiopeia/Bindings/Terminators/Source/KSTermSecondariesBuilder.cxx)|—    |—    |`name`|*`string`*|

### KSTermStepsize

|element name|source files|child elements|child types|attributes|attribute types|
|-----|-----|-----|-----|-----|-----|
|<a name="kstermstepsize">`ksterm_stepsize`</a>|[*KSTermStepsizeBuilder.cxx*](https://github.com/KATRIN-Experiment/Kassiopeia/tree/main/Kassiopeia/Bindings/Terminators/Source/KSTermStepsizeBuilder.cxx)|—    |—    |`max_length`<br>`min_length`<br>`name`|*`double`*<br>*`double`*<br>*`string`*|

### KSTermTrapped

|element name|source files|child elements|child types|attributes|attribute types|
|-----|-----|-----|-----|-----|-----|
|<a name="kstermtrapped">`ksterm_trapped`</a>|[*KSTermTrappedBuilder.cxx*](https://github.com/KATRIN-Experiment/Kassiopeia/tree/main/Kassiopeia/Bindings/Terminators/Source/KSTermTrappedBuilder.cxx)|—    |—    |`max_turns`<br>`name`<br>`use_electric_field`<br>`use_magnetic_field`|*`int`*<br>*`string`*<br>*`bool`*<br>*`bool`*|

### KSTermZHRadius

|element name|source files|child elements|child types|attributes|attribute types|
|-----|-----|-----|-----|-----|-----|
|<a name="kstermzhradius">`ksterm_zh_radius`</a>|[*KSTermZHRadiusBuilder.cxx*](https://github.com/KATRIN-Experiment/Kassiopeia/tree/main/Kassiopeia/Bindings/Terminators/Source/KSTermZHRadiusBuilder.cxx)|—    |—    |`central_expansion`<br>`electric_field`<br>`magnetic_field`<br>`name`<br>`remote_expansion`|*`bool`*<br>*`string`*<br>*`string`*<br>*`string`*<br>*`bool`*|

### KSTrajControlBChange

|element name|source files|child elements|child types|attributes|attribute types|
|-----|-----|-----|-----|-----|-----|
|<a name="kstrajcontrolbchange">`kstraj_control_B_change`</a>|[*KSTrajControlBChangeBuilder.cxx*](https://github.com/KATRIN-Experiment/Kassiopeia/tree/main/Kassiopeia/Bindings/Trajectories/Source/KSTrajControlBChangeBuilder.cxx)<br>[*KSTrajTrajectoryAdiabaticBuilder.cxx*](https://github.com/KATRIN-Experiment/Kassiopeia/tree/main/Kassiopeia/Bindings/Trajectories/Source/KSTrajTrajectoryAdiabaticBuilder.cxx)<br>[*KSTrajTrajectoryAdiabaticSpinBuilder.cxx*](https://github.com/KATRIN-Experiment/Kassiopeia/tree/main/Kassiopeia/Bindings/Trajectories/Source/KSTrajTrajectoryAdiabaticSpinBuilder.cxx)<br>[*KSTrajTrajectoryElectricBuilder.cxx*](https://github.com/KATRIN-Experiment/Kassiopeia/tree/main/Kassiopeia/Bindings/Trajectories/Source/KSTrajTrajectoryElectricBuilder.cxx)<br>[*KSTrajTrajectoryExactBuilder.cxx*](https://github.com/KATRIN-Experiment/Kassiopeia/tree/main/Kassiopeia/Bindings/Trajectories/Source/KSTrajTrajectoryExactBuilder.cxx)<br>[*KSTrajTrajectoryExactSpinBuilder.cxx*](https://github.com/KATRIN-Experiment/Kassiopeia/tree/main/Kassiopeia/Bindings/Trajectories/Source/KSTrajTrajectoryExactSpinBuilder.cxx)<br>[*KSTrajTrajectoryMagneticBuilder.cxx*](https://github.com/KATRIN-Experiment/Kassiopeia/tree/main/Kassiopeia/Bindings/Trajectories/Source/KSTrajTrajectoryMagneticBuilder.cxx)|—    |—    |`fraction`<br>`name`|*`double`*<br>*`string`*|

### KSTrajControlCyclotron

|element name|source files|child elements|child types|attributes|attribute types|
|-----|-----|-----|-----|-----|-----|
|<a name="kstrajcontrolcyclotron">`kstraj_control_cyclotron`</a>|[*KSTrajControlCyclotronBuilder.cxx*](https://github.com/KATRIN-Experiment/Kassiopeia/tree/main/Kassiopeia/Bindings/Trajectories/Source/KSTrajControlCyclotronBuilder.cxx)<br>[*KSTrajTrajectoryAdiabaticBuilder.cxx*](https://github.com/KATRIN-Experiment/Kassiopeia/tree/main/Kassiopeia/Bindings/Trajectories/Source/KSTrajTrajectoryAdiabaticBuilder.cxx)<br>[*KSTrajTrajectoryAdiabaticSpinBuilder.cxx*](https://github.com/KATRIN-Experiment/Kassiopeia/tree/main/Kassiopeia/Bindings/Trajectories/Source/KSTrajTrajectoryAdiabaticSpinBuilder.cxx)<br>[*KSTrajTrajectoryExactBuilder.cxx*](https://github.com/KATRIN-Experiment/Kassiopeia/tree/main/Kassiopeia/Bindings/Trajectories/Source/KSTrajTrajectoryExactBuilder.cxx)<br>[*KSTrajTrajectoryExactSpinBuilder.cxx*](https://github.com/KATRIN-Experiment/Kassiopeia/tree/main/Kassiopeia/Bindings/Trajectories/Source/KSTrajTrajectoryExactSpinBuilder.cxx)|—    |—    |`fraction`<br>`name`|*`double`*<br>*`string`*|

### KSTrajControlEnergy

|element name|source files|child elements|child types|attributes|attribute types|
|-----|-----|-----|-----|-----|-----|
|<a name="kstrajcontrolenergy">`kstraj_control_energy`</a>|[*KSTrajControlEnergyBuilder.cxx*](https://github.com/KATRIN-Experiment/Kassiopeia/tree/main/Kassiopeia/Bindings/Trajectories/Source/KSTrajControlEnergyBuilder.cxx)<br>[*KSTrajTrajectoryAdiabaticBuilder.cxx*](https://github.com/KATRIN-Experiment/Kassiopeia/tree/main/Kassiopeia/Bindings/Trajectories/Source/KSTrajTrajectoryAdiabaticBuilder.cxx)<br>[*KSTrajTrajectoryAdiabaticSpinBuilder.cxx*](https://github.com/KATRIN-Experiment/Kassiopeia/tree/main/Kassiopeia/Bindings/Trajectories/Source/KSTrajTrajectoryAdiabaticSpinBuilder.cxx)<br>[*KSTrajTrajectoryExactBuilder.cxx*](https://github.com/KATRIN-Experiment/Kassiopeia/tree/main/Kassiopeia/Bindings/Trajectories/Source/KSTrajTrajectoryExactBuilder.cxx)<br>[*KSTrajTrajectoryExactSpinBuilder.cxx*](https://github.com/KATRIN-Experiment/Kassiopeia/tree/main/Kassiopeia/Bindings/Trajectories/Source/KSTrajTrajectoryExactSpinBuilder.cxx)|—    |—    |`adjustment`<br>`adjustment_down`<br>`adjustment_up`<br>`initial_step`<br>`lower_limit`<br>`max_length`<br>`min_length`<br>`name`<br>`step_rescale`<br>`upper_limit`|*`double`*<br>*`double`*<br>*`double`*<br>*`double`*<br>*`double`*<br>*`double`*<br>*`double`*<br>*`string`*<br>*`double`*<br>*`double`*|

### KSTrajControlLength

|element name|source files|child elements|child types|attributes|attribute types|
|-----|-----|-----|-----|-----|-----|
|<a name="kstrajcontrollength">`kstraj_control_length`</a>|[*KSTrajControlLengthBuilder.cxx*](https://github.com/KATRIN-Experiment/Kassiopeia/tree/main/Kassiopeia/Bindings/Trajectories/Source/KSTrajControlLengthBuilder.cxx)<br>[*KSTrajTrajectoryAdiabaticBuilder.cxx*](https://github.com/KATRIN-Experiment/Kassiopeia/tree/main/Kassiopeia/Bindings/Trajectories/Source/KSTrajTrajectoryAdiabaticBuilder.cxx)<br>[*KSTrajTrajectoryAdiabaticSpinBuilder.cxx*](https://github.com/KATRIN-Experiment/Kassiopeia/tree/main/Kassiopeia/Bindings/Trajectories/Source/KSTrajTrajectoryAdiabaticSpinBuilder.cxx)<br>[*KSTrajTrajectoryElectricBuilder.cxx*](https://github.com/KATRIN-Experiment/Kassiopeia/tree/main/Kassiopeia/Bindings/Trajectories/Source/KSTrajTrajectoryElectricBuilder.cxx)<br>[*KSTrajTrajectoryExactBuilder.cxx*](https://github.com/KATRIN-Experiment/Kassiopeia/tree/main/Kassiopeia/Bindings/Trajectories/Source/KSTrajTrajectoryExactBuilder.cxx)<br>[*KSTrajTrajectoryExactSpinBuilder.cxx*](https://github.com/KATRIN-Experiment/Kassiopeia/tree/main/Kassiopeia/Bindings/Trajectories/Source/KSTrajTrajectoryExactSpinBuilder.cxx)<br>[*KSTrajTrajectoryExactTrappedBuilder.cxx*](https://github.com/KATRIN-Experiment/Kassiopeia/tree/main/Kassiopeia/Bindings/Trajectories/Source/KSTrajTrajectoryExactTrappedBuilder.cxx)<br>[*KSTrajTrajectoryMagneticBuilder.cxx*](https://github.com/KATRIN-Experiment/Kassiopeia/tree/main/Kassiopeia/Bindings/Trajectories/Source/KSTrajTrajectoryMagneticBuilder.cxx)|—    |—    |`length`<br>`name`|*`double`*<br>*`string`*|

### KSTrajControlMDot

|element name|source files|child elements|child types|attributes|attribute types|
|-----|-----|-----|-----|-----|-----|
|<a name="kstrajcontrolmdot">`kstraj_control_m_dot`</a>|[*KSTrajControlMDotBuilder.cxx*](https://github.com/KATRIN-Experiment/Kassiopeia/tree/main/Kassiopeia/Bindings/Trajectories/Source/KSTrajControlMDotBuilder.cxx)<br>[*KSTrajTrajectoryAdiabaticSpinBuilder.cxx*](https://github.com/KATRIN-Experiment/Kassiopeia/tree/main/Kassiopeia/Bindings/Trajectories/Source/KSTrajTrajectoryAdiabaticSpinBuilder.cxx)|—    |—    |`fraction`<br>`name`|*`double`*<br>*`string`*|

### KSTrajControlMagneticMoment

|element name|source files|child elements|child types|attributes|attribute types|
|-----|-----|-----|-----|-----|-----|
|<a name="kstrajcontrolmagneticmoment">`kstraj_control_magnetic_moment`</a>|[*KSTrajControlMagneticMomentBuilder.cxx*](https://github.com/KATRIN-Experiment/Kassiopeia/tree/main/Kassiopeia/Bindings/Trajectories/Source/KSTrajControlMagneticMomentBuilder.cxx)<br>[*KSTrajTrajectoryAdiabaticBuilder.cxx*](https://github.com/KATRIN-Experiment/Kassiopeia/tree/main/Kassiopeia/Bindings/Trajectories/Source/KSTrajTrajectoryAdiabaticBuilder.cxx)<br>[*KSTrajTrajectoryAdiabaticSpinBuilder.cxx*](https://github.com/KATRIN-Experiment/Kassiopeia/tree/main/Kassiopeia/Bindings/Trajectories/Source/KSTrajTrajectoryAdiabaticSpinBuilder.cxx)<br>[*KSTrajTrajectoryExactBuilder.cxx*](https://github.com/KATRIN-Experiment/Kassiopeia/tree/main/Kassiopeia/Bindings/Trajectories/Source/KSTrajTrajectoryExactBuilder.cxx)<br>[*KSTrajTrajectoryExactSpinBuilder.cxx*](https://github.com/KATRIN-Experiment/Kassiopeia/tree/main/Kassiopeia/Bindings/Trajectories/Source/KSTrajTrajectoryExactSpinBuilder.cxx)|—    |—    |`lower_limit`<br>`name`<br>`upper_limit`|*`double`*<br>*`string`*<br>*`double`*|

### KSTrajControlMomentumNumericalError

|element name|source files|child elements|child types|attributes|attribute types|
|-----|-----|-----|-----|-----|-----|
|<a name="kstrajcontrolmomentumnumericalerror">`kstraj_control_momentum_numerical_error`</a>|[*KSTrajControlMomentumNumericalErrorBuilder.cxx*](https://github.com/KATRIN-Experiment/Kassiopeia/tree/main/Kassiopeia/Bindings/Trajectories/Source/KSTrajControlMomentumNumericalErrorBuilder.cxx)<br>[*KSTrajTrajectoryAdiabaticBuilder.cxx*](https://github.com/KATRIN-Experiment/Kassiopeia/tree/main/Kassiopeia/Bindings/Trajectories/Source/KSTrajTrajectoryAdiabaticBuilder.cxx)<br>[*KSTrajTrajectoryAdiabaticSpinBuilder.cxx*](https://github.com/KATRIN-Experiment/Kassiopeia/tree/main/Kassiopeia/Bindings/Trajectories/Source/KSTrajTrajectoryAdiabaticSpinBuilder.cxx)<br>[*KSTrajTrajectoryExactBuilder.cxx*](https://github.com/KATRIN-Experiment/Kassiopeia/tree/main/Kassiopeia/Bindings/Trajectories/Source/KSTrajTrajectoryExactBuilder.cxx)<br>[*KSTrajTrajectoryExactSpinBuilder.cxx*](https://github.com/KATRIN-Experiment/Kassiopeia/tree/main/Kassiopeia/Bindings/Trajectories/Source/KSTrajTrajectoryExactSpinBuilder.cxx)|—    |—    |`absolute_momentum_error`<br>`name`<br>`safety_factor`<br>`solver_order`|*`double`*<br>*`string`*<br>*`double`*<br>*`double`*|

### KSTrajControlPositionNumericalError

|element name|source files|child elements|child types|attributes|attribute types|
|-----|-----|-----|-----|-----|-----|
|<a name="kstrajcontrolpositionnumericalerror">`kstraj_control_position_numerical_error`</a>|[*KSTrajControlPositionNumericalErrorBuilder.cxx*](https://github.com/KATRIN-Experiment/Kassiopeia/tree/main/Kassiopeia/Bindings/Trajectories/Source/KSTrajControlPositionNumericalErrorBuilder.cxx)<br>[*KSTrajTrajectoryAdiabaticBuilder.cxx*](https://github.com/KATRIN-Experiment/Kassiopeia/tree/main/Kassiopeia/Bindings/Trajectories/Source/KSTrajTrajectoryAdiabaticBuilder.cxx)<br>[*KSTrajTrajectoryAdiabaticSpinBuilder.cxx*](https://github.com/KATRIN-Experiment/Kassiopeia/tree/main/Kassiopeia/Bindings/Trajectories/Source/KSTrajTrajectoryAdiabaticSpinBuilder.cxx)<br>[*KSTrajTrajectoryExactBuilder.cxx*](https://github.com/KATRIN-Experiment/Kassiopeia/tree/main/Kassiopeia/Bindings/Trajectories/Source/KSTrajTrajectoryExactBuilder.cxx)<br>[*KSTrajTrajectoryExactSpinBuilder.cxx*](https://github.com/KATRIN-Experiment/Kassiopeia/tree/main/Kassiopeia/Bindings/Trajectories/Source/KSTrajTrajectoryExactSpinBuilder.cxx)|—    |—    |`absolute_position_error`<br>`name`<br>`safety_factor`<br>`solver_order`|*`double`*<br>*`string`*<br>*`double`*<br>*`double`*|

### KSTrajControlSpinPrecession

|element name|source files|child elements|child types|attributes|attribute types|
|-----|-----|-----|-----|-----|-----|
|<a name="kstrajcontrolspinprecession">`kstraj_control_spin_precession`</a>|[*KSTrajControlSpinPrecessionBuilder.cxx*](https://github.com/KATRIN-Experiment/Kassiopeia/tree/main/Kassiopeia/Bindings/Trajectories/Source/KSTrajControlSpinPrecessionBuilder.cxx)<br>[*KSTrajTrajectoryAdiabaticSpinBuilder.cxx*](https://github.com/KATRIN-Experiment/Kassiopeia/tree/main/Kassiopeia/Bindings/Trajectories/Source/KSTrajTrajectoryAdiabaticSpinBuilder.cxx)<br>[*KSTrajTrajectoryExactSpinBuilder.cxx*](https://github.com/KATRIN-Experiment/Kassiopeia/tree/main/Kassiopeia/Bindings/Trajectories/Source/KSTrajTrajectoryExactSpinBuilder.cxx)|—    |—    |`fraction`<br>`name`|*`double`*<br>*`string`*|

### KSTrajControlTime

|element name|source files|child elements|child types|attributes|attribute types|
|-----|-----|-----|-----|-----|-----|
|<a name="kstrajcontroltime">`kstraj_control_time`</a>|[*KSTrajControlTimeBuilder.cxx*](https://github.com/KATRIN-Experiment/Kassiopeia/tree/main/Kassiopeia/Bindings/Trajectories/Source/KSTrajControlTimeBuilder.cxx)<br>[*KSTrajTrajectoryAdiabaticBuilder.cxx*](https://github.com/KATRIN-Experiment/Kassiopeia/tree/main/Kassiopeia/Bindings/Trajectories/Source/KSTrajTrajectoryAdiabaticBuilder.cxx)<br>[*KSTrajTrajectoryAdiabaticSpinBuilder.cxx*](https://github.com/KATRIN-Experiment/Kassiopeia/tree/main/Kassiopeia/Bindings/Trajectories/Source/KSTrajTrajectoryAdiabaticSpinBuilder.cxx)<br>[*KSTrajTrajectoryElectricBuilder.cxx*](https://github.com/KATRIN-Experiment/Kassiopeia/tree/main/Kassiopeia/Bindings/Trajectories/Source/KSTrajTrajectoryElectricBuilder.cxx)<br>[*KSTrajTrajectoryExactBuilder.cxx*](https://github.com/KATRIN-Experiment/Kassiopeia/tree/main/Kassiopeia/Bindings/Trajectories/Source/KSTrajTrajectoryExactBuilder.cxx)<br>[*KSTrajTrajectoryExactSpinBuilder.cxx*](https://github.com/KATRIN-Experiment/Kassiopeia/tree/main/Kassiopeia/Bindings/Trajectories/Source/KSTrajTrajectoryExactSpinBuilder.cxx)<br>[*KSTrajTrajectoryExactTrappedBuilder.cxx*](https://github.com/KATRIN-Experiment/Kassiopeia/tree/main/Kassiopeia/Bindings/Trajectories/Source/KSTrajTrajectoryExactTrappedBuilder.cxx)<br>[*KSTrajTrajectoryMagneticBuilder.cxx*](https://github.com/KATRIN-Experiment/Kassiopeia/tree/main/Kassiopeia/Bindings/Trajectories/Source/KSTrajTrajectoryMagneticBuilder.cxx)|—    |—    |`name`<br>`time`|*`string`*<br>*`double`*|

### KSTrajIntegratorRK54

|element name|source files|child elements|child types|attributes|attribute types|
|-----|-----|-----|-----|-----|-----|
|<a name="kstrajintegratorrk54">`kstraj_integrator_rk54`</a>|[*KSTrajIntegratorRK54Builder.cxx*](https://github.com/KATRIN-Experiment/Kassiopeia/tree/main/Kassiopeia/Bindings/Trajectories/Source/KSTrajIntegratorRK54Builder.cxx)<br>[*KSTrajTrajectoryAdiabaticBuilder.cxx*](https://github.com/KATRIN-Experiment/Kassiopeia/tree/main/Kassiopeia/Bindings/Trajectories/Source/KSTrajTrajectoryAdiabaticBuilder.cxx)<br>[*KSTrajTrajectoryAdiabaticSpinBuilder.cxx*](https://github.com/KATRIN-Experiment/Kassiopeia/tree/main/Kassiopeia/Bindings/Trajectories/Source/KSTrajTrajectoryAdiabaticSpinBuilder.cxx)<br>[*KSTrajTrajectoryElectricBuilder.cxx*](https://github.com/KATRIN-Experiment/Kassiopeia/tree/main/Kassiopeia/Bindings/Trajectories/Source/KSTrajTrajectoryElectricBuilder.cxx)<br>[*KSTrajTrajectoryExactBuilder.cxx*](https://github.com/KATRIN-Experiment/Kassiopeia/tree/main/Kassiopeia/Bindings/Trajectories/Source/KSTrajTrajectoryExactBuilder.cxx)<br>[*KSTrajTrajectoryExactSpinBuilder.cxx*](https://github.com/KATRIN-Experiment/Kassiopeia/tree/main/Kassiopeia/Bindings/Trajectories/Source/KSTrajTrajectoryExactSpinBuilder.cxx)<br>[*KSTrajTrajectoryMagneticBuilder.cxx*](https://github.com/KATRIN-Experiment/Kassiopeia/tree/main/Kassiopeia/Bindings/Trajectories/Source/KSTrajTrajectoryMagneticBuilder.cxx)|—    |—    |`name`|*`string`*|

### KSTrajIntegratorRK65

|element name|source files|child elements|child types|attributes|attribute types|
|-----|-----|-----|-----|-----|-----|
|<a name="kstrajintegratorrk65">`kstraj_integrator_rk65`</a>|[*KSTrajIntegratorRK65Builder.cxx*](https://github.com/KATRIN-Experiment/Kassiopeia/tree/main/Kassiopeia/Bindings/Trajectories/Source/KSTrajIntegratorRK65Builder.cxx)<br>[*KSTrajTrajectoryAdiabaticBuilder.cxx*](https://github.com/KATRIN-Experiment/Kassiopeia/tree/main/Kassiopeia/Bindings/Trajectories/Source/KSTrajTrajectoryAdiabaticBuilder.cxx)<br>[*KSTrajTrajectoryAdiabaticSpinBuilder.cxx*](https://github.com/KATRIN-Experiment/Kassiopeia/tree/main/Kassiopeia/Bindings/Trajectories/Source/KSTrajTrajectoryAdiabaticSpinBuilder.cxx)<br>[*KSTrajTrajectoryElectricBuilder.cxx*](https://github.com/KATRIN-Experiment/Kassiopeia/tree/main/Kassiopeia/Bindings/Trajectories/Source/KSTrajTrajectoryElectricBuilder.cxx)<br>[*KSTrajTrajectoryExactBuilder.cxx*](https://github.com/KATRIN-Experiment/Kassiopeia/tree/main/Kassiopeia/Bindings/Trajectories/Source/KSTrajTrajectoryExactBuilder.cxx)<br>[*KSTrajTrajectoryExactSpinBuilder.cxx*](https://github.com/KATRIN-Experiment/Kassiopeia/tree/main/Kassiopeia/Bindings/Trajectories/Source/KSTrajTrajectoryExactSpinBuilder.cxx)<br>[*KSTrajTrajectoryMagneticBuilder.cxx*](https://github.com/KATRIN-Experiment/Kassiopeia/tree/main/Kassiopeia/Bindings/Trajectories/Source/KSTrajTrajectoryMagneticBuilder.cxx)|—    |—    |`name`|*`string`*|

### KSTrajIntegratorRK8

|element name|source files|child elements|child types|attributes|attribute types|
|-----|-----|-----|-----|-----|-----|
|<a name="kstrajintegratorrk8">`kstraj_integrator_rk8`</a>|[*KSTrajIntegratorRK8Builder.cxx*](https://github.com/KATRIN-Experiment/Kassiopeia/tree/main/Kassiopeia/Bindings/Trajectories/Source/KSTrajIntegratorRK8Builder.cxx)<br>[*KSTrajTrajectoryAdiabaticBuilder.cxx*](https://github.com/KATRIN-Experiment/Kassiopeia/tree/main/Kassiopeia/Bindings/Trajectories/Source/KSTrajTrajectoryAdiabaticBuilder.cxx)<br>[*KSTrajTrajectoryAdiabaticSpinBuilder.cxx*](https://github.com/KATRIN-Experiment/Kassiopeia/tree/main/Kassiopeia/Bindings/Trajectories/Source/KSTrajTrajectoryAdiabaticSpinBuilder.cxx)<br>[*KSTrajTrajectoryElectricBuilder.cxx*](https://github.com/KATRIN-Experiment/Kassiopeia/tree/main/Kassiopeia/Bindings/Trajectories/Source/KSTrajTrajectoryElectricBuilder.cxx)<br>[*KSTrajTrajectoryExactBuilder.cxx*](https://github.com/KATRIN-Experiment/Kassiopeia/tree/main/Kassiopeia/Bindings/Trajectories/Source/KSTrajTrajectoryExactBuilder.cxx)<br>[*KSTrajTrajectoryExactSpinBuilder.cxx*](https://github.com/KATRIN-Experiment/Kassiopeia/tree/main/Kassiopeia/Bindings/Trajectories/Source/KSTrajTrajectoryExactSpinBuilder.cxx)<br>[*KSTrajTrajectoryExactTrappedBuilder.cxx*](https://github.com/KATRIN-Experiment/Kassiopeia/tree/main/Kassiopeia/Bindings/Trajectories/Source/KSTrajTrajectoryExactTrappedBuilder.cxx)<br>[*KSTrajTrajectoryMagneticBuilder.cxx*](https://github.com/KATRIN-Experiment/Kassiopeia/tree/main/Kassiopeia/Bindings/Trajectories/Source/KSTrajTrajectoryMagneticBuilder.cxx)|—    |—    |`name`|*`string`*|

### KSTrajIntegratorRK86

|element name|source files|child elements|child types|attributes|attribute types|
|-----|-----|-----|-----|-----|-----|
|<a name="kstrajintegratorrk86">`kstraj_integrator_rk86`</a>|[*KSTrajIntegratorRK86Builder.cxx*](https://github.com/KATRIN-Experiment/Kassiopeia/tree/main/Kassiopeia/Bindings/Trajectories/Source/KSTrajIntegratorRK86Builder.cxx)<br>[*KSTrajTrajectoryAdiabaticBuilder.cxx*](https://github.com/KATRIN-Experiment/Kassiopeia/tree/main/Kassiopeia/Bindings/Trajectories/Source/KSTrajTrajectoryAdiabaticBuilder.cxx)<br>[*KSTrajTrajectoryAdiabaticSpinBuilder.cxx*](https://github.com/KATRIN-Experiment/Kassiopeia/tree/main/Kassiopeia/Bindings/Trajectories/Source/KSTrajTrajectoryAdiabaticSpinBuilder.cxx)<br>[*KSTrajTrajectoryElectricBuilder.cxx*](https://github.com/KATRIN-Experiment/Kassiopeia/tree/main/Kassiopeia/Bindings/Trajectories/Source/KSTrajTrajectoryElectricBuilder.cxx)<br>[*KSTrajTrajectoryExactBuilder.cxx*](https://github.com/KATRIN-Experiment/Kassiopeia/tree/main/Kassiopeia/Bindings/Trajectories/Source/KSTrajTrajectoryExactBuilder.cxx)<br>[*KSTrajTrajectoryExactSpinBuilder.cxx*](https://github.com/KATRIN-Experiment/Kassiopeia/tree/main/Kassiopeia/Bindings/Trajectories/Source/KSTrajTrajectoryExactSpinBuilder.cxx)<br>[*KSTrajTrajectoryMagneticBuilder.cxx*](https://github.com/KATRIN-Experiment/Kassiopeia/tree/main/Kassiopeia/Bindings/Trajectories/Source/KSTrajTrajectoryMagneticBuilder.cxx)|—    |—    |`name`|*`string`*|

### KSTrajIntegratorRK87

|element name|source files|child elements|child types|attributes|attribute types|
|-----|-----|-----|-----|-----|-----|
|<a name="kstrajintegratorrk87">`kstraj_integrator_rk87`</a>|[*KSTrajIntegratorRK87Builder.cxx*](https://github.com/KATRIN-Experiment/Kassiopeia/tree/main/Kassiopeia/Bindings/Trajectories/Source/KSTrajIntegratorRK87Builder.cxx)<br>[*KSTrajTrajectoryAdiabaticBuilder.cxx*](https://github.com/KATRIN-Experiment/Kassiopeia/tree/main/Kassiopeia/Bindings/Trajectories/Source/KSTrajTrajectoryAdiabaticBuilder.cxx)<br>[*KSTrajTrajectoryAdiabaticSpinBuilder.cxx*](https://github.com/KATRIN-Experiment/Kassiopeia/tree/main/Kassiopeia/Bindings/Trajectories/Source/KSTrajTrajectoryAdiabaticSpinBuilder.cxx)<br>[*KSTrajTrajectoryElectricBuilder.cxx*](https://github.com/KATRIN-Experiment/Kassiopeia/tree/main/Kassiopeia/Bindings/Trajectories/Source/KSTrajTrajectoryElectricBuilder.cxx)<br>[*KSTrajTrajectoryExactBuilder.cxx*](https://github.com/KATRIN-Experiment/Kassiopeia/tree/main/Kassiopeia/Bindings/Trajectories/Source/KSTrajTrajectoryExactBuilder.cxx)<br>[*KSTrajTrajectoryExactSpinBuilder.cxx*](https://github.com/KATRIN-Experiment/Kassiopeia/tree/main/Kassiopeia/Bindings/Trajectories/Source/KSTrajTrajectoryExactSpinBuilder.cxx)<br>[*KSTrajTrajectoryMagneticBuilder.cxx*](https://github.com/KATRIN-Experiment/Kassiopeia/tree/main/Kassiopeia/Bindings/Trajectories/Source/KSTrajTrajectoryMagneticBuilder.cxx)|—    |—    |`name`|*`string`*|

### KSTrajIntegratorRKDP54

|element name|source files|child elements|child types|attributes|attribute types|
|-----|-----|-----|-----|-----|-----|
|<a name="kstrajintegratorrkdp54">`kstraj_integrator_rkdp54`</a>|[*KSTrajIntegratorRKDP54Builder.cxx*](https://github.com/KATRIN-Experiment/Kassiopeia/tree/main/Kassiopeia/Bindings/Trajectories/Source/KSTrajIntegratorRKDP54Builder.cxx)<br>[*KSTrajTrajectoryAdiabaticBuilder.cxx*](https://github.com/KATRIN-Experiment/Kassiopeia/tree/main/Kassiopeia/Bindings/Trajectories/Source/KSTrajTrajectoryAdiabaticBuilder.cxx)<br>[*KSTrajTrajectoryAdiabaticSpinBuilder.cxx*](https://github.com/KATRIN-Experiment/Kassiopeia/tree/main/Kassiopeia/Bindings/Trajectories/Source/KSTrajTrajectoryAdiabaticSpinBuilder.cxx)<br>[*KSTrajTrajectoryElectricBuilder.cxx*](https://github.com/KATRIN-Experiment/Kassiopeia/tree/main/Kassiopeia/Bindings/Trajectories/Source/KSTrajTrajectoryElectricBuilder.cxx)<br>[*KSTrajTrajectoryExactBuilder.cxx*](https://github.com/KATRIN-Experiment/Kassiopeia/tree/main/Kassiopeia/Bindings/Trajectories/Source/KSTrajTrajectoryExactBuilder.cxx)<br>[*KSTrajTrajectoryExactSpinBuilder.cxx*](https://github.com/KATRIN-Experiment/Kassiopeia/tree/main/Kassiopeia/Bindings/Trajectories/Source/KSTrajTrajectoryExactSpinBuilder.cxx)<br>[*KSTrajTrajectoryMagneticBuilder.cxx*](https://github.com/KATRIN-Experiment/Kassiopeia/tree/main/Kassiopeia/Bindings/Trajectories/Source/KSTrajTrajectoryMagneticBuilder.cxx)|—    |—    |`name`|*`string`*|

### KSTrajIntegratorRKDP853

|element name|source files|child elements|child types|attributes|attribute types|
|-----|-----|-----|-----|-----|-----|
|<a name="kstrajintegratorrkdp853">`kstraj_integrator_rkdp853`</a>|[*KSTrajIntegratorRKDP853Builder.cxx*](https://github.com/KATRIN-Experiment/Kassiopeia/tree/main/Kassiopeia/Bindings/Trajectories/Source/KSTrajIntegratorRKDP853Builder.cxx)<br>[*KSTrajTrajectoryAdiabaticBuilder.cxx*](https://github.com/KATRIN-Experiment/Kassiopeia/tree/main/Kassiopeia/Bindings/Trajectories/Source/KSTrajTrajectoryAdiabaticBuilder.cxx)<br>[*KSTrajTrajectoryAdiabaticSpinBuilder.cxx*](https://github.com/KATRIN-Experiment/Kassiopeia/tree/main/Kassiopeia/Bindings/Trajectories/Source/KSTrajTrajectoryAdiabaticSpinBuilder.cxx)<br>[*KSTrajTrajectoryElectricBuilder.cxx*](https://github.com/KATRIN-Experiment/Kassiopeia/tree/main/Kassiopeia/Bindings/Trajectories/Source/KSTrajTrajectoryElectricBuilder.cxx)<br>[*KSTrajTrajectoryExactBuilder.cxx*](https://github.com/KATRIN-Experiment/Kassiopeia/tree/main/Kassiopeia/Bindings/Trajectories/Source/KSTrajTrajectoryExactBuilder.cxx)<br>[*KSTrajTrajectoryExactSpinBuilder.cxx*](https://github.com/KATRIN-Experiment/Kassiopeia/tree/main/Kassiopeia/Bindings/Trajectories/Source/KSTrajTrajectoryExactSpinBuilder.cxx)<br>[*KSTrajTrajectoryMagneticBuilder.cxx*](https://github.com/KATRIN-Experiment/Kassiopeia/tree/main/Kassiopeia/Bindings/Trajectories/Source/KSTrajTrajectoryMagneticBuilder.cxx)|—    |—    |`name`|*`string`*|

### KSTrajIntegratorSym4

|element name|source files|child elements|child types|attributes|attribute types|
|-----|-----|-----|-----|-----|-----|
|<a name="kstrajintegratorsym4">`kstraj_integrator_sym4`</a>|[*KSTrajIntegratorSym4Builder.cxx*](https://github.com/KATRIN-Experiment/Kassiopeia/tree/main/Kassiopeia/Bindings/Trajectories/Source/KSTrajIntegratorSym4Builder.cxx)<br>[*KSTrajTrajectoryExactTrappedBuilder.cxx*](https://github.com/KATRIN-Experiment/Kassiopeia/tree/main/Kassiopeia/Bindings/Trajectories/Source/KSTrajTrajectoryExactTrappedBuilder.cxx)|—    |—    |`name`|*`string`*|

### KSTrajInterpolatorContinuousRungeKutta

|element name|source files|child elements|child types|attributes|attribute types|
|-----|-----|-----|-----|-----|-----|
|<a name="kstrajinterpolatorcontinuousrungekutta">`kstraj_interpolator_crk`</a>|[*KSTrajInterpolatorContinuousRungeKuttaBuilder.cxx*](https://github.com/KATRIN-Experiment/Kassiopeia/tree/main/Kassiopeia/Bindings/Trajectories/Source/KSTrajInterpolatorContinuousRungeKuttaBuilder.cxx)<br>[*KSTrajTrajectoryAdiabaticBuilder.cxx*](https://github.com/KATRIN-Experiment/Kassiopeia/tree/main/Kassiopeia/Bindings/Trajectories/Source/KSTrajTrajectoryAdiabaticBuilder.cxx)<br>[*KSTrajTrajectoryAdiabaticSpinBuilder.cxx*](https://github.com/KATRIN-Experiment/Kassiopeia/tree/main/Kassiopeia/Bindings/Trajectories/Source/KSTrajTrajectoryAdiabaticSpinBuilder.cxx)<br>[*KSTrajTrajectoryElectricBuilder.cxx*](https://github.com/KATRIN-Experiment/Kassiopeia/tree/main/Kassiopeia/Bindings/Trajectories/Source/KSTrajTrajectoryElectricBuilder.cxx)<br>[*KSTrajTrajectoryExactBuilder.cxx*](https://github.com/KATRIN-Experiment/Kassiopeia/tree/main/Kassiopeia/Bindings/Trajectories/Source/KSTrajTrajectoryExactBuilder.cxx)<br>[*KSTrajTrajectoryExactSpinBuilder.cxx*](https://github.com/KATRIN-Experiment/Kassiopeia/tree/main/Kassiopeia/Bindings/Trajectories/Source/KSTrajTrajectoryExactSpinBuilder.cxx)<br>[*KSTrajTrajectoryMagneticBuilder.cxx*](https://github.com/KATRIN-Experiment/Kassiopeia/tree/main/Kassiopeia/Bindings/Trajectories/Source/KSTrajTrajectoryMagneticBuilder.cxx)|—    |—    |`name`|*`string`*|

### KSTrajInterpolatorFast

|element name|source files|child elements|child types|attributes|attribute types|
|-----|-----|-----|-----|-----|-----|
|<a name="kstrajinterpolatorfast">`kstraj_interpolator_fast`</a>|[*KSTrajInterpolatorFastBuilder.cxx*](https://github.com/KATRIN-Experiment/Kassiopeia/tree/main/Kassiopeia/Bindings/Trajectories/Source/KSTrajInterpolatorFastBuilder.cxx)<br>[*KSTrajTrajectoryAdiabaticBuilder.cxx*](https://github.com/KATRIN-Experiment/Kassiopeia/tree/main/Kassiopeia/Bindings/Trajectories/Source/KSTrajTrajectoryAdiabaticBuilder.cxx)<br>[*KSTrajTrajectoryAdiabaticSpinBuilder.cxx*](https://github.com/KATRIN-Experiment/Kassiopeia/tree/main/Kassiopeia/Bindings/Trajectories/Source/KSTrajTrajectoryAdiabaticSpinBuilder.cxx)<br>[*KSTrajTrajectoryElectricBuilder.cxx*](https://github.com/KATRIN-Experiment/Kassiopeia/tree/main/Kassiopeia/Bindings/Trajectories/Source/KSTrajTrajectoryElectricBuilder.cxx)<br>[*KSTrajTrajectoryExactBuilder.cxx*](https://github.com/KATRIN-Experiment/Kassiopeia/tree/main/Kassiopeia/Bindings/Trajectories/Source/KSTrajTrajectoryExactBuilder.cxx)<br>[*KSTrajTrajectoryExactSpinBuilder.cxx*](https://github.com/KATRIN-Experiment/Kassiopeia/tree/main/Kassiopeia/Bindings/Trajectories/Source/KSTrajTrajectoryExactSpinBuilder.cxx)<br>[*KSTrajTrajectoryMagneticBuilder.cxx*](https://github.com/KATRIN-Experiment/Kassiopeia/tree/main/Kassiopeia/Bindings/Trajectories/Source/KSTrajTrajectoryMagneticBuilder.cxx)|—    |—    |`name`|*`string`*|

### KSTrajInterpolatorHermite

|element name|source files|child elements|child types|attributes|attribute types|
|-----|-----|-----|-----|-----|-----|
|<a name="kstrajinterpolatorhermite">`kstraj_interpolator_hermite`</a>|[*KSTrajInterpolatorHermiteBuilder.cxx*](https://github.com/KATRIN-Experiment/Kassiopeia/tree/main/Kassiopeia/Bindings/Trajectories/Source/KSTrajInterpolatorHermiteBuilder.cxx)<br>[*KSTrajTrajectoryAdiabaticBuilder.cxx*](https://github.com/KATRIN-Experiment/Kassiopeia/tree/main/Kassiopeia/Bindings/Trajectories/Source/KSTrajTrajectoryAdiabaticBuilder.cxx)<br>[*KSTrajTrajectoryAdiabaticSpinBuilder.cxx*](https://github.com/KATRIN-Experiment/Kassiopeia/tree/main/Kassiopeia/Bindings/Trajectories/Source/KSTrajTrajectoryAdiabaticSpinBuilder.cxx)<br>[*KSTrajTrajectoryElectricBuilder.cxx*](https://github.com/KATRIN-Experiment/Kassiopeia/tree/main/Kassiopeia/Bindings/Trajectories/Source/KSTrajTrajectoryElectricBuilder.cxx)<br>[*KSTrajTrajectoryExactBuilder.cxx*](https://github.com/KATRIN-Experiment/Kassiopeia/tree/main/Kassiopeia/Bindings/Trajectories/Source/KSTrajTrajectoryExactBuilder.cxx)<br>[*KSTrajTrajectoryExactSpinBuilder.cxx*](https://github.com/KATRIN-Experiment/Kassiopeia/tree/main/Kassiopeia/Bindings/Trajectories/Source/KSTrajTrajectoryExactSpinBuilder.cxx)<br>[*KSTrajTrajectoryMagneticBuilder.cxx*](https://github.com/KATRIN-Experiment/Kassiopeia/tree/main/Kassiopeia/Bindings/Trajectories/Source/KSTrajTrajectoryMagneticBuilder.cxx)|—    |—    |`name`|*`string`*|

### KSTrajTermConstantForcePropagation

|element name|source files|child elements|child types|attributes|attribute types|
|-----|-----|-----|-----|-----|-----|
|<a name="kstrajtermconstantforcepropagation">`kstraj_term_constant_force_propagation`</a>|[*KSTrajTermConstantForcePropagationBuilder.cxx*](https://github.com/KATRIN-Experiment/Kassiopeia/tree/main/Kassiopeia/Bindings/Trajectories/Source/KSTrajTermConstantForcePropagationBuilder.cxx)|—    |—    |`force`<br>`name`|*`KThreeVector`*<br>*`string`*|

### KSTrajTermDrift

|element name|source files|child elements|child types|attributes|attribute types|
|-----|-----|-----|-----|-----|-----|
|<a name="kstrajtermdrift">`kstraj_term_drift`</a>|[*KSTrajTermDriftBuilder.cxx*](https://github.com/KATRIN-Experiment/Kassiopeia/tree/main/Kassiopeia/Bindings/Trajectories/Source/KSTrajTermDriftBuilder.cxx)<br>[*KSTrajTrajectoryAdiabaticBuilder.cxx*](https://github.com/KATRIN-Experiment/Kassiopeia/tree/main/Kassiopeia/Bindings/Trajectories/Source/KSTrajTrajectoryAdiabaticBuilder.cxx)|—    |—    |`name`|*`string`*|

### KSTrajTermGravity

|element name|source files|child elements|child types|attributes|attribute types|
|-----|-----|-----|-----|-----|-----|
|<a name="kstrajtermgravity">`kstraj_term_gravity`</a>|[*KSTrajTermGravityBuilder.cxx*](https://github.com/KATRIN-Experiment/Kassiopeia/tree/main/Kassiopeia/Bindings/Trajectories/Source/KSTrajTermGravityBuilder.cxx)<br>[*KSTrajTrajectoryAdiabaticSpinBuilder.cxx*](https://github.com/KATRIN-Experiment/Kassiopeia/tree/main/Kassiopeia/Bindings/Trajectories/Source/KSTrajTrajectoryAdiabaticSpinBuilder.cxx)<br>[*KSTrajTrajectoryExactBuilder.cxx*](https://github.com/KATRIN-Experiment/Kassiopeia/tree/main/Kassiopeia/Bindings/Trajectories/Source/KSTrajTrajectoryExactBuilder.cxx)<br>[*KSTrajTrajectoryExactSpinBuilder.cxx*](https://github.com/KATRIN-Experiment/Kassiopeia/tree/main/Kassiopeia/Bindings/Trajectories/Source/KSTrajTrajectoryExactSpinBuilder.cxx)|—    |—    |`gravity`<br>`name`|*`KThreeVector`*<br>*`string`*|

### KSTrajTermGyration

|element name|source files|child elements|child types|attributes|attribute types|
|-----|-----|-----|-----|-----|-----|
|<a name="kstrajtermgyration">`kstraj_term_gyration`</a>|[*KSTrajTermGyrationBuilder.cxx*](https://github.com/KATRIN-Experiment/Kassiopeia/tree/main/Kassiopeia/Bindings/Trajectories/Source/KSTrajTermGyrationBuilder.cxx)<br>[*KSTrajTrajectoryAdiabaticBuilder.cxx*](https://github.com/KATRIN-Experiment/Kassiopeia/tree/main/Kassiopeia/Bindings/Trajectories/Source/KSTrajTrajectoryAdiabaticBuilder.cxx)|—    |—    |`name`|*`string`*|

### KSTrajTermPropagation

|element name|source files|child elements|child types|attributes|attribute types|
|-----|-----|-----|-----|-----|-----|
|<a name="kstrajtermpropagation">`kstraj_term_propagation`</a>|[*KSTrajTermPropagationBuilder.cxx*](https://github.com/KATRIN-Experiment/Kassiopeia/tree/main/Kassiopeia/Bindings/Trajectories/Source/KSTrajTermPropagationBuilder.cxx)<br>[*KSTrajTrajectoryAdiabaticBuilder.cxx*](https://github.com/KATRIN-Experiment/Kassiopeia/tree/main/Kassiopeia/Bindings/Trajectories/Source/KSTrajTrajectoryAdiabaticBuilder.cxx)<br>[*KSTrajTrajectoryAdiabaticSpinBuilder.cxx*](https://github.com/KATRIN-Experiment/Kassiopeia/tree/main/Kassiopeia/Bindings/Trajectories/Source/KSTrajTrajectoryAdiabaticSpinBuilder.cxx)<br>[*KSTrajTrajectoryElectricBuilder.cxx*](https://github.com/KATRIN-Experiment/Kassiopeia/tree/main/Kassiopeia/Bindings/Trajectories/Source/KSTrajTrajectoryElectricBuilder.cxx)<br>[*KSTrajTrajectoryExactBuilder.cxx*](https://github.com/KATRIN-Experiment/Kassiopeia/tree/main/Kassiopeia/Bindings/Trajectories/Source/KSTrajTrajectoryExactBuilder.cxx)<br>[*KSTrajTrajectoryExactSpinBuilder.cxx*](https://github.com/KATRIN-Experiment/Kassiopeia/tree/main/Kassiopeia/Bindings/Trajectories/Source/KSTrajTrajectoryExactSpinBuilder.cxx)<br>[*KSTrajTrajectoryExactTrappedBuilder.cxx*](https://github.com/KATRIN-Experiment/Kassiopeia/tree/main/Kassiopeia/Bindings/Trajectories/Source/KSTrajTrajectoryExactTrappedBuilder.cxx)<br>[*KSTrajTrajectoryMagneticBuilder.cxx*](https://github.com/KATRIN-Experiment/Kassiopeia/tree/main/Kassiopeia/Bindings/Trajectories/Source/KSTrajTrajectoryMagneticBuilder.cxx)|—    |—    |`direction`<br>`name`|*`string`*<br>*`string`*|

### KSTrajTermSynchrotron

|element name|source files|child elements|child types|attributes|attribute types|
|-----|-----|-----|-----|-----|-----|
|<a name="kstrajtermsynchrotron">`kstraj_term_synchrotron`</a>|[*KSTrajTermSynchrotronBuilder.cxx*](https://github.com/KATRIN-Experiment/Kassiopeia/tree/main/Kassiopeia/Bindings/Trajectories/Source/KSTrajTermSynchrotronBuilder.cxx)<br>[*KSTrajTrajectoryAdiabaticBuilder.cxx*](https://github.com/KATRIN-Experiment/Kassiopeia/tree/main/Kassiopeia/Bindings/Trajectories/Source/KSTrajTrajectoryAdiabaticBuilder.cxx)<br>[*KSTrajTrajectoryAdiabaticSpinBuilder.cxx*](https://github.com/KATRIN-Experiment/Kassiopeia/tree/main/Kassiopeia/Bindings/Trajectories/Source/KSTrajTrajectoryAdiabaticSpinBuilder.cxx)<br>[*KSTrajTrajectoryExactBuilder.cxx*](https://github.com/KATRIN-Experiment/Kassiopeia/tree/main/Kassiopeia/Bindings/Trajectories/Source/KSTrajTrajectoryExactBuilder.cxx)<br>[*KSTrajTrajectoryExactSpinBuilder.cxx*](https://github.com/KATRIN-Experiment/Kassiopeia/tree/main/Kassiopeia/Bindings/Trajectories/Source/KSTrajTrajectoryExactSpinBuilder.cxx)<br>[*KSTrajTrajectoryExactTrappedBuilder.cxx*](https://github.com/KATRIN-Experiment/Kassiopeia/tree/main/Kassiopeia/Bindings/Trajectories/Source/KSTrajTrajectoryExactTrappedBuilder.cxx)|—    |—    |`enhancement`<br>`name`<br>`old_methode`|*`double`*<br>*`string`*<br>*`bool`*|

### KSTrajTrajectoryAdiabatic

|element name|source files|child elements|child types|attributes|attribute types|
|-----|-----|-----|-----|-----|-----|
|<a name="kstrajtrajectoryadiabatic">`kstraj_trajectory_adiabatic`</a>|[*KSTrajTrajectoryAdiabaticBuilder.cxx*](https://github.com/KATRIN-Experiment/Kassiopeia/tree/main/Kassiopeia/Bindings/Trajectories/Source/KSTrajTrajectoryAdiabaticBuilder.cxx)|[`control_B_change`](#kstrajcontrolbchange)<br>[`control_cyclotron`](#kstrajcontrolcyclotron)<br>[`control_energy`](#kstrajcontrolenergy)<br>[`control_length`](#kstrajcontrollength)<br>[`control_magnetic_moment`](#kstrajcontrolmagneticmoment)<br>[`control_momentum_error`](#kstrajcontrolmomentumnumericalerror)<br>[`control_position_error`](#kstrajcontrolpositionnumericalerror)<br>[`control_time`](#kstrajcontroltime)<br>[`integrator_rk54`](#kstrajintegratorrk54)<br>[`integrator_rk65`](#kstrajintegratorrk65)<br>[`integrator_rk8`](#kstrajintegratorrk8)<br>[`integrator_rk86`](#kstrajintegratorrk86)<br>[`integrator_rk87`](#kstrajintegratorrk87)<br>[`integrator_rkdp54`](#kstrajintegratorrkdp54)<br>[`integrator_rkdp853`](#kstrajintegratorrkdp853)<br>[`interpolator_crk`](#kstrajinterpolatorcontinuousrungekutta)<br>[`interpolator_fast`](#kstrajinterpolatorfast)<br>[`interpolator_hermite`](#kstrajinterpolatorhermite)<br>[`term_drift`](#kstrajtermdrift)<br>[`term_gyration`](#kstrajtermgyration)<br>[`term_propagation`](#kstrajtermpropagation)<br>[`term_synchrotron`](#kstrajtermsynchrotron)|*`KSTrajControlBChange`*<br>*`KSTrajControlCyclotron`*<br>*`KSTrajControlEnergy`*<br>*`KSTrajControlLength`*<br>*`KSTrajControlMagneticMoment`*<br>*`KSTrajControlMomentumNumericalError`*<br>*`KSTrajControlPositionNumericalError`*<br>*`KSTrajControlTime`*<br>*`KSTrajIntegratorRK54`*<br>*`KSTrajIntegratorRK65`*<br>*`KSTrajIntegratorRK8`*<br>*`KSTrajIntegratorRK86`*<br>*`KSTrajIntegratorRK87`*<br>*`KSTrajIntegratorRKDP54`*<br>*`KSTrajIntegratorRKDP853`*<br>*`KSTrajInterpolatorContinuousRungeKutta`*<br>*`KSTrajInterpolatorFast`*<br>*`KSTrajInterpolatorHermite`*<br>*`KSTrajTermDrift`*<br>*`KSTrajTermGyration`*<br>*`KSTrajTermPropagation`*<br>*`KSTrajTermSynchrotron`*|`attempt_limit`<br>`cyclotron_fraction`<br>`max_segments`<br>`name`<br>`piecewise_tolerance`<br>`use_true_position`|*`unsigned int`*<br>*`double`*<br>*`unsigned int`*<br>*`string`*<br>*`double`*<br>*`bool`*|

### KSTrajTrajectoryAdiabaticSpin

|element name|source files|child elements|child types|attributes|attribute types|
|-----|-----|-----|-----|-----|-----|
|<a name="kstrajtrajectoryadiabaticspin">`kstraj_trajectory_adiabatic_spin`</a>|[*KSTrajTrajectoryAdiabaticSpinBuilder.cxx*](https://github.com/KATRIN-Experiment/Kassiopeia/tree/main/Kassiopeia/Bindings/Trajectories/Source/KSTrajTrajectoryAdiabaticSpinBuilder.cxx)|[`control_B_change`](#kstrajcontrolbchange)<br>[`control_cyclotron`](#kstrajcontrolcyclotron)<br>[`control_energy`](#kstrajcontrolenergy)<br>[`control_length`](#kstrajcontrollength)<br>[`control_m_dot`](#kstrajcontrolmdot)<br>[`control_magnetic_moment`](#kstrajcontrolmagneticmoment)<br>[`control_momentum_error`](#kstrajcontrolmomentumnumericalerror)<br>[`control_position_error`](#kstrajcontrolpositionnumericalerror)<br>[`control_spin_precession`](#kstrajcontrolspinprecession)<br>[`control_time`](#kstrajcontroltime)<br>[`integrator_rk54`](#kstrajintegratorrk54)<br>[`integrator_rk65`](#kstrajintegratorrk65)<br>[`integrator_rk8`](#kstrajintegratorrk8)<br>[`integrator_rk86`](#kstrajintegratorrk86)<br>[`integrator_rk87`](#kstrajintegratorrk87)<br>[`integrator_rkdp54`](#kstrajintegratorrkdp54)<br>[`integrator_rkdp853`](#kstrajintegratorrkdp853)<br>[`interpolator_crk`](#kstrajinterpolatorcontinuousrungekutta)<br>[`interpolator_fast`](#kstrajinterpolatorfast)<br>[`interpolator_hermite`](#kstrajinterpolatorhermite)<br>[`term_gravity`](#kstrajtermgravity)<br>[`term_propagation`](#kstrajtermpropagation)<br>[`term_synchrotron`](#kstrajtermsynchrotron)|*`KSTrajControlBChange`*<br>*`KSTrajControlCyclotron`*<br>*`KSTrajControlEnergy`*<br>*`KSTrajControlLength`*<br>*`KSTrajControlMDot`*<br>*`KSTrajControlMagneticMoment`*<br>*`KSTrajControlMomentumNumericalError`*<br>*`KSTrajControlPositionNumericalError`*<br>*`KSTrajControlSpinPrecession`*<br>*`KSTrajControlTime`*<br>*`KSTrajIntegratorRK54`*<br>*`KSTrajIntegratorRK65`*<br>*`KSTrajIntegratorRK8`*<br>*`KSTrajIntegratorRK86`*<br>*`KSTrajIntegratorRK87`*<br>*`KSTrajIntegratorRKDP54`*<br>*`KSTrajIntegratorRKDP853`*<br>*`KSTrajInterpolatorContinuousRungeKutta`*<br>*`KSTrajInterpolatorFast`*<br>*`KSTrajInterpolatorHermite`*<br>*`KSTrajTermGravity`*<br>*`KSTrajTermPropagation`*<br>*`KSTrajTermSynchrotron`*|`attempt_limit`<br>`max_segments`<br>`name`<br>`piecewise_tolerance`|*`unsigned int`*<br>*`unsigned int`*<br>*`string`*<br>*`double`*|

### KSTrajTrajectoryElectric

|element name|source files|child elements|child types|attributes|attribute types|
|-----|-----|-----|-----|-----|-----|
|<a name="kstrajtrajectoryelectric">`kstraj_trajectory_electric`</a>|[*KSTrajTrajectoryElectricBuilder.cxx*](https://github.com/KATRIN-Experiment/Kassiopeia/tree/main/Kassiopeia/Bindings/Trajectories/Source/KSTrajTrajectoryElectricBuilder.cxx)|[`control_B_change`](#kstrajcontrolbchange)<br>[`control_length`](#kstrajcontrollength)<br>[`control_time`](#kstrajcontroltime)<br>[`integrator_rk54`](#kstrajintegratorrk54)<br>[`integrator_rk65`](#kstrajintegratorrk65)<br>[`integrator_rk8`](#kstrajintegratorrk8)<br>[`integrator_rk86`](#kstrajintegratorrk86)<br>[`integrator_rk87`](#kstrajintegratorrk87)<br>[`integrator_rkdp54`](#kstrajintegratorrkdp54)<br>[`integrator_rkdp853`](#kstrajintegratorrkdp853)<br>[`interpolator_crk`](#kstrajinterpolatorcontinuousrungekutta)<br>[`interpolator_fast`](#kstrajinterpolatorfast)<br>[`interpolator_hermite`](#kstrajinterpolatorhermite)<br>[`term_propagation`](#kstrajtermpropagation)|*`KSTrajControlBChange`*<br>*`KSTrajControlLength`*<br>*`KSTrajControlTime`*<br>*`KSTrajIntegratorRK54`*<br>*`KSTrajIntegratorRK65`*<br>*`KSTrajIntegratorRK8`*<br>*`KSTrajIntegratorRK86`*<br>*`KSTrajIntegratorRK87`*<br>*`KSTrajIntegratorRKDP54`*<br>*`KSTrajIntegratorRKDP853`*<br>*`KSTrajInterpolatorContinuousRungeKutta`*<br>*`KSTrajInterpolatorFast`*<br>*`KSTrajInterpolatorHermite`*<br>*`KSTrajTermPropagation`*|`attempt_limit`<br>`max_segments`<br>`name`<br>`piecewise_tolerance`|*`unsigned int`*<br>*`unsigned int`*<br>*`string`*<br>*`double`*|

### KSTrajTrajectoryExact

|element name|source files|child elements|child types|attributes|attribute types|
|-----|-----|-----|-----|-----|-----|
|<a name="kstrajtrajectoryexact">`kstraj_trajectory_exact`</a>|[*KSTrajTrajectoryExactBuilder.cxx*](https://github.com/KATRIN-Experiment/Kassiopeia/tree/main/Kassiopeia/Bindings/Trajectories/Source/KSTrajTrajectoryExactBuilder.cxx)|[`control_B_change`](#kstrajcontrolbchange)<br>[`control_cyclotron`](#kstrajcontrolcyclotron)<br>[`control_energy`](#kstrajcontrolenergy)<br>[`control_length`](#kstrajcontrollength)<br>[`control_magnetic_moment`](#kstrajcontrolmagneticmoment)<br>[`control_momentum_error`](#kstrajcontrolmomentumnumericalerror)<br>[`control_position_error`](#kstrajcontrolpositionnumericalerror)<br>[`control_time`](#kstrajcontroltime)<br>[`integrator_rk54`](#kstrajintegratorrk54)<br>[`integrator_rk65`](#kstrajintegratorrk65)<br>[`integrator_rk8`](#kstrajintegratorrk8)<br>[`integrator_rk86`](#kstrajintegratorrk86)<br>[`integrator_rk87`](#kstrajintegratorrk87)<br>[`integrator_rkdp54`](#kstrajintegratorrkdp54)<br>[`integrator_rkdp853`](#kstrajintegratorrkdp853)<br>[`interpolator_crk`](#kstrajinterpolatorcontinuousrungekutta)<br>[`interpolator_fast`](#kstrajinterpolatorfast)<br>[`interpolator_hermite`](#kstrajinterpolatorhermite)<br>[`term_gravity`](#kstrajtermgravity)<br>[`term_propagation`](#kstrajtermpropagation)<br>[`term_synchrotron`](#kstrajtermsynchrotron)|*`KSTrajControlBChange`*<br>*`KSTrajControlCyclotron`*<br>*`KSTrajControlEnergy`*<br>*`KSTrajControlLength`*<br>*`KSTrajControlMagneticMoment`*<br>*`KSTrajControlMomentumNumericalError`*<br>*`KSTrajControlPositionNumericalError`*<br>*`KSTrajControlTime`*<br>*`KSTrajIntegratorRK54`*<br>*`KSTrajIntegratorRK65`*<br>*`KSTrajIntegratorRK8`*<br>*`KSTrajIntegratorRK86`*<br>*`KSTrajIntegratorRK87`*<br>*`KSTrajIntegratorRKDP54`*<br>*`KSTrajIntegratorRKDP853`*<br>*`KSTrajInterpolatorContinuousRungeKutta`*<br>*`KSTrajInterpolatorFast`*<br>*`KSTrajInterpolatorHermite`*<br>*`KSTrajTermGravity`*<br>*`KSTrajTermPropagation`*<br>*`KSTrajTermSynchrotron`*|`attempt_limit`<br>`max_segments`<br>`name`<br>`piecewise_tolerance`|*`unsigned int`*<br>*`unsigned int`*<br>*`string`*<br>*`double`*|

### KSTrajTrajectoryExactSpin

|element name|source files|child elements|child types|attributes|attribute types|
|-----|-----|-----|-----|-----|-----|
|<a name="kstrajtrajectoryexactspin">`kstraj_trajectory_exact_spin`</a>|[*KSTrajTrajectoryExactSpinBuilder.cxx*](https://github.com/KATRIN-Experiment/Kassiopeia/tree/main/Kassiopeia/Bindings/Trajectories/Source/KSTrajTrajectoryExactSpinBuilder.cxx)|[`control_B_change`](#kstrajcontrolbchange)<br>[`control_cyclotron`](#kstrajcontrolcyclotron)<br>[`control_energy`](#kstrajcontrolenergy)<br>[`control_length`](#kstrajcontrollength)<br>[`control_magnetic_moment`](#kstrajcontrolmagneticmoment)<br>[`control_momentum_error`](#kstrajcontrolmomentumnumericalerror)<br>[`control_position_error`](#kstrajcontrolpositionnumericalerror)<br>[`control_spin_precession`](#kstrajcontrolspinprecession)<br>[`control_time`](#kstrajcontroltime)<br>[`integrator_rk54`](#kstrajintegratorrk54)<br>[`integrator_rk65`](#kstrajintegratorrk65)<br>[`integrator_rk8`](#kstrajintegratorrk8)<br>[`integrator_rk86`](#kstrajintegratorrk86)<br>[`integrator_rk87`](#kstrajintegratorrk87)<br>[`integrator_rkdp54`](#kstrajintegratorrkdp54)<br>[`integrator_rkdp853`](#kstrajintegratorrkdp853)<br>[`interpolator_crk`](#kstrajinterpolatorcontinuousrungekutta)<br>[`interpolator_fast`](#kstrajinterpolatorfast)<br>[`interpolator_hermite`](#kstrajinterpolatorhermite)<br>[`term_gravity`](#kstrajtermgravity)<br>[`term_propagation`](#kstrajtermpropagation)<br>[`term_synchrotron`](#kstrajtermsynchrotron)|*`KSTrajControlBChange`*<br>*`KSTrajControlCyclotron`*<br>*`KSTrajControlEnergy`*<br>*`KSTrajControlLength`*<br>*`KSTrajControlMagneticMoment`*<br>*`KSTrajControlMomentumNumericalError`*<br>*`KSTrajControlPositionNumericalError`*<br>*`KSTrajControlSpinPrecession`*<br>*`KSTrajControlTime`*<br>*`KSTrajIntegratorRK54`*<br>*`KSTrajIntegratorRK65`*<br>*`KSTrajIntegratorRK8`*<br>*`KSTrajIntegratorRK86`*<br>*`KSTrajIntegratorRK87`*<br>*`KSTrajIntegratorRKDP54`*<br>*`KSTrajIntegratorRKDP853`*<br>*`KSTrajInterpolatorContinuousRungeKutta`*<br>*`KSTrajInterpolatorFast`*<br>*`KSTrajInterpolatorHermite`*<br>*`KSTrajTermGravity`*<br>*`KSTrajTermPropagation`*<br>*`KSTrajTermSynchrotron`*|`attempt_limit`<br>`max_segments`<br>`name`<br>`piecewise_tolerance`|*`unsigned int`*<br>*`unsigned int`*<br>*`string`*<br>*`double`*|

### KSTrajTrajectoryExactTrapped

|element name|source files|child elements|child types|attributes|attribute types|
|-----|-----|-----|-----|-----|-----|
|<a name="kstrajtrajectoryexacttrapped">`kstraj_trajectory_exact_trapped`</a>|[*KSTrajTrajectoryExactTrappedBuilder.cxx*](https://github.com/KATRIN-Experiment/Kassiopeia/tree/main/Kassiopeia/Bindings/Trajectories/Source/KSTrajTrajectoryExactTrappedBuilder.cxx)|[`control_length`](#kstrajcontrollength)<br>[`control_time`](#kstrajcontroltime)<br>[`integrator_rk8`](#kstrajintegratorrk8)<br>[`integrator_sym4`](#kstrajintegratorsym4)<br>[`interpolator_crk`](#kstrajinterpolatorcontinuousrungekutta )<br>[`interpolator_fast`](#kstrajinterpolatorfast )<br>[`interpolator_hermite`](#kstrajinterpolatorhermite )<br>[`term_propagation`](#kstrajtermpropagation)<br>[`term_synchrotron`](#kstrajtermsynchrotron)|*`KSTrajControlLength`*<br>*`KSTrajControlTime`*<br>*`KSTrajIntegratorRK8`*<br>*`KSTrajIntegratorSym4`*<br>*`KSTrajInterpolatorContinuousRungeKutta `*<br>*`KSTrajInterpolatorFast `*<br>*`KSTrajInterpolatorHermite `*<br>*`KSTrajTermPropagation`*<br>*`KSTrajTermSynchrotron`*|`attempt_limit`<br>`max_segments`<br>`name`<br>`piecewise_tolerance`|*`unsigned int`*<br>*`unsigned int`*<br>*`string`*<br>*`double`*|

#### KSTrajInterpolatorContinuousRungeKutta 

|element name|source files|child elements|child types|attributes|attribute types|
|-----|-----|-----|-----|-----|-----|
|<a name="kstrajinterpolatorcontinuousrungekutta ">`interpolator_crk`</a>|[*KSTrajTrajectoryExactTrappedBuilder.cxx*](https://github.com/KATRIN-Experiment/Kassiopeia/tree/main/Kassiopeia/Bindings/Trajectories/Source/KSTrajTrajectoryExactTrappedBuilder.cxx)|—    |—    |—    |—    |

#### KSTrajInterpolatorFast 

|element name|source files|child elements|child types|attributes|attribute types|
|-----|-----|-----|-----|-----|-----|
|<a name="kstrajinterpolatorfast ">`interpolator_fast`</a>|[*KSTrajTrajectoryExactTrappedBuilder.cxx*](https://github.com/KATRIN-Experiment/Kassiopeia/tree/main/Kassiopeia/Bindings/Trajectories/Source/KSTrajTrajectoryExactTrappedBuilder.cxx)|—    |—    |—    |—    |

#### KSTrajInterpolatorHermite 

|element name|source files|child elements|child types|attributes|attribute types|
|-----|-----|-----|-----|-----|-----|
|<a name="kstrajinterpolatorhermite ">`interpolator_hermite`</a>|[*KSTrajTrajectoryExactTrappedBuilder.cxx*](https://github.com/KATRIN-Experiment/Kassiopeia/tree/main/Kassiopeia/Bindings/Trajectories/Source/KSTrajTrajectoryExactTrappedBuilder.cxx)|—    |—    |—    |—    |

### KSTrajTrajectoryLinear

|element name|source files|child elements|child types|attributes|attribute types|
|-----|-----|-----|-----|-----|-----|
|<a name="kstrajtrajectorylinear">`kstraj_trajectory_linear`</a>|[*KSTrajTrajectoryLinearBuilder.cxx*](https://github.com/KATRIN-Experiment/Kassiopeia/tree/main/Kassiopeia/Bindings/Trajectories/Source/KSTrajTrajectoryLinearBuilder.cxx)|—    |—    |`length`<br>`name`|*`double`*<br>*`string`*|

### KSTrajTrajectoryMagnetic

|element name|source files|child elements|child types|attributes|attribute types|
|-----|-----|-----|-----|-----|-----|
|<a name="kstrajtrajectorymagnetic">`kstraj_trajectory_magnetic`</a>|[*KSTrajTrajectoryMagneticBuilder.cxx*](https://github.com/KATRIN-Experiment/Kassiopeia/tree/main/Kassiopeia/Bindings/Trajectories/Source/KSTrajTrajectoryMagneticBuilder.cxx)|[`control_B_change`](#kstrajcontrolbchange)<br>[`control_length`](#kstrajcontrollength)<br>[`control_time`](#kstrajcontroltime)<br>[`integrator_rk54`](#kstrajintegratorrk54)<br>[`integrator_rk65`](#kstrajintegratorrk65)<br>[`integrator_rk8`](#kstrajintegratorrk8)<br>[`integrator_rk86`](#kstrajintegratorrk86)<br>[`integrator_rk87`](#kstrajintegratorrk87)<br>[`integrator_rkdp54`](#kstrajintegratorrkdp54)<br>[`integrator_rkdp853`](#kstrajintegratorrkdp853)<br>[`interpolator_crk`](#kstrajinterpolatorcontinuousrungekutta)<br>[`interpolator_fast`](#kstrajinterpolatorfast)<br>[`interpolator_hermite`](#kstrajinterpolatorhermite)<br>[`term_propagation`](#kstrajtermpropagation)|*`KSTrajControlBChange`*<br>*`KSTrajControlLength`*<br>*`KSTrajControlTime`*<br>*`KSTrajIntegratorRK54`*<br>*`KSTrajIntegratorRK65`*<br>*`KSTrajIntegratorRK8`*<br>*`KSTrajIntegratorRK86`*<br>*`KSTrajIntegratorRK87`*<br>*`KSTrajIntegratorRKDP54`*<br>*`KSTrajIntegratorRKDP853`*<br>*`KSTrajInterpolatorContinuousRungeKutta`*<br>*`KSTrajInterpolatorFast`*<br>*`KSTrajInterpolatorHermite`*<br>*`KSTrajTermPropagation`*|`attempt_limit`<br>`max_segments`<br>`name`<br>`piecewise_tolerance`|*`unsigned int`*<br>*`unsigned int`*<br>*`string`*<br>*`double`*|

### KSWriteASCII

|element name|source files|child elements|child types|attributes|attribute types|
|-----|-----|-----|-----|-----|-----|
|<a name="kswriteascii">`kswrite_ascii`</a>|[*KSWriteASCIIBuilder.cxx*](https://github.com/KATRIN-Experiment/Kassiopeia/tree/main/Kassiopeia/Bindings/Writers/Source/KSWriteASCIIBuilder.cxx)|—    |—    |`base`<br>`name`<br>`path`<br>`precision`|*`string`*<br>*`string`*<br>*`string`*<br>*`unsigned int`*|

### KSWriteROOT

|element name|source files|child elements|child types|attributes|attribute types|
|-----|-----|-----|-----|-----|-----|
|<a name="kswriteroot">`kswrite_root`</a>|[*KSWriteROOTBuilder.cxx*](https://github.com/KATRIN-Experiment/Kassiopeia/tree/main/Kassiopeia/Bindings/Writers/Source/KSWriteROOTBuilder.cxx)|—    |—    |`base`<br>`name`<br>`path`|*`string`*<br>*`string`*<br>*`string`*|

### KSWriteROOTConditionOutputData

|element name|source files|child elements|child types|attributes|attribute types|
|-----|-----|-----|-----|-----|-----|
|<a name="kswriterootconditionoutputdata">`kswrite_root_condition_output`</a>|[*KSWriteROOTConditionOutputBuilder.cxx*](https://github.com/KATRIN-Experiment/Kassiopeia/tree/main/Kassiopeia/Bindings/Writers/Source/KSWriteROOTConditionOutputBuilder.cxx)|—    |—    |`group`<br>`max_value`<br>`min_value`<br>`name`<br>`parent`|*`string`*<br>*`double`*<br>*`double`*<br>*`string`*<br>*`string`*|

### KSWriteROOTConditionPeriodicData

|element name|source files|child elements|child types|attributes|attribute types|
|-----|-----|-----|-----|-----|-----|
|<a name="kswriterootconditionperiodicdata">`kswrite_root_condition_periodic`</a>|[*KSWriteROOTConditionPeriodicBuilder.cxx*](https://github.com/KATRIN-Experiment/Kassiopeia/tree/main/Kassiopeia/Bindings/Writers/Source/KSWriteROOTConditionPeriodicBuilder.cxx)|—    |—    |`group`<br>`increment`<br>`initial_max`<br>`initial_min`<br>`name`<br>`parent`<br>`reset_max`<br>`reset_min`|*`string`*<br>*`double`*<br>*`double`*<br>*`double`*<br>*`string`*<br>*`string`*<br>*`double`*<br>*`double`*|

### KSWriteROOTConditionStepData

|element name|source files|child elements|child types|attributes|attribute types|
|-----|-----|-----|-----|-----|-----|
|<a name="kswriterootconditionstepdata">`kswrite_root_condition_step`</a>|[*KSWriteROOTConditionStepBuilder.cxx*](https://github.com/KATRIN-Experiment/Kassiopeia/tree/main/Kassiopeia/Bindings/Writers/Source/KSWriteROOTConditionStepBuilder.cxx)|—    |—    |`group`<br>`name`<br>`nth_step`<br>`parent`|*`string`*<br>*`string`*<br>*`int`*<br>*`string`*|

### KSWriteROOTConditionTerminatorData

|element name|source files|child elements|child types|attributes|attribute types|
|-----|-----|-----|-----|-----|-----|
|<a name="kswriterootconditionterminatordata">`kswrite_root_condition_terminator`</a>|[*KSWriteROOTConditionTerminatorBuilder.cxx*](https://github.com/KATRIN-Experiment/Kassiopeia/tree/main/Kassiopeia/Bindings/Writers/Source/KSWriteROOTConditionTerminatorBuilder.cxx)|—    |—    |`group`<br>`match_terminator`<br>`name`<br>`parent`|*`string`*<br>*`string`*<br>*`string`*<br>*`string`*|

### KSWriteVTK

|element name|source files|child elements|child types|attributes|attribute types|
|-----|-----|-----|-----|-----|-----|
|<a name="kswritevtk">`kswrite_vtk`</a>|[*KSWriteVTKBuilder.cxx*](https://github.com/KATRIN-Experiment/Kassiopeia/tree/main/Kassiopeia/Bindings/Writers/Source/KSWriteVTKBuilder.cxx)|—    |—    |`base`<br>`name`<br>`path`|*`string`*<br>*`string`*<br>*`string`*|

## KEMRoot

|element name|source files|child elements|child types|attributes|attribute types|
|-----|-----|-----|-----|-----|-----|
|<a name="kemroot">`kemfield`</a>|[*KEMToolboxBuilder.cc*](https://github.com/KATRIN-Experiment/Kassiopeia/tree/main/KEMField/Source/Bindings/Utilities/src/KEMToolboxBuilder.cc)|[`constant_electric_field`](#kelectrostaticconstantfield)<br>[`constant_magnetic_field`](#kmagnetostaticconstantfield)<br>[`electric_potentialmap`](#kelectrostaticpotentialmap)<br>[`electric_potentialmap_calculator`](#kelectrostaticpotentialmapcalculator)<br>[`electric_quadrupole_field`](#kelectricquadrupolefield)<br>[`electromagnet_field`](#kgstaticelectromagnetfield)<br>[`electrostatic_field`](#kgelectrostaticboundaryfield)<br>[`induced_azimuthal_electric_field`](#kinducedazimuthalelectricfield)<br>[`linear_electric_field`](#kelectrostaticlinearfield)<br>[`magnetic_dipole_field`](#kmagneticdipolefield)<br>[`magnetic_fieldmap`](#kmagnetostaticfieldmap)<br>[`magnetic_fieldmap_calculator`](#kmagnetostaticfieldmapcalculator)<br>[`magnetic_superposition_field`](#kmagneticsuperpositionfield)<br>[`ramped_electric_field`](#krampedelectricfield)<br>[`ramped_magnetic_field`](#krampedmagneticfield)<br>[`ramped_transitional_electric_field`](#krampedelectric2field)|*`KElectrostaticConstantField`*<br>*`KMagnetostaticConstantField`*<br>*`KElectrostaticPotentialmap`*<br>*`KElectrostaticPotentialmapCalculator`*<br>*`KElectricQuadrupoleField`*<br>*`KGStaticElectromagnetField`*<br>*`KGElectrostaticBoundaryField`*<br>*`KInducedAzimuthalElectricField`*<br>*`KElectrostaticLinearField`*<br>*`KMagneticDipoleField`*<br>*`KMagnetostaticFieldmap`*<br>*`KMagnetostaticFieldmapCalculator`*<br>*`KMagneticSuperpositionField`*<br>*`KRampedElectricField`*<br>*`KRampedMagneticField`*<br>*`KRampedElectric2Field`*|—    |—    |

### KElectrostaticLinearField

|element name|source files|child elements|child types|attributes|attribute types|
|-----|-----|-----|-----|-----|-----|
|<a name="kelectrostaticlinearfield">`linear_electric_field`</a>|[*KElectrostaticLinearFieldBuilder.cc*](https://github.com/KATRIN-Experiment/Kassiopeia/tree/main/KEMField/Source/Bindings/Fields/Electric/src/KElectrostaticLinearFieldBuilder.cc)|—    |—    |`U1`<br>`U2`<br>`name`<br>`surface`<br>`z1`<br>`z2`|*`double`*<br>*`double`*<br>*`string`*<br>*`string`*<br>*`double`*<br>*`double`*|

## KMessageTable

|element name|source files|child elements|child types|attributes|attribute types|
|-----|-----|-----|-----|-----|-----|
|<a name="kmessagetable">`messages`</a>|[*KMessageBuilder.cxx*](https://github.com/KATRIN-Experiment/Kassiopeia/tree/main/Kommon/Core/Bindings/KMessageBuilder.cxx)|[`file`](#ktextfile)<br>[`message`](#kmessagedata)|*`KTextFile`*<br>*`KMessageData`*|`format`<br>`log`<br>`parser_context`<br>`precision`<br>`shutdown_message`<br>`terminal`|*`string`*<br>*`string`*<br>*`bool`*<br>*`KMessagePrecision`*<br>*`bool`*<br>*`string`*|

### KTextFile

|element name|source files|child elements|child types|attributes|attribute types|
|-----|-----|-----|-----|-----|-----|
|<a name="ktextfile">`file`</a>|[*KMessageBuilder.cxx*](https://github.com/KATRIN-Experiment/Kassiopeia/tree/main/Kommon/Core/Bindings/KMessageBuilder.cxx)|—    |—    |—    |—    |

### KMessageData

|element name|source files|child elements|child types|attributes|attribute types|
|-----|-----|-----|-----|-----|-----|
|<a name="kmessagedata">`message`</a>|[*KMessageBuilder.cxx*](https://github.com/KATRIN-Experiment/Kassiopeia/tree/main/Kommon/Core/Bindings/KMessageBuilder.cxx)|—    |—    |`format`<br>`key`<br>`log`<br>`parser_context`<br>`precision`<br>`shutdown_message`<br>`terminal`|*`string`*<br>*`string`*<br>*`string`*<br>*`bool`*<br>*`KMessagePrecision`*<br>*`bool`*<br>*`string`*|

## KROOTWindow

|element name|source files|child elements|child types|attributes|attribute types|
|-----|-----|-----|-----|-----|-----|
|<a name="krootwindow">`root_window`</a>|[*KROOTWindowBuilder.cxx*](https://github.com/KATRIN-Experiment/Kassiopeia/tree/main/Kommon/Root/Utility/KROOTWindowBuilder.cxx)|[`root_geometry_painter`](#kgrootgeometrypainter)<br>[`root_magfield_painter`](#ksrootmagfieldpainter)<br>[`root_pad`](#krootpad)<br>[`root_potential_painter`](#ksrootpotentialpainter)<br>[`root_track_painter`](#ksroottrackpainter)<br>[`root_zh_painter`](#ksrootzonalharmonicspainter)|*`KGROOTGeometryPainter`*<br>*`KSROOTMagFieldPainter`*<br>*`KROOTPad`*<br>*`KSROOTPotentialPainter`*<br>*`KSROOTTrackPainter`*<br>*`KSROOTZonalHarmonicsPainter`*|`active`<br>`canvas_height`<br>`canvas_width`<br>`name`<br>`path`<br>`write_enabled`|*`bool`*<br>*`unsigned int`*<br>*`unsigned int`*<br>*`string`*<br>*`string`*<br>*`bool`*|

### KGROOTGeometryPainter

|element name|source files|child elements|child types|attributes|attribute types|
|-----|-----|-----|-----|-----|-----|
|<a name="kgrootgeometrypainter">`root_geometry_painter`</a>|[*KGROOTGeometryPainterBuilder.cc*](https://github.com/KATRIN-Experiment/Kassiopeia/tree/main/KGeoBag/Source/Bindings/Visualization/Root/Source/KGROOTGeometryPainterBuilder.cc)|—    |—    |`epsilon`<br>`file`<br>`name`<br>`path`<br>`plane_normal`<br>`plane_point`<br>`save_json`<br>`save_svg`<br>`show_labels`<br>`spaces`<br>`surfaces`<br>`swap_axis`|*`double`*<br>*`string`*<br>*`string`*<br>*`string`*<br>*`KThreeVector`*<br>*`KThreeVector`*<br>*`bool`*<br>*`bool`*<br>*`bool`*<br>*`string`*<br>*`string`*<br>*`bool`*|

### KSROOTMagFieldPainter

|element name|source files|child elements|child types|attributes|attribute types|
|-----|-----|-----|-----|-----|-----|
|<a name="ksrootmagfieldpainter">`root_magfield_painter`</a>|[*KSROOTMagFieldPainterBuilder.cxx*](https://github.com/KATRIN-Experiment/Kassiopeia/tree/main/Kassiopeia/Bindings/Visualization/Source/KSROOTMagFieldPainterBuilder.cxx)|—    |—    |`axial_symmetry`<br>`draw`<br>`magnetic_field`<br>`magnetic_gradient_numerical`<br>`name`<br>`plot`<br>`r_max`<br>`r_steps`<br>`x_axis`<br>`y_axis`<br>`z_axis_logscale`<br>`z_fix`<br>`z_max`<br>`z_min`<br>`z_steps`|*`bool`*<br>*`string`*<br>*`string`*<br>*`bool`*<br>*`string`*<br>*`string`*<br>*`double`*<br>*`int`*<br>*`string`*<br>*`string`*<br>*`bool`*<br>*`double`*<br>*`double`*<br>*`double`*<br>*`int`*|

### KROOTPad

|element name|source files|child elements|child types|attributes|attribute types|
|-----|-----|-----|-----|-----|-----|
|<a name="krootpad">`root_pad`</a>|[*KROOTWindowBuilder.cxx*](https://github.com/KATRIN-Experiment/Kassiopeia/tree/main/Kommon/Root/Utility/KROOTWindowBuilder.cxx)|[`root_geometry_painter`](#kgrootgeometrypainter)<br>[`root_magfield_painter`](#ksrootmagfieldpainter)<br>[`root_potential_painter`](#ksrootpotentialpainter)<br>[`root_track_painter`](#ksroottrackpainter)<br>[`root_zh_painter`](#ksrootzonalharmonicspainter)|*`KGROOTGeometryPainter`*<br>*`KSROOTMagFieldPainter`*<br>*`KSROOTPotentialPainter`*<br>*`KSROOTTrackPainter`*<br>*`KSROOTZonalHarmonicsPainter`*|`name`<br>`xlow`<br>`xup`<br>`ylow`<br>`yup`|*`string`*<br>*`double`*<br>*`double`*<br>*`double`*<br>*`double`*|

#### KSROOTPotentialPainter

|element name|source files|child elements|child types|attributes|attribute types|
|-----|-----|-----|-----|-----|-----|
|<a name="ksrootpotentialpainter">`root_potential_painter`</a>|[*KSROOTPotentialPainterBuilder.cxx*](https://github.com/KATRIN-Experiment/Kassiopeia/tree/main/Kassiopeia/Bindings/Visualization/Source/KSROOTPotentialPainterBuilder.cxx)|—    |—    |`calc_pot`<br>`compare_fields`<br>`electric_field`<br>`name`<br>`r_max`<br>`r_steps`<br>`reference_field`<br>`x_axis`<br>`y_axis`<br>`z_max`<br>`z_min`<br>`z_steps`|*`bool`*<br>*`bool`*<br>*`string`*<br>*`string`*<br>*`double`*<br>*`int`*<br>*`string`*<br>*`string`*<br>*`string`*<br>*`double`*<br>*`double`*<br>*`int`*|

#### KSROOTTrackPainter

|element name|source files|child elements|child types|attributes|attribute types|
|-----|-----|-----|-----|-----|-----|
|<a name="ksroottrackpainter">`root_track_painter`</a>|[*KSROOTTrackPainterBuilder.cxx*](https://github.com/KATRIN-Experiment/Kassiopeia/tree/main/Kassiopeia/Bindings/Visualization/Source/KSROOTTrackPainterBuilder.cxx)|—    |—    |`add_color`<br>`axial_mirror`<br>`base`<br>`color`<br>`color_mode`<br>`color_palette`<br>`color_variable`<br>`draw_options`<br>`epsilon`<br>`name`<br>`path`<br>`plane_normal`<br>`plane_point`<br>`plot_mode`<br>`position_name`<br>`step_output_group_name`<br>`swap_axis`<br>`track_output_group_name`<br>`x_axis`<br>`y_axis`|*`string`*<br>*`bool`*<br>*`string`*<br>*`string`*<br>*`string`*<br>*`string`*<br>*`string`*<br>*`string`*<br>*`double`*<br>*`string`*<br>*`string`*<br>*`KThreeVector`*<br>*`KThreeVector`*<br>*`string`*<br>*`string`*<br>*`string`*<br>*`bool`*<br>*`string`*<br>*`string`*<br>*`string`*|

#### KSROOTZonalHarmonicsPainter

|element name|source files|child elements|child types|attributes|attribute types|
|-----|-----|-----|-----|-----|-----|
|<a name="ksrootzonalharmonicspainter">`root_zh_painter`</a>|[*KSROOTZonalHarmonicsPainterBuilder.cxx*](https://github.com/KATRIN-Experiment/Kassiopeia/tree/main/Kassiopeia/Bindings/Visualization/Source/KSROOTZonalHarmonicsPainterBuilder.cxx)|—    |—    |`electric_field`<br>`file`<br>`geometry_type`<br>`magnetic_field`<br>`name`<br>`path`<br>`r_dist`<br>`r_max`<br>`r_min`<br>`r_steps`<br>`radial_safety_margin`<br>`write`<br>`x_axis`<br>`y_axis`<br>`z_dist`<br>`z_max`<br>`z_min`<br>`z_steps`|*`string`*<br>*`string`*<br>*`string `*<br>*`string`*<br>*`string`*<br>*`string`*<br>*`double`*<br>*`double`*<br>*`double`*<br>*`int`*<br>*`double `*<br>*`bool`*<br>*`string`*<br>*`string`*<br>*`double`*<br>*`double`*<br>*`double`*<br>*`int`*|

## KApplicationRunner

|element name|source files|child elements|child types|attributes|attribute types|
|-----|-----|-----|-----|-----|-----|
|<a name="kapplicationrunner">`run`</a>|[*KApplicationRunnerBuilder.cxx*](https://github.com/KATRIN-Experiment/Kassiopeia/tree/main/Kommon/Core/Bindings/KApplicationRunnerBuilder.cxx)|[`app`](#knamedreference)|*`KNamedReference`*|`name`|*`string`*|

### KNamedReference

|element name|source files|child elements|child types|attributes|attribute types|
|-----|-----|-----|-----|-----|-----|
|<a name="knamedreference">`app`</a>|[*KApplicationRunnerBuilder.cxx*](https://github.com/KATRIN-Experiment/Kassiopeia/tree/main/Kommon/Core/Bindings/KApplicationRunnerBuilder.cxx)|—    |—    |`Name`|*`string`*|

## KVTKWindow

|element name|source files|child elements|child types|attributes|attribute types|
|-----|-----|-----|-----|-----|-----|
|<a name="kvtkwindow">`vtk_window`</a>|[*KVTKWindowBuilder.cxx*](https://github.com/KATRIN-Experiment/Kassiopeia/tree/main/Kommon/Vtk/Utility/KVTKWindowBuilder.cxx)|[`vtk_axial_mesh_painter`](#kgvtkaxialmeshpainter)<br>[`vtk_distance_tester`](#kgvtkdistancetester)<br>[`vtk_generator_painter`](#ksvtkgeneratorpainter)<br>[`vtk_geometry_painter`](#kgvtkgeometrypainter)<br>[`vtk_mesh_painter`](#kgvtkmeshpainter)<br>[`vtk_normal_tester`](#kgvtknormaltester)<br>[`vtk_outside_tester`](#kgvtkoutsidetester)<br>[`vtk_point_tester`](#kgvtkpointtester)<br>[`vtk_random_point_tester`](#kgvtkrandompointtester)<br>[`vtk_track_painter`](#ksvtktrackpainter)<br>[`vtk_track_terminator_painter`](#ksvtktrackterminatorpainter)|*`KGVTKAxialMeshPainter`*<br>*`KGVTKDistanceTester`*<br>*`KSVTKGeneratorPainter`*<br>*`KGVTKGeometryPainter`*<br>*`KGVTKMeshPainter`*<br>*`KGVTKNormalTester`*<br>*`KGVTKOutsideTester`*<br>*`KGVTKPointTester`*<br>*`KGVTKRandomPointTester`*<br>*`KSVTKTrackPainter`*<br>*`KSVTKTrackTerminatorPainter`*|`depth_peeling`<br>`enable_axis`<br>`enable_data`<br>`enable_display`<br>`enable_help`<br>`enable_parallel_projection`<br>`enable_write`<br>`eye_angle`<br>`frame_color_blue`<br>`frame_color_green`<br>`frame_color_red`<br>`frame_size_x`<br>`frame_size_y`<br>`frame_title`<br>`multi_samples`<br>`name`<br>`view_angle`|*`unsigned int`*<br>*`bool`*<br>*`bool`*<br>*`bool`*<br>*`bool`*<br>*`bool`*<br>*`bool`*<br>*`double`*<br>*`float`*<br>*`float`*<br>*`float`*<br>*`unsigned int`*<br>*`unsigned int`*<br>*`string`*<br>*`unsigned int`*<br>*`string`*<br>*`double`*|

### KGVTKAxialMeshPainter

|element name|source files|child elements|child types|attributes|attribute types|
|-----|-----|-----|-----|-----|-----|
|<a name="kgvtkaxialmeshpainter">`vtk_axial_mesh_painter`</a>|[*KGVTKAxialMeshPainterBuilder.cc*](https://github.com/KATRIN-Experiment/Kassiopeia/tree/main/KGeoBag/Source/Bindings/Visualization/Vtk/Source/KGVTKAxialMeshPainterBuilder.cc)|—    |—    |`arc_count`<br>`color_mode`<br>`file`<br>`name`<br>`spaces`<br>`surfaces`|*`unsigned int`*<br>*`string`*<br>*`string`*<br>*`string`*<br>*`string`*<br>*`string`*|

### KGVTKDistanceTester

|element name|source files|child elements|child types|attributes|attribute types|
|-----|-----|-----|-----|-----|-----|
|<a name="kgvtkdistancetester">`vtk_distance_tester`</a>|[*KGVTKDistanceTesterBuilder.cc*](https://github.com/KATRIN-Experiment/Kassiopeia/tree/main/KGeoBag/Source/Bindings/Visualization/Vtk/Source/KGVTKDistanceTesterBuilder.cc)|—    |—    |`name`<br>`sample_count`<br>`sample_disk_normal`<br>`sample_disk_origin`<br>`sample_disk_radius`<br>`spaces`<br>`surfaces`<br>`vertex_size`|*`string`*<br>*`unsigned int`*<br>*`KThreeVector`*<br>*`KThreeVector`*<br>*`double`*<br>*`string`*<br>*`string`*<br>*`double`*|

### KSVTKGeneratorPainter

|element name|source files|child elements|child types|attributes|attribute types|
|-----|-----|-----|-----|-----|-----|
|<a name="ksvtkgeneratorpainter">`vtk_generator_painter`</a>|[*KSVTKGeneratorPainterBuilder.cxx*](https://github.com/KATRIN-Experiment/Kassiopeia/tree/main/Kassiopeia/Bindings/Visualization/Source/KSVTKGeneratorPainterBuilder.cxx)|—    |—    |`add_color`<br>`add_generator`<br>`color_variable`<br>`electric_field`<br>`file`<br>`magnetic_field`<br>`name`<br>`num_samples`<br>`path`<br>`scale_factor`|*`KThreeVector`*<br>*`string`*<br>*`string`*<br>*`string`*<br>*`string`*<br>*`string`*<br>*`string`*<br>*`int`*<br>*`string`*<br>*`double`*|

### KGVTKGeometryPainter

|element name|source files|child elements|child types|attributes|attribute types|
|-----|-----|-----|-----|-----|-----|
|<a name="kgvtkgeometrypainter">`vtk_geometry_painter`</a>|[*KGVTKGeometryPainterBuilder.cc*](https://github.com/KATRIN-Experiment/Kassiopeia/tree/main/KGeoBag/Source/Bindings/Visualization/Vtk/Source/KGVTKGeometryPainterBuilder.cc)|—    |—    |`file`<br>`name`<br>`path`<br>`spaces`<br>`surfaces`<br>`write_stl`|*`string`*<br>*`string`*<br>*`string`*<br>*`string`*<br>*`string`*<br>*`bool`*|

### KGVTKMeshPainter

|element name|source files|child elements|child types|attributes|attribute types|
|-----|-----|-----|-----|-----|-----|
|<a name="kgvtkmeshpainter">`vtk_mesh_painter`</a>|[*KGVTKMeshPainterBuilder.cc*](https://github.com/KATRIN-Experiment/Kassiopeia/tree/main/KGeoBag/Source/Bindings/Visualization/Vtk/Source/KGVTKMeshPainterBuilder.cc)|—    |—    |`arc_count`<br>`color_mode`<br>`file`<br>`name`<br>`spaces`<br>`surfaces`|*`unsigned int`*<br>*`string`*<br>*`string`*<br>*`string`*<br>*`string`*<br>*`string`*|

### KGVTKNormalTester

|element name|source files|child elements|child types|attributes|attribute types|
|-----|-----|-----|-----|-----|-----|
|<a name="kgvtknormaltester">`vtk_normal_tester`</a>|[*KGVTKNormalTesterBuilder.cc*](https://github.com/KATRIN-Experiment/Kassiopeia/tree/main/KGeoBag/Source/Bindings/Visualization/Vtk/Source/KGVTKNormalTesterBuilder.cc)|—    |—    |`line_size`<br>`name`<br>`normal_color`<br>`normal_length`<br>`point_color`<br>`sample_color`<br>`sample_count`<br>`sample_disk_normal`<br>`sample_disk_origin`<br>`sample_disk_radius`<br>`spaces`<br>`surfaces`<br>`vertex_size`|*`double`*<br>*`string`*<br>*`KGRGBColor`*<br>*`double`*<br>*`KGRGBColor`*<br>*`KGRGBColor`*<br>*`unsigned int`*<br>*`KThreeVector`*<br>*`KThreeVector`*<br>*`double`*<br>*`string`*<br>*`string`*<br>*`double`*|

### KGVTKOutsideTester

|element name|source files|child elements|child types|attributes|attribute types|
|-----|-----|-----|-----|-----|-----|
|<a name="kgvtkoutsidetester">`vtk_outside_tester`</a>|[*KGVTKOutsideTesterBuilder.cc*](https://github.com/KATRIN-Experiment/Kassiopeia/tree/main/KGeoBag/Source/Bindings/Visualization/Vtk/Source/KGVTKOutsideTesterBuilder.cc)|—    |—    |`inside_color`<br>`name`<br>`outside_color`<br>`sample_count`<br>`sample_disk_normal`<br>`sample_disk_origin`<br>`sample_disk_radius`<br>`spaces`<br>`surfaces`<br>`vertex_size`|*`KGRGBColor`*<br>*`string`*<br>*`KGRGBColor`*<br>*`unsigned int`*<br>*`KThreeVector`*<br>*`KThreeVector`*<br>*`double`*<br>*`string`*<br>*`string`*<br>*`double`*|

### KGVTKPointTester

|element name|source files|child elements|child types|attributes|attribute types|
|-----|-----|-----|-----|-----|-----|
|<a name="kgvtkpointtester">`vtk_point_tester`</a>|[*KGVTKPointTesterBuilder.cc*](https://github.com/KATRIN-Experiment/Kassiopeia/tree/main/KGeoBag/Source/Bindings/Visualization/Vtk/Source/KGVTKPointTesterBuilder.cc)|—    |—    |`line_size`<br>`name`<br>`point_color`<br>`sample_color`<br>`sample_count`<br>`sample_disk_normal`<br>`sample_disk_origin`<br>`sample_disk_radius`<br>`spaces`<br>`surfaces`<br>`vertex_size`|*`double`*<br>*`string`*<br>*`KGRGBColor`*<br>*`KGRGBColor`*<br>*`unsigned int`*<br>*`KThreeVector`*<br>*`KThreeVector`*<br>*`double`*<br>*`string`*<br>*`string`*<br>*`double`*|

### KGVTKRandomPointTester

|element name|source files|child elements|child types|attributes|attribute types|
|-----|-----|-----|-----|-----|-----|
|<a name="kgvtkrandompointtester">`vtk_random_point_tester`</a>|[*KGVTKRandomPointTesterBuilder.cc*](https://github.com/KATRIN-Experiment/Kassiopeia/tree/main/KGeoBag/Source/Bindings/Visualization/Vtk/Source/KGVTKRandomPointTesterBuilder.cc)|—    |—    |`name`<br>`sample_color`<br>`spaces`<br>`surfaces`<br>`vertex_size`|*`string`*<br>*`KGRGBColor`*<br>*`string`*<br>*`string`*<br>*`double`*|

### KSVTKTrackPainter

|element name|source files|child elements|child types|attributes|attribute types|
|-----|-----|-----|-----|-----|-----|
|<a name="ksvtktrackpainter">`vtk_track_painter`</a>|[*KSVTKTrackPainterBuilder.cxx*](https://github.com/KATRIN-Experiment/Kassiopeia/tree/main/Kassiopeia/Bindings/Visualization/Source/KSVTKTrackPainterBuilder.cxx)|—    |—    |`color_object`<br>`color_variable`<br>`file`<br>`line_width`<br>`name`<br>`outfile`<br>`path`<br>`point_object`<br>`point_variable`|*`string`*<br>*`string`*<br>*`string`*<br>*`int`*<br>*`string`*<br>*`string`*<br>*`string`*<br>*`string`*<br>*`string`*|

### KSVTKTrackTerminatorPainter

|element name|source files|child elements|child types|attributes|attribute types|
|-----|-----|-----|-----|-----|-----|
|<a name="ksvtktrackterminatorpainter">`vtk_track_terminator_painter`</a>|[*KSVTKTrackTerminatorPainterBuilder.cxx*](https://github.com/KATRIN-Experiment/Kassiopeia/tree/main/Kassiopeia/Bindings/Visualization/Source/KSVTKTrackTerminatorPainterBuilder.cxx)|—    |—    |`add_color`<br>`add_terminator`<br>`file`<br>`name`<br>`outfile`<br>`path`<br>`point_object`<br>`point_size`<br>`point_variable`<br>`terminator_object`<br>`terminator_variable`|*`KThreeVector`*<br>*`string`*<br>*`string`*<br>*`string`*<br>*`string`*<br>*`string`*<br>*`string`*<br>*`int`*<br>*`string`*<br>*`string`*<br>*`string`*|
