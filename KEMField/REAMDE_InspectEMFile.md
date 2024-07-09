InspectEMFile output
====================

The `InspectEMFile` application allows to inspect the content of a KEMField checkpoint file (`.kbd` file) and list its data keys.
It can be used to compare saved hashes with the ones that KEMField computed during initialization. For a file to load correctly, the hashes must match the ones from KEMField's current configuration.

## Structure

Main output groups are always surrounded by `<Key>...</Key>` and follow the structure:
```
<Key>
    <0>   ObjectName
    <1>   ClassName
    <2>   ObjectHash
    <3>   NumLabels
    <4+>  Labels
    <N-2> ObjectLocation
    <N-1> ObjectSize
</Key>
```
with *N* lines in total. Lines 4+ contain labels that depend on the object type; the last two lines always contain object info. If `NumLabels == 0`, no labels are included.

### SurfaceContainer

Contains 2 labels:
```
    <4>   ClassName
    <5>   ObjectHash
```

Both values should be the same as lines 1+2.

The `ObjectName` field contains the geometry/shape hash as well:
```
    <0>SurfaceContainer_<GeometryHash><\0>
```

The geometry/shape hash is compared to the current geometry when a checkpoint file is read.

### ResidualThreshold

Contains 4 labels:
```
    <4>   ClassName
    <5>   ObjectHash
    <6>   ShapeBoundaryHash
    <7>   ShapeHash
```

Both values should be the same as lines 1+2. The `ShapeHash` is the same as for the corresponding `SurfaceContainer`.

The `ObjectName` field contains the geometry/shape hash as well:
```
    <0>ResidualThreshold_<GeometryHash><\0>
```

The shape and shape+boundary hashes are compared to the current geometry when a checkpoint file is read.

### KFMElectrostaticTreeData

Does not contain any labels. Two hashes are contained in the `ObjectName` field, however:
```
    <0>KFMElectrostaticTreeData_<SolutionHash>_<ParameterHash><\0>
```

The solution and parameter hashes are compared to the current geometry when a checkpoint file is read.

### ZonalHarmonicContainer

Contains 3 labels:
```
    <4>   ClassName_<BasisType>
    <5>   ShapeBoundarySolutionHash
    <6>   ParameterHash
```

The `BasisType` is e.g. `KZHElectromagnet` or `KZHElectrostaticSurface`, depending on the underlying field.

The `ObjectName` field contains the two hashes as well:
```
    <0>ZonalHarmonicContainer_<BasisType>_<ShapeBoundarySolutionHash>_<ParameterHash><\0>
```

The solution and parameter hashes are compared to the current geometry when a checkpoint file is read.

## Example:

This file contains all three kinds of containers.

```
<Key>
    <0>(std::string)KFMElectrostaticTreeData_37492b0686a02fc60fa71be736ec087a_fa5d16569ceb5fbad38fb084beef5997<\0>
    <1>(std::string)KFMElectrostaticTreeData<\1>
    <2>(std::string)a795001111d37758453bb6f576e4318d<\2>
    <3>(unsigned int)0<\3>
    <4>(unsigned long)239985780<\4>
    <5>(unsigned long)187415886<\5>
<\Key>

<Key>
    <0>(std::string)ResidualThreshold_435bf6286a5939ba30af0c02a8788349<\0>
    <1>(std::string)ResidualThreshold<\1>
    <2>(std::string)1b61499dd6842d16b7669cd33d67407e<\2>
    <3>(unsigned int)4<\3>
    <4>(std::string)ResidualThreshold<\4>
    <5>(std::string)8a278dd547a275e848c992780eec6109<\5>
    <6>(std::string)2be86535e518dcb64ca91593b98169c7<\6>
    <7>(std::string)435bf6286a5939ba30af0c02a8788349<\7>
    <8>(unsigned long)427401926<\8>
    <9>(unsigned long)44<\9>
<\Key>

<Key>
    <0>(std::string)SurfaceContainer_435bf6286a5939ba30af0c02a8788349<\0>
    <1>(std::string)SurfaceContainer<\1>
    <2>(std::string)435bf6286a5939ba30af0c02a8788349<\2>
    <3>(unsigned int)2<\3>
    <4>(std::string)SurfaceContainer<\4>
    <5>(std::string)435bf6286a5939ba30af0c02a8788349<\5>
    <6>(unsigned long)427402155<\6>
    <7>(unsigned long)13241720<\7>
<\Key>
```
